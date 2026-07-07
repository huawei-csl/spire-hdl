"""The insert gate — the design DB's trust boundary.

A design enters ``designs/`` only through :func:`insert_design`, which (1) runs the slot's
**frozen verification** (Tier-0 CEC in S1) and rejects failures, (2) dedups by AAG structural
hash, (3) stamps intrinsic metrics (AIG nodes/depth) + a heavy-pipeline Yosys transistor count,
and (4) records provenance. Untrusted producers propose; this gate disposes.

Heavy spire imports (pyosys/aigverse via ``spire.helpers`` / ``spire.aig``) are deferred into the
function body so ``import spire.design_db`` stays dependency-light.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from spire.design_db.store import DesignDB, DesignDBError
from spire.design_db.verify import (CECInapplicable, SlotUnverified, VerificationFailed, cec_check)


@dataclass
class InsertResult:
    design_id: str
    deduped: bool
    metrics: Dict[str, Any] = field(default_factory=dict)


def _materialize(design: Any, tdir: Path) -> Path:
    """Accept a spire ``Component``/``Netlist``, a Verilog file path, or raw Verilog text; return a
    Verilog file inside ``tdir``. (Python design *files* go through ``_elaborate_python`` first —
    see ``_materialize_any``.)"""
    if hasattr(design, "_ports") or hasattr(design, "to_netlist"):   # a spire design — lower it
        from spire.design_db.keys import normalize
        text = normalize(design).to_verilog()
        target = tdir / "candidate.v"
        target.write_text(text if text.endswith("\n") else text + "\n")
        return target
    if isinstance(design, Path) or (isinstance(design, str) and "\n" not in design):
        src = Path(design)
        if not src.exists():
            raise DesignDBError(f"design file not found: {src}")
        target = tdir / ("candidate" + (src.suffix or ".v"))
        shutil.copyfile(src, target)
        return target
    text = str(design)
    target = tdir / "candidate.v"
    target.write_text(text if text.endswith("\n") else text + "\n")
    return target


def _local_import_closure(entry: Path) -> Dict[str, Path]:
    """The design's own helper modules: the transitive imports of ``entry`` whose files live under
    its project root (git root, else its directory) and outside any site-packages — i.e. local by
    containment. Stdlib and installed packages (spire included) resolve elsewhere and are never
    vendored. Returns ``{root-relative path: file}``, bounded at 64 files."""
    import ast
    import importlib.util
    entry = entry.resolve()
    root = next((p for p in entry.parents if (p / ".git").exists()), entry.parent)

    def _imported_names(f: Path) -> set:
        names = set()
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        return names

    def _local_file(name: str) -> Optional[Path]:
        try:
            origin = getattr(importlib.util.find_spec(name), "origin", None)
        except (ImportError, ValueError):
            return None
        if not origin or not origin.endswith(".py"):
            return None
        p = Path(origin).resolve()
        is_local = p.is_relative_to(root) and "site-packages" not in p.parts and p != entry
        return p if is_local else None

    closure: Dict[str, Path] = {}
    queue = [entry]
    sys.path.insert(0, str(entry.parent))                # names resolve as the design imports them
    try:
        while queue and len(closure) < 64:
            for name in sorted(_imported_names(queue.pop())):
                p = _local_file(name)
                if p and str(p.relative_to(root)) not in closure:
                    closure[str(p.relative_to(root))] = p
                    queue.append(p)
    finally:
        sys.path.remove(str(entry.parent))
    return closure


def _elaborate_python(py: Path, tdir: Path) -> tuple:
    """Elaborate a python design file — it must define ``build() -> Component/Netlist`` — into a
    Verilog file inside ``tdir``. The generated Verilog is what the gate verifies and what the DB
    stores as the canonical ``design.v``; the python is stored alongside as the *source* (correct
    by construction: the .v is its elaboration). Returns ``(verilog_path, python_source_info)``."""
    import runpy
    py = Path(py).resolve()
    if not py.exists():
        raise DesignDBError(f"design file not found: {py}")
    entry_dir = str(py.parent)
    sys.path.insert(0, entry_dir)                    # local helper imports resolve from the entry
    try:
        try:
            ns = runpy.run_path(str(py))
        except DesignDBError:
            raise
        except Exception as exc:
            raise DesignDBError(f"python design failed to execute: "
                                f"{type(exc).__name__}: {exc}") from exc
        build = ns.get("build")
        if not callable(build):
            raise DesignDBError(f"{py.name} must define build() -> Component/Netlist")
        try:
            module = build()
        except Exception as exc:
            raise DesignDBError(f"build() failed: {type(exc).__name__}: {exc}") from exc
    finally:
        sys.path.remove(entry_dir)
    from spire.design_db.keys import normalize
    text = normalize(module).to_verilog()
    target = tdir / "candidate.v"
    target.write_text(text if text.endswith("\n") else text + "\n")
    info = {"entry": py, "closure": _local_import_closure(py)}
    return target, info


def _materialize_any(design: Any, tdir: Path) -> tuple:
    """``_materialize`` + the python-design dispatch: a ``.py`` path is elaborated (build() →
    Verilog) and its source travels with the result. Returns ``(verilog_path, python_src|None)``."""
    if isinstance(design, (str, Path)) and "\n" not in str(design) \
            and str(design).endswith(".py"):
        return _elaborate_python(Path(design), tdir)
    return _materialize(design, tdir), None


def _candidate_aag(design_v: Path, workdir: Path) -> List[str]:
    """Convert a candidate to AAG via a **subprocess** yosys (dedup hash / ports / metrics).

    Deliberately not the in-process pyosys path: a pyosys ``log_error`` (e.g. ``write_aiger`` on
    an async-reset FF) hard-exits the host process — unacceptable for a gate fed arbitrary input.
    ``async2sync`` legalizes async-reset FFs into AIGER-expressible sync form first; any failure
    becomes a clean rejection.
    """
    if shutil.which("yosys") is None:
        raise DesignDBError("yosys not found on PATH — insert unavailable")
    out = workdir / "candidate.aag"
    script = "; ".join([
        f"read_verilog -sv {design_v.resolve()}",
        "hierarchy -auto-top",
        "proc",
        "synth -flatten",
        "async2sync",
        "dffunmap",
        "clean",
        "aigmap",
        f"write_aiger -ascii -symbols -no-startoffset {out}",
    ])
    proc = subprocess.run(["yosys", "-q", "-p", script], cwd=str(workdir),
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stdout + proc.stderr)[-600:]
        raise VerificationFailed(f"candidate could not be converted to an AIG:\n{tail}")
    return out.read_text().splitlines()


def _aag_port_names(aag_lines: List[str]) -> Dict[str, set]:
    """Base port names from the AAG symbol table (``i<n> a[3]`` → ``a``)."""
    import re
    head = aag_lines[0].split()
    n_in, n_latch, n_out, n_and = (int(x) for x in head[2:6])
    names: Dict[str, set] = {"input": set(), "output": set()}
    for line in aag_lines[1 + n_in + n_latch + n_out + n_and:]:
        if line == "c":
            break
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[0] or parts[0][0] not in "io":
            continue
        base = re.sub(r"\[\d+\]$", "", parts[1].strip())
        names["input" if parts[0][0] == "i" else "output"].add(base)
    return names


def _check_ports(aag_lines: List[str], ports: List[Dict[str, Any]]) -> None:
    """The candidate's interface must match the slot's port spec (clear error beats an abc mystery)."""
    got = _aag_port_names(aag_lines)
    want = {"input": {p["name"] for p in ports if p["dir"] == "input"},
            "output": {p["name"] for p in ports if p["dir"] == "output"}}
    if got != want:
        raise VerificationFailed(
            f"port mismatch vs the slot spec — candidate inputs={sorted(got['input'])} "
            f"outputs={sorted(got['output'])}, expected inputs={sorted(want['input'])} "
            f"outputs={sorted(want['output'])}")


def _aag_stats(aag_lines: List[str]) -> Dict[str, int]:
    """Intrinsic metrics straight from AAG lines: node count (A), latch count (L), and AND-depth
    (inputs/latches/consts are depth 0; AAG definitions are topologically ordered)."""
    head = aag_lines[0].split()
    _m, n_in, n_latch, n_out, n_and = (int(x) for x in head[1:6])
    start = 1 + n_in + n_latch + n_out
    depth: Dict[int, int] = {}
    max_depth = 0
    for line in aag_lines[start:start + n_and]:
        parts = line.split()
        if len(parts) != 3:
            continue
        lhs, a, b = (int(p) for p in parts)
        d = 1 + max(depth.get(a // 2, 0), depth.get(b // 2, 0))
        depth[lhs // 2] = d
        max_depth = max(max_depth, d)
    return {"aig_nodes": n_and, "aig_depth": max_depth, "aig_latches": n_latch}


def _resolve_verification(d: DesignDB, spec_key: str, spec: Dict[str, Any],
                          budget_s: Optional[float]) -> tuple:
    """(verification dict, method, budget) for a slot's frozen oracle — or raise. Shared by the
    gate (``insert_design``) and the advisory check (``check_design``) so both apply exactly the
    same oracle."""
    verification = d.read_json(d.slot_dir(spec_key) / "verification.json", None)
    if verification is None:
        raise SlotUnverified(
            "slot has no verification set — choose one first: "
            "spire db set-verification --slot <key> [--cec | --auto | --stimulus <file>]")
    method = verification.get("method")
    if method not in ("cec", "sim"):
        raise DesignDBError(f"verification method {method!r} is not supported")
    if method == "cec" and spec.get("class") == "sequential":
        raise CECInapplicable("CEC is inapplicable to sequential slots (no register mapping)")
    if budget_s is not None:
        budget = float(budget_s)
    elif method == "cec":
        budget = float(verification.get("budget_s", 120.0))
    else:
        budget = float(verification.get("sim_budget_s", 300.0))
    return verification, method, budget


def _run_gate(d: DesignDB, spec_key: str, method: str, design_v: Path, tdir: Path,
              budget: float, db: Optional[str | Path]) -> None:
    """Run the frozen oracle against one candidate (raises on anything but PASS)."""
    if method == "cec":
        cec_check(design_v, d.slot_dir(spec_key) / "golden.v", tdir / "cec", budget_s=budget)
    else:
        from spire.design_db.verify_sim import run_frozen_tb
        run_frozen_tb(spec_key, design_v, tdir / "sim", db=db, budget_s=budget)


def check_design(spec_key: str, design: Any, *, db: Optional[str | Path] = None,
                 budget_s: Optional[float] = None) -> Dict[str, Any]:
    """Advisory verification: run the slot's frozen oracle against ``design`` **without admitting
    or writing anything** — the same check ``insert_design`` gates on. Returns
    ``{"verdict": "PASS", "method": …}``; raises the same ``VerificationError`` subclasses (and a
    port/`SlotUnverified` error) on failure. This is the read-only sibling of ``insert_design``."""
    d = DesignDB.open(db)
    spec = d.read_json(d.slot_dir(spec_key) / "spec.json", None)
    if spec is None:
        raise DesignDBError(f"unknown slot {spec_key[:12]}… — register it first")
    _verification, method, budget = _resolve_verification(d, spec_key, spec, budget_s)
    with tempfile.TemporaryDirectory(prefix="spire_ddb_chk_") as td:
        tdir = Path(td)
        design_v, _python_src = _materialize_any(design, tdir)
        aag_lines = _candidate_aag(design_v, tdir)
        _check_ports(aag_lines, spec.get("ports", []))
        _run_gate(d, spec_key, method, design_v, tdir, budget, db)
    return {"verdict": "PASS", "method": method}


def insert_design(spec_key: str, design: Any, *, source: str,
                  db: Optional[str | Path] = None, budget_s: Optional[float] = None,
                  python_copy: Optional[str | Path] = None,
                  provenance: Optional[Dict[str, Any]] = None) -> InsertResult:
    """Verify ``design`` against the slot's frozen verification and, if correct, admit it.

    ``design`` may be a **python design file** (``*.py`` defining ``build() -> Component/Netlist``
    — the primary, spire-first path: it is elaborated here, the generated Verilog becomes the
    canonical ``design.v``, and the source + its project-local import closure are stored with the
    design, correct by construction), a spire ``Component``/``Netlist`` object (lowered
    internally), a Verilog file path, or raw Verilog text. Verilog is the DB's intermediate
    representation: whatever the input, all downstream processing (gate, dedup, metrics, splice)
    runs on ``design.v``.

    ``python_copy`` attaches a .py as *provenance only* (tagged ``kind: copied``, not validated) —
    used by ``seed_original`` to carry the slot's starting point; prefer inserting a ``.py``.

    Raises ``SlotUnverified`` (no frozen verification), ``VerificationFailed`` (rejected),
    ``CECTimeout`` (budget exceeded — options message included), ``CECInapplicable`` /
    ``DesignDBError`` (misuse or tooling problems). Returns :class:`InsertResult`;
    ``deduped=True`` means a structurally identical design was already stored.
    """
    d = DesignDB.open(db)
    slot = d.slot_dir(spec_key)
    spec = d.read_json(slot / "spec.json", None)
    if spec is None:
        raise DesignDBError(f"unknown slot {spec_key[:12]}… — register it first")
    verification, method, budget = _resolve_verification(d, spec_key, spec, budget_s)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    with tempfile.TemporaryDirectory(prefix="spire_ddb_") as td:
        tdir = Path(td)
        design_v, python_src = _materialize_any(design, tdir)

        # Structural dedup key (also feeds the intrinsic metrics + the port check).
        aag_lines = _candidate_aag(design_v, tdir)
        _check_ports(aag_lines, spec.get("ports", []))
        struct_hash = hashlib.sha256("\n".join(aag_lines).encode("utf-8")).hexdigest()

        index = d.derive_index(spec_key)             # designs/ is the source of truth
        for design_id, entry in index.items():
            if entry.get("struct_hash") == struct_hash:
                return InsertResult(design_id, True, entry.get("metrics", {}))

        # The gate: run the frozen verification (raises on anything but PASS).
        _run_gate(d, spec_key, method, design_v, tdir, budget, db)

        # Self-describing measurement systems: each block carries raw `metrics` + an `objectives`
        # map (objective → own field, or a `sibling.field` borrow). The transistor system borrows
        # the AIG depth for its delay axis rather than duplicating it.
        metrics: Dict[str, Any] = {
            "aig": {"metrics": _aag_stats(aag_lines),
                    "objectives": {"area": "aig_nodes", "delay": "aig_depth"}},
        }
        try:
            from spire.helpers import extract_yosys_heavy_metrics_from_verilog
            heavy = extract_yosys_heavy_metrics_from_verilog(design_v.read_text().splitlines())
            metrics["transistors"] = {
                "metrics": {"transistors_heavy": int(heavy["estimated_num_transistors"])},
                "objectives": {"area": "transistors_heavy", "delay": "aig.aig_depth"}}
        except Exception as exc:  # metric enrichment must never block a correct insert
            metrics["notes"] = f"transistor estimate unavailable: {exc}"

        design_id = f"{source}:{struct_hash[:10]}"
        final_dir = slot / "designs" / design_id
        tmp_dir = slot / "designs" / (design_id + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        shutil.copyfile(design_v, tmp_dir / "design.v")
        (tmp_dir / "design.aag").write_text("\n".join(aag_lines) + "\n")   # precomputed splice input
        prov = {"schema": 1, "source": source, "created": now, "struct_hash": struct_hash,
                "verification": {"tier": verification.get("tier", 0), "method": method,
                                 "verdict": "PASS", "budget_s": budget}}
        if python_src is not None:            # a .py insert: design.v IS its elaboration
            shutil.copyfile(python_src["entry"], tmp_dir / "design.py")
            for rel, srcp in sorted(python_src["closure"].items()):
                dst = tmp_dir / "source" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(srcp, dst)
            prov["python_source"] = {"kind": "elaborated", "entry": "design.py",
                                     "local_modules": sorted(python_src["closure"])}
        elif python_copy is not None:         # provenance copy (seed: the slot's starting point)
            py = Path(python_copy)
            if py.exists():
                shutil.copyfile(py, tmp_dir / "design.py")
                prov["python_source"] = {"kind": "copied", "entry": "design.py"}
        if provenance:
            prov.update(provenance)
        (tmp_dir / "metrics.json").write_text(_json(metrics))
        (tmp_dir / "provenance.json").write_text(_json(prov))
        if final_dir.exists():                      # lost a race — treat as dedup
            shutil.rmtree(tmp_dir)
            return InsertResult(design_id, True, metrics)
        os.replace(tmp_dir, final_dir)              # the admit: the only write that matters

    d.read_index(spec_key)                          # refresh the index.json cache (best effort)
    return InsertResult(design_id, False, metrics)


def seed_original(spec_key: str, *, db: Optional[str | Path] = None,
                  budget_s: Optional[float] = None) -> InsertResult:
    """Insert the slot's own golden as the baseline candidate (``source="original"``).

    Gives selection a *floor* (argmin can never pick worse than the original) and gives
    reports/Pareto a baseline to compare against. Idempotent via structural dedup. When the slot
    has a captured ``starting_point.py`` (decorator-registered slots), it is stored with the
    seeded design as its python source (provenance copy — the origin of the golden).
    """
    d = DesignDB.open(db)
    golden = d.slot_dir(spec_key) / "golden.v"
    if not golden.exists():
        raise DesignDBError(f"slot {spec_key[:12]}… has no golden.v — register it first")
    starting_point = d.slot_dir(spec_key) / "starting_point.py"
    return insert_design(spec_key, golden, source="original", db=db, budget_s=budget_s,
                         python_copy=starting_point if starting_point.exists() else None)


def _json(obj: Any) -> str:
    import json
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"
