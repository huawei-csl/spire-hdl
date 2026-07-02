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
    Verilog file inside ``tdir``."""
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


def insert_design(spec_key: str, design: Any, *, source: str,
                  db: Optional[str | Path] = None, design_py: Optional[str | Path] = None,
                  budget_s: Optional[float] = None,
                  provenance: Optional[Dict[str, Any]] = None) -> InsertResult:
    """Verify ``design`` against the slot's frozen verification and, if correct, admit it.

    ``design`` may be a spire ``Component``/``Netlist`` (lowered to Verilog internally), a Verilog
    file path, or raw Verilog text.

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
    verification = d.read_json(slot / "verification.json", None)
    if verification is None:
        raise SlotUnverified(
            "slot has no frozen verification (sim tiers arrive in S3) — options: "
            "spire db verify --slot <key> --auto | --stimulus <file>")
    if verification.get("method") != "cec":
        raise DesignDBError(f"verification method {verification.get('method')!r} is not supported "
                            f"yet (S1 implements Tier-0 cec)")
    if spec.get("class") == "sequential":
        raise CECInapplicable("CEC is inapplicable to sequential slots (no register mapping)")
    budget = float(budget_s) if budget_s is not None else float(verification.get("budget_s", 120.0))
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    with tempfile.TemporaryDirectory(prefix="spire_ddb_") as td:
        tdir = Path(td)
        design_v = _materialize(design, tdir)

        # Structural dedup key (also feeds the intrinsic metrics) — heavy import deferred.
        from spire.aig.aig_yosys import verilog_to_aag_lines_via_yosys
        aag_lines = verilog_to_aag_lines_via_yosys(str(design_v))
        _check_ports(aag_lines, spec.get("ports", []))
        struct_hash = hashlib.sha256("\n".join(aag_lines).encode("utf-8")).hexdigest()

        index = d.read_json(slot / "index.json", {})
        for design_id, entry in index.items():
            if entry.get("struct_hash") == struct_hash:
                return InsertResult(design_id, True, entry.get("metrics", {}))

        # The gate: run the frozen verification (raises on anything but PASS).
        cec_check(design_v, slot / "golden.v", tdir / "cec", budget_s=budget)

        metrics: Dict[str, Any] = {"intrinsic": _aag_stats(aag_lines)}
        try:
            from spire.helpers import extract_yosys_heavy_metrics_from_verilog
            heavy = extract_yosys_heavy_metrics_from_verilog(design_v.read_text().splitlines())
            metrics["transistors_heavy"] = int(heavy["estimated_num_transistors"])
        except Exception as exc:  # metric enrichment must never block a correct insert
            metrics["transistors_heavy"] = None
            metrics["notes"] = f"transistor estimate unavailable: {exc}"

        design_id = f"{source}:{struct_hash[:10]}"
        final_dir = slot / "designs" / design_id
        tmp_dir = slot / "designs" / (design_id + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        shutil.copyfile(design_v, tmp_dir / "design.v")
        (tmp_dir / "design.aag").write_text("\n".join(aag_lines) + "\n")   # precomputed splice input
        if design_py is not None:
            py = Path(design_py)
            (tmp_dir / "design.py").write_text(py.read_text() if py.exists() else str(design_py))
        prov = {"schema": 1, "source": source, "created": now,
                "verification": {"tier": verification.get("tier", 0), "method": "cec",
                                 "verdict": "PASS", "budget_s": budget}}
        if provenance:
            prov.update(provenance)
        (tmp_dir / "metrics.json").write_text(_json(metrics))
        (tmp_dir / "provenance.json").write_text(_json(prov))
        if final_dir.exists():                      # lost a race — treat as dedup
            shutil.rmtree(tmp_dir)
            return InsertResult(design_id, True, metrics)
        os.replace(tmp_dir, final_dir)

    index[design_id] = {"struct_hash": struct_hash, "source": source, "created": now,
                        "metrics": metrics}
    d.write_json(slot / "index.json", index)
    d.refresh_manifest_counts(spec_key, len(index))
    return InsertResult(design_id, False, metrics)


def seed_original(spec_key: str, *, db: Optional[str | Path] = None,
                  budget_s: Optional[float] = None) -> InsertResult:
    """Insert the slot's own golden as the baseline candidate (``source="original"``).

    Gives selection a *floor* (argmin can never pick worse than the original) and gives
    reports/Pareto a baseline to compare against. Idempotent via structural dedup.
    """
    d = DesignDB.open(db)
    golden = d.slot_dir(spec_key) / "golden.v"
    if not golden.exists():
        raise DesignDBError(f"slot {spec_key[:12]}… has no golden.v — register it first")
    return insert_design(spec_key, golden, source="original", db=db, budget_s=budget_s)


def _json(obj: Any) -> str:
    import json
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"
