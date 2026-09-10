"""Lean tier: a slot's canonical spec is equivalence to its golden; designs and spec layers are
admitted with kernel-checked proofs. The package never writes a proof.

Freeze (`set-verification --lean`): generate Interface / GoldenCircuit / Spec from the golden model
captured at registration, compile once, write verification.json. No author, no proof.

Spec layer (`add-spec NAME Spec.lean SpecProof.lean`): a nicer formulation `NAME.Correct`, admitted with
`NAME.equiv : ∀ e, NAME.Correct e ↔ Spec.Correct e`. Append-only; earlier proofs stay valid because
`Spec.Correct` never changes.

Gate (`insert --proof`): the candidate becomes `D<hash>_Circuit.lean`; the submitted `D<hash>_Proof.lean`
must prove `implements : Spec.Correct D<hash>_Circuit.eval`. It may import any layer and any admitted
design (delta proofs). The checker audits the final statement and the transitive axioms only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from spire.design_db.store import DesignDB, DesignDBError, now_ts
from spire.design_db.verify import VERIFICATION_SCHEMA, ProofFailed, ProofRejected, SpecRejected, VerificationFailed

LEAN_TIER = 3
DEFAULT_LAKE_BUDGET_S = 600.0
FROZEN = ("SpireSemantics.lean", "Interface.lean", "GoldenCircuit.lean", "Spec.lean")
_NAME = re.compile(r"^[A-Z][A-Za-z0-9_]*$")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(src: Any) -> str:
    return str(src) if "\n" in str(src) else Path(src).read_text()


def _slot(d: DesignDB, spec_key: str) -> tuple:
    spec = d.read_json(d.slot_dir(spec_key) / "spec.json", None)
    if spec is None:
        raise DesignDBError(f"unknown slot {spec_key[:12]}… — register it first")
    return d.slot_dir(spec_key), spec


def _publish(dst: Path, files: Dict[str, str]) -> None:
    """Write files read-only; a directory is replaced atomically."""
    dst.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (dst / name).write_text(text)
        (dst / name).chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _project(workdir: Path, files: Dict[str, str]) -> Path:
    from spire.design_db.lean.bridge import write_lean_project
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    write_lean_project(workdir, [n[:-5] for n in files if n.endswith(".lean")])
    for name, text in files.items():
        (workdir / name).write_text(text)
    return workdir


def _build(workdir: Path, target: str, budget_s: float, err, what: str) -> str:
    from spire.design_db.lean.bridge import LeanBuildError, lake_build
    try:
        return lake_build(workdir, target, timeout=budget_s)
    except LeanBuildError as exc:
        raise err(f"{what}:\n{str(exc)[-2000:]}") from None
    except subprocess.TimeoutExpired:
        raise err(f"{what}: lake build exceeded {budget_s:g} s") from None


def _check(workdir: Path, module: str, theorem: str, expected: str, err) -> Dict[str, Any]:
    from spire.design_db.lean.bridge import ProofCheckError, TheoremCheck, check_theorems
    try:
        return check_theorems(workdir, [TheoremCheck(module=module, theorem=theorem, expected_type=expected)])
    except ProofCheckError as exc:
        raise err(str(exc)[-2000:]) from None


# --- registration capture + freeze -------------------------------------------------------------

def golden_lean_capture(netlist, ports) -> Optional[Dict[str, Any]]:
    """Called by register_slot: the golden's Lean circuit model, or None when the translator does not
    cover it (the slot then cannot be Lean-gated)."""
    from spire.design_db.lean.bridge import InterfaceMismatch, LeanExportOptions, collect_signed_arith_shapes, netlist_to_lean
    try:
        circuit = netlist_to_lean(netlist, options=LeanExportOptions(namespace="GoldenCircuit", source_name="golden", ports=ports))
    except (NotImplementedError, InterfaceMismatch, ValueError):
        return None
    return {"circuit": circuit, "shapes": [list(s) for s in sorted(collect_signed_arith_shapes(netlist))]}


def freeze_lean_verification(spec_key: str, *, author: Optional[str] = None,
                             lake_budget_s: float = DEFAULT_LAKE_BUDGET_S, dry_run: bool = False,
                             keep_dir: Optional[Path] = None, db: Optional[str | Path] = None) -> Dict[str, Any]:
    """Freeze the Lean oracle: Spec = equivalence to the golden. Compiles the four frozen files once;
    writes nothing on failure or with `dry_run`; `keep_dir` keeps the project."""
    from spire.design_db.lean.bridge import LEAN_TOOLCHAIN, SEMANTICS_FILE, default_spec_lean, interface_to_lean
    d = DesignDB.open(db)
    slot, spec = _slot(d, spec_key)
    if spec.get("class") != "combinational":
        raise DesignDBError("the Lean tier is combinational-only")
    golden = d.read_json(slot / "golden_lean.json", None)
    if golden is None:
        raise DesignDBError("the golden was not translatable to Lean at registration — this slot cannot be Lean-gated")
    files = {"SpireSemantics.lean": SEMANTICS_FILE.read_text(), "Interface.lean": interface_to_lean(spec["ports"]),
             "GoldenCircuit.lean": golden["circuit"], "Spec.lean": default_spec_lean()}
    with tempfile.TemporaryDirectory(prefix="spire_ddb_frz_") as td:
        work = _project(Path(keep_dir) if keep_dir is not None else Path(td) / "lean", files)
        _build(work, "Spec", lake_budget_s, DesignDBError, "frozen Lean files do not compile")
    verification = {"schema": VERIFICATION_SCHEMA, "tier": LEAN_TIER, "method": "lean",
                    "lake_budget_s": float(lake_budget_s), "toolchain": LEAN_TOOLCHAIN,
                    "spec_sha": _sha(files["Spec.lean"]), "interface_sha": _sha(files["Interface.lean"]),
                    "semantics_sha": _sha(files["SpireSemantics.lean"]), "author": author, "frozen": now_ts()}
    if dry_run:
        return {**verification, "dry_run": True}
    tmp, final = slot / "lean.tmp", slot / "lean"
    shutil.rmtree(tmp, ignore_errors=True)
    _publish(tmp, files)
    (tmp / "specs").mkdir()
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    d.write_json(slot / "verification.json", verification)
    return verification


# --- context: everything a proof may import ------------------------------------------------------

def _context(d: DesignDB, spec_key: str) -> Dict[str, str]:
    """Frozen files + admitted spec layers + admitted designs' circuit/proof modules."""
    lean = d.slot_dir(spec_key) / "lean"
    if not all((lean / n).exists() for n in FROZEN):
        raise DesignDBError("slot has no frozen Lean oracle — run set-verification --lean first")
    files = {n: (lean / n).read_text() for n in FROZEN}
    for p in sorted((lean / "specs").glob("*.lean")):
        files[p.name] = p.read_text()
    for p in sorted((d.slot_dir(spec_key) / "designs").glob("*/lean/D*_*.lean")):
        files[p.name] = p.read_text()
    return files


def spec_layers(d: DesignDB, spec_key: str) -> List[str]:
    return sorted(p.stem for p in (d.slot_dir(spec_key) / "lean" / "specs").glob("*.json"))


# --- spec layers ---------------------------------------------------------------------------------

def add_spec(spec_key: str, name: str, spec_lean: Any, proof_lean: Any, *, author: Optional[str] = None,
             lake_budget_s: float = DEFAULT_LAKE_BUDGET_S, dry_run: bool = False, keep_dir: Optional[Path] = None,
             db: Optional[str | Path] = None) -> Dict[str, Any]:
    """Admit a spec layer `name`: `<name>.lean` defines `<name>.Correct`, `<name>Proof.lean` proves
    `<name>.equiv : ∀ e, <name>.Correct e ↔ Spec.Correct e`. Raises SpecRejected; append-only."""
    from spire.design_db.lean.bridge import RESERVED_MODULES, lean_imports
    d = DesignDB.open(db)
    slot, _spec = _slot(d, spec_key)
    ctx = _context(d, spec_key)
    if not _NAME.match(name) or name in RESERVED_MODULES or name.endswith("Proof") or re.match(r"^D[0-9a-f]{10}_", name):
        raise DesignDBError(f"invalid spec layer name {name!r} (capitalised identifier, not reserved)")
    if name in spec_layers(d, spec_key):
        raise DesignDBError(f"spec layer {name!r} already exists (layers are append-only)")
    files = {f"{name}.lean": _text(spec_lean), f"{name}Proof.lean": _text(proof_lean)}
    expected = f"∀ e, {name}.Correct e ↔ Spec.Correct e"
    with tempfile.TemporaryDirectory(prefix="spire_ddb_spec_") as td:
        work = _project(Path(keep_dir) if keep_dir is not None else Path(td) / "lean", {**ctx, **files})
        log = _build(work, f"{name}Proof", lake_budget_s, SpecRejected, f"spec layer {name}: equiv proof does not build")
        axioms = _check(work, f"{name}Proof", f"{name}.equiv", expected, SpecRejected)
    record = {"name": name, "author": author, "admitted": now_ts(), "statement": expected, "axioms": axioms,
              "depends_on": lean_imports(files[f"{name}Proof.lean"])}
    if dry_run:
        return {**record, "dry_run": True, "log": log}
    _publish(slot / "lean" / "specs", {**files, f"{name}.json": json.dumps(record, indent=2, sort_keys=True) + "\n"})
    return record


# --- gate ----------------------------------------------------------------------------------------

def run_lean_gate(spec_key: str, netlist, proof: Any, workdir: Path, *, hash10: str, budget_s: float,
                  db: Optional[str | Path] = None) -> Dict[str, Any]:
    """Prove one candidate against the frozen spec. `proof` None: write the workspace (context +
    D<hash>_Circuit.lean) and raise VerificationFailed telling the agent what to write. Returns
    {files, axioms, log, depends_on}; raises ProofFailed / ProofRejected / VerificationFailed."""
    from spire.design_db.lean.bridge import (FINAL_THEOREM, InterfaceMismatch, LeanExportOptions, design_modules,
                                             lean_imports, netlist_to_lean)
    d = DesignDB.open(db)
    _slot_dir, spec = _slot(d, spec_key)
    ctx = _context(d, spec_key)
    circuit_mod, proof_mod = design_modules(hash10)
    try:
        circuit = netlist_to_lean(netlist, options=LeanExportOptions(namespace=circuit_mod, source_name=hash10, ports=spec["ports"]))
    except (NotImplementedError, InterfaceMismatch, ValueError) as exc:
        raise VerificationFailed(f"design not translatable to Lean: {exc}") from None
    files = {f"{circuit_mod}.lean": circuit}
    if proof is None:
        _project(workdir, {**ctx, **files})
        raise VerificationFailed(f"no proof given; workspace written to {workdir} — write {proof_mod}.lean with "
                                 f"`theorem {FINAL_THEOREM} : Spec.Correct {circuit_mod}.eval` (layers: "
                                 f"{spec_layers(d, spec_key) or 'none'})")
    files[f"{proof_mod}.lean"] = _text(proof)
    work = _project(workdir, {**ctx, **files})
    log = _build(work, proof_mod, budget_s, ProofFailed, "proof does not check")
    axioms = _check(work, proof_mod, f"{proof_mod}.{FINAL_THEOREM}", f"Spec.Correct {circuit_mod}.eval", ProofRejected)
    return {"files": files, "axioms": axioms, "log": log, "depends_on": lean_imports(files[f"{proof_mod}.lean"])}


def golden_evidence() -> Dict[str, Any]:
    """The golden satisfies Spec by definition; seeding it stores no proof."""
    return {"files": {}, "axioms": {}, "log": "golden: Spec.Correct GoldenCircuit.eval holds by definition\n", "depends_on": []}
