"""Verification seam: circuit-class detection, the frozen-verification format, and Tier-0 CEC.

Fail-and-choose, no auto-fallback: a CEC timeout raises :class:`CECTimeout` carrying the options
message; the *caller* (human or orchestrator) explicitly picks the next rung (bigger ``--budget``,
the Tier-1 auto sim harness, or authored stimulus — both arrive in S3). The class check is a
guardrail + default-picker, not a router: CEC is refused for sequential slots (no register mapping).

Tier-0 CEC mirrors rtlscout's ``core/equivalence.py``: both sides are synthesized to flattened BLIF
with yosys, then compared with ``yosys-abc``'s ``cec`` command.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CEC_BUDGET_S = 120.0
VERIFICATION_SCHEMA = 1


class VerificationError(Exception):
    """Base class for verification-gate failures."""


class VerificationFailed(VerificationError):
    """The candidate is not correct w.r.t. the slot's frozen verification (rejected)."""


class CECTimeout(VerificationError):
    """CEC blew past its budget. Clean failure — the caller chooses the next rung."""


class CECInapplicable(VerificationError):
    """CEC requested for a slot it cannot check (sequential — no register mapping)."""


class SlotUnverified(VerificationError):
    """The slot has no frozen verification; inserts are refused until one is frozen."""


def timeout_options_message(budget_s: float) -> str:
    return (f"CEC timed out after {budget_s:g} s. "
            f"Options: --budget <t> | --auto (Tier-1 sim harness, S3) | --stimulus <file> (S3)")


def detect_class(module: Any) -> str:
    """``"combinational"`` or ``"sequential"`` — by presence of register signals.

    Registers created via ``Netlist.reg(...)`` carry ``kind == "reg"``. A module with a clock port
    but no registers is still combinational for verification purposes.
    """
    signals = list(getattr(module, "_signals", [])) + list(getattr(module, "_ports", []))
    return "sequential" if any(getattr(s, "kind", None) == "reg" for s in signals) else "combinational"


def default_verification(circuit_class: str) -> Optional[Dict[str, Any]]:
    """The verification a slot gets at registration: combinational → Tier-0 CEC; sequential → none
    (unverified until a sim-tier verification is frozen — S3)."""
    if circuit_class == "combinational":
        return {"schema": VERIFICATION_SCHEMA, "tier": 0, "method": "cec",
                "budget_s": DEFAULT_CEC_BUDGET_S}
    return None


# --- Tier-0 CEC -----------------------------------------------------------------------------


def _synth_to_blif(src: Path, out_blif: Path, cwd: Path, budget_s: float) -> None:
    """Synthesize one Verilog file to flattened BLIF (mirrors rtlscout ``_synth_to_blif``)."""
    script = "; ".join([
        f"read_verilog -sv {src}",
        "hierarchy -auto-top",
        "proc", "opt", "techmap", "opt",
        "synth -flatten",
        "async2sync", "dffunmap",
        "clean -purge",
        f"write_blif {out_blif}",
    ])
    try:
        proc = subprocess.run(["yosys", "-q", "-p", script], cwd=str(cwd),
                              capture_output=True, text=True, timeout=budget_s)
    except subprocess.TimeoutExpired:
        raise CECTimeout(timeout_options_message(budget_s)) from None
    if proc.returncode != 0 or not out_blif.exists():
        tail = (proc.stdout + proc.stderr)[-800:]
        raise VerificationError(f"yosys synth failed for {Path(src).name}:\n{tail}")


def cec_check(design_v: Path, golden_v: Path, workdir: Path, *,
              budget_s: float = DEFAULT_CEC_BUDGET_S) -> None:
    """Combinational equivalence of ``design_v`` vs ``golden_v``; raises on any non-PASS outcome.

    Raises ``VerificationFailed`` (not equivalent), ``CECTimeout`` (budget exceeded — options
    message included), or ``VerificationError`` (tooling problems).
    """
    if shutil.which("yosys") is None or shutil.which("yosys-abc") is None:
        raise VerificationError("yosys / yosys-abc not found on PATH — CEC unavailable")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    design_blif = workdir / "design.blif"
    golden_blif = workdir / "golden.blif"
    _synth_to_blif(Path(design_v).resolve(), design_blif, workdir, budget_s)
    _synth_to_blif(Path(golden_v).resolve(), golden_blif, workdir, budget_s)
    try:
        proc = subprocess.run(["yosys-abc", "-c", f"cec {golden_blif} {design_blif}; print_stats -S;"],
                              cwd=str(workdir), capture_output=True, text=True, timeout=budget_s)
    except subprocess.TimeoutExpired:
        raise CECTimeout(timeout_options_message(budget_s)) from None
    out = proc.stdout + proc.stderr
    upper = out.upper()
    # Order matters: "NOT EQUIVALENT" must never be misread as the equivalent case.
    if "NOT EQUIVALENT" in upper:
        raise VerificationFailed("candidate is NOT equivalent to the slot golden (CEC)\n" + out[-600:])
    if "EQUIVALENT" in upper:
        return
    raise VerificationError("could not parse yosys-abc cec verdict:\n" + out[-600:])
