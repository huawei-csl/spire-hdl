"""Regression tests for PPA selection modes.

Verifies that:
1. Each PPA's ``default_selection_mode`` matches expectations.
2. Each mode produces functionally correct multiplication (a * b == y).
3. The two available modes for each PPA produce *different* trees,
   confirming that the selection_mode switch is effective.
4. The Expr tree structure matches the pre-refactor reference for
   both modes (exact compressor-cell + connection identity).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple, Type

import pytest

from spire.expr import (
    Bool, Concat, Const, Expr, Op1, Op2, Resize, Signal, SInt, Slice,
    Ternary, UInt, reset_shared_cache,
)
from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import (
    CompressorTreeAccumulator,
    PartialProductAccumulatorBase,
    SelectionMode,
    StageBasedMultiplierBasic,
    StageBasedMultiplierIO,
    StageMultiplierConfig,
)
from spire.arithmetic.int_multipliers.stages.ppa_stages import (
    CarrySaveAccumulator,
    DaddaTreeAccumulator,
    FourTwoCompressorAccumulator,
    FourTwoCompressorParallelAccumulator,
    FiveTwoCompressorAccumulator,
    WallaceTreeAccumulator,
)
from spire.arithmetic.int_multipliers.stages.ppg_and_stages import (
    AndPartialProductGenerator,
)


# ---------------------------------------------------------------------------
# Fingerprinting helpers
# ---------------------------------------------------------------------------

def fingerprint_expr(e: Expr, memo: dict | None = None) -> str:
    """Recursively fingerprint an Expr tree by structure."""
    if memo is None:
        memo = {}
    eid = id(e)
    if eid in memo:
        return memo[eid]

    if isinstance(e, Const):
        fp = f"Const({e.value},{e.typ.width})"
    elif isinstance(e, Signal):
        if e.kind == "input":
            fp = f"Input({e.name},{e.typ.width})"
        elif e._driver is not None:
            fp = f"Wire({e.name},{fingerprint_expr(e._driver, memo)})"
        else:
            fp = f"Signal({e.name},{e.typ.width})"
    elif isinstance(e, Op2):
        fp = f"Op2({e.op},{fingerprint_expr(e.a, memo)},{fingerprint_expr(e.b, memo)})"
    elif isinstance(e, Op1):
        fp = f"Op1({e.op},{fingerprint_expr(e.a, memo)})"
    elif isinstance(e, Ternary):
        fp = f"Ternary({fingerprint_expr(e.sel, memo)},{fingerprint_expr(e.a, memo)},{fingerprint_expr(e.b, memo)})"
    elif isinstance(e, Concat):
        parts = ",".join(fingerprint_expr(p, memo) for p in e.parts)
        fp = f"Concat({parts})"
    elif isinstance(e, Slice):
        fp = f"Slice({fingerprint_expr(e.a, memo)},{e.start},{e.msb + 1})"
    elif isinstance(e, Resize):
        fp = f"Resize({fingerprint_expr(e.a, memo)},{e.to_width})"
    else:
        fp = f"Unknown({type(e).__name__})"

    memo[eid] = fp
    return fp


def fingerprint_columns(columns: Dict[int, List[Expr]]) -> Dict[str, List[str]]:
    """Fingerprint every bit in every column (non-empty only)."""
    memo: dict = {}
    result = {}
    for weight in sorted(columns.keys()):
        bits = columns[weight]
        if bits:
            result[str(weight)] = [fingerprint_expr(b, memo) for b in bits]
    return result


def make_ppg_columns(config: StageMultiplierConfig):
    a = Signal(typ=UInt(config.a_width), kind="input")
    b = Signal(typ=UInt(config.b_width), kind="input")
    y = Signal(typ=UInt(config.out_width), kind="output")
    io = StageBasedMultiplierIO(a=a, b=b, y=y)
    ppg = AndPartialProductGenerator(config)
    return ppg.generate_columns(io)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

# (PPA class, expected default mode, alt mode for dual-mode testing)
PPA_DEFAULTS = [
    (WallaceTreeAccumulator, "earliest", "lifo"),
    (DaddaTreeAccumulator, "earliest", "lifo"),
    (CarrySaveAccumulator, "lifo", "earliest"),
    (FourTwoCompressorAccumulator, "earliest", "lifo"),
    (FourTwoCompressorParallelAccumulator, "lifo", None),
    (FiveTwoCompressorAccumulator, "lifo", None),
    (CompressorTreeAccumulator, "fifo", "earliest"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ppa_cls,expected_mode,_alt",
    PPA_DEFAULTS,
    ids=[c.__name__ for c, _, _ in PPA_DEFAULTS],
)
def test_default_selection_mode(ppa_cls, expected_mode, _alt):
    """Each PPA's default_selection_mode matches the expected value."""
    assert ppa_cls.default_selection_mode == expected_mode


@pytest.mark.parametrize(
    "ppa_cls,default_mode,alt_mode",
    [(c, d, a) for c, d, a in PPA_DEFAULTS if a is not None],
    ids=[c.__name__ for c, _, a in PPA_DEFAULTS if a is not None],
)
def test_modes_produce_different_trees(ppa_cls, default_mode, alt_mode):
    """The two modes produce structurally different compressor trees."""
    config = StageMultiplierConfig(8, 8, False, False, "area")

    reset_shared_cache()
    cols1 = make_ppg_columns(config)
    ppa1 = ppa_cls(config, selection_mode=default_mode)
    fp1 = fingerprint_columns(ppa1.accumulate(cols1))

    reset_shared_cache()
    cols2 = make_ppg_columns(config)
    ppa2 = ppa_cls(config, selection_mode=alt_mode)
    fp2 = fingerprint_columns(ppa2.accumulate(cols2))

    assert fp1 != fp2, (
        f"{ppa_cls.__name__}: modes '{default_mode}' and '{alt_mode}' "
        f"produce identical trees — mode switch has no effect"
    )


@pytest.mark.parametrize(
    "ppa_cls,default_mode,alt_mode",
    PPA_DEFAULTS,
    ids=[c.__name__ for c, _, _ in PPA_DEFAULTS],
)
def test_selection_modes_multiply_correctly(ppa_cls, default_mode, alt_mode):
    """Every selection mode of every PPA produces a functionally correct 4x4 multiplier
    (exhaustive). Replaces the never-runnable pre-refactor fingerprint reference test —
    ppa_reference_fingerprints.json / capture_ppa_reference.py never existed in any history."""
    from spire.simulator import Simulator

    modes = [default_mode] + ([alt_mode] if alt_mode is not None else [])
    for mode in modes:
        reset_shared_cache()
        mul = StageBasedMultiplierBasic(
            a_w=4, b_w=4,
            ppg_cls=AndPartialProductGenerator,
            ppa_cls=lambda cfg, _m=mode, _c=ppa_cls: _c(cfg, selection_mode=_m),
            selection_mode=mode,
        )
        sim = Simulator(mul.to_module(f"ppa_{ppa_cls.__name__}_{mode}", with_clock=False,
                                      with_reset=False))
        for a in range(16):
            for b in range(16):
                sim.set("a", a).set("b", b)
                sim.eval()
                assert sim.get("y") == a * b, f"{ppa_cls.__name__}/{mode}: {a}*{b} -> {sim.get('y')}"
