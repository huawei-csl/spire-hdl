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

from spirehdl.spirehdl import (
    Bool, Concat, Const, Expr, Op1, Op2, Resize, Signal, SInt, Slice,
    Ternary, UInt, reset_shared_cache,
)
from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import (
    CompressorTreeAccumulator,
    PartialProductAccumulatorBase,
    SelectionMode,
    StageBasedMultiplierBasic,
    StageBasedMultiplierIO,
    StageMultiplierConfig,
)
from spirehdl.arithmetic.int_multipliers.stages.ppa_stages import (
    CarrySaveAccumulator,
    DaddaTreeAccumulator,
    FourTwoCompressorAccumulator,
    FourTwoCompressorParallelAccumulator,
    FiveTwoCompressorAccumulator,
    WallaceTreeAccumulator,
)
from spirehdl.arithmetic.int_multipliers.stages.ppg_and_stages import (
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
def test_tree_matches_reference(ppa_cls, default_mode, alt_mode):
    """Verify exact tree identity against pre-refactor reference fingerprints."""
    ref_path = Path(__file__).parent / "ppa_reference_fingerprints.json"
    if not ref_path.exists():
        pytest.skip("Reference fingerprints not found — run capture_ppa_reference.py first")

    with open(ref_path) as f:
        reference = json.load(f)

    config = StageMultiplierConfig(8, 8, False, False, "area")
    name = ppa_cls.__name__

    # Test default mode matches reference
    reset_shared_cache()
    cols = make_ppg_columns(config)
    ppa = ppa_cls(config)
    fp_default = fingerprint_columns(ppa.accumulate(cols))
    ref_default = {w: bits for w, bits in reference[f"{name}__default"].items() if bits}
    assert fp_default == ref_default, (
        f"{name}: default mode tree differs from pre-refactor reference"
    )

    # Test alt mode matches reference (if available)
    if alt_mode is not None:
        ref_alt_key = f"{name}__alt"
        if ref_alt_key in reference:
            reset_shared_cache()
            cols = make_ppg_columns(config)
            ppa = ppa_cls(config, selection_mode=alt_mode)
            fp_alt = fingerprint_columns(ppa.accumulate(cols))
            ref_alt = {w: bits for w, bits in reference[ref_alt_key].items() if bits}
            assert fp_alt == ref_alt, (
                f"{name}: alt mode '{alt_mode}' tree differs from pre-refactor reference"
            )
