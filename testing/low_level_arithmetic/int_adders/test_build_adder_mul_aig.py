"""Round-trip tests for ``build_adder`` / ``build_multiplier`` through the
AIG bit-level simulator.

These tests were added after a bug where ``spire_aiger.visit_op2`` did
not sign-extend operands for ``+``/``-`` when the result width exceeded
the operand widths. The existing adder/multiplier tests exercised only the
SpireHDL ``Simulator`` (which carries HDLType metadata through the graph),
so the AIG path stayed silent even though downstream flows
(``sim_and_switch_count``, PPA sweeps) relied on it being correct.

The matrix below deliberately covers both axes that the bug lived on:

    build fn    × {use_operator=True, use_operator=False}
                × {unsigned, twos_complement}
                × {SpireHDL Simulator, AIG sim}

Adding to any of these alone would not have caught the original bug. All
four together would — and the ``test_signed_add_sext_in_aig`` focused case
is a minimal regression test that reproduces the exact zero-extension
failure in ``bv_add``.
"""
from __future__ import annotations

import pytest

from spire.arithmetic.int_arithmetic_config import (
    AdderConfig,
    MultiplierConfig,
    build_adder,
    build_multiplier,
)
from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    PPAOption,
    PPGOption,
    MultiplierOption,
    TwoInputAritEncodings,
)
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding
from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import (
    StageBasedMultiplierIO,
)
from spire.helpers import refactor_module_to_aig, run_vectors
from spire.expr import Signal, SInt, UInt, reset_shared_cache
from spire.component import Component


N_BITS = 8
N_VECS = 64


# ---------------------------------------------------------------------------
# Small components that wrap ``build_adder`` / ``build_multiplier``
# ---------------------------------------------------------------------------

def _make_adder_component(adder_cfg: AdderConfig, signed: bool):
    a_t = SInt(N_BITS) if signed else UInt(N_BITS)
    b_t = SInt(N_BITS) if signed else UInt(N_BITS)
    # full_output_bit=True → extra bit for carry / sign
    y_t = UInt(N_BITS + 1)

    class AdderWrap(Component):
        def __init__(self):
            self.io = StageBasedMultiplierIO(
                a=Signal(typ=a_t, kind="input"),
                b=Signal(typ=b_t, kind="input"),
                y=Signal(typ=y_t, kind="output"),
            )
            y_val = build_adder(self.io.a, self.io.b, adder_cfg)
            # build_adder always returns UInt(N_BITS+1); assign directly
            self.io.y <<= y_val

        def elaborate(self):  # logic built in __init__; no-op to satisfy the abstract method
            pass

    return AdderWrap()


def _make_multiplier_component(mult_cfg: MultiplierConfig, signed: bool):
    a_t = SInt(N_BITS) if signed else UInt(N_BITS)
    b_t = SInt(N_BITS) if signed else UInt(N_BITS)
    y_t = UInt(2 * N_BITS)

    class MulWrap(Component):
        def __init__(self):
            self.io = StageBasedMultiplierIO(
                a=Signal(typ=a_t, kind="input"),
                b=Signal(typ=b_t, kind="input"),
                y=Signal(typ=y_t, kind="output"),
            )
            y_val = build_multiplier(self.io.a, self.io.b, mult_cfg)
            self.io.y <<= y_val

        def elaborate(self):  # logic built in __init__; no-op to satisfy the abstract method
            pass

    return MulWrap()


# ---------------------------------------------------------------------------
# Golden vector generation (independent of any HDL — pure integer model)
# ---------------------------------------------------------------------------

def _bits(v: int, w: int) -> int:
    return v & ((1 << w) - 1)


def _adder_vectors(signed: bool):
    # Handpicked corners + deterministic pseudo-random fill.
    # Focus on edge values that exercise sign extension.
    if signed:
        hi = (1 << (N_BITS - 1)) - 1
        lo = -(1 << (N_BITS - 1))
        corners = [
            (lo, lo),     # most-negative + most-negative (overflow into extra bit)
            (lo, -1),     # negative + negative
            (-1, 1),      # cancellation
            (-1, -1),     # both sign bits set
            (hi, hi),     # most-positive + most-positive
            (hi, 1),      # positive overflow into carry bit
            (lo, hi),     # -lo + hi = -1
        ]
    else:
        hi = (1 << N_BITS) - 1
        corners = [
            (0, 0), (0, hi), (hi, 0), (hi, hi), (1, hi), (hi, 1), (hi >> 1, (hi >> 1) + 1),
        ]

    vecs = []
    for a, b in corners:
        y = a + b
        vecs.append((
            f"corner_{a}_{b}",
            {"a": _bits(a, N_BITS), "b": _bits(b, N_BITS)},
            {"y": _bits(y, N_BITS + 1)},
        ))

    # Deterministic fill using a linear congruential walk for reproducibility.
    state = 0x12345
    for i in range(N_VECS - len(corners)):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        raw_a = state & ((1 << N_BITS) - 1)
        raw_b = (state >> N_BITS) & ((1 << N_BITS) - 1)
        if signed:
            sign = 1 << (N_BITS - 1)
            a = raw_a - (1 << N_BITS) if raw_a & sign else raw_a
            b = raw_b - (1 << N_BITS) if raw_b & sign else raw_b
        else:
            a, b = raw_a, raw_b
        y = a + b
        vecs.append((
            f"rand_{i}",
            {"a": _bits(a, N_BITS), "b": _bits(b, N_BITS)},
            {"y": _bits(y, N_BITS + 1)},
        ))
    return vecs


def _mul_vectors(signed: bool):
    if signed:
        hi = (1 << (N_BITS - 1)) - 1
        lo = -(1 << (N_BITS - 1))
        corners = [
            (0, 0), (1, 1), (-1, 1), (1, -1), (-1, -1),
            (lo, lo),        # negative * negative → large positive
            (lo, hi),        # negative * positive
            (hi, hi),        # max positive square
            (lo, -1),        # edge: most-negative * -1
        ]
    else:
        hi = (1 << N_BITS) - 1
        corners = [
            (0, 0), (1, 1), (hi, hi), (hi, 1), (1, hi), (hi >> 1, hi),
        ]

    vecs = []
    for a, b in corners:
        y = a * b
        vecs.append((
            f"corner_{a}_{b}",
            {"a": _bits(a, N_BITS), "b": _bits(b, N_BITS)},
            {"y": _bits(y, 2 * N_BITS)},
        ))

    state = 0xBEEF1
    for i in range(N_VECS - len(corners)):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        raw_a = state & ((1 << N_BITS) - 1)
        raw_b = (state >> N_BITS) & ((1 << N_BITS) - 1)
        if signed:
            sign = 1 << (N_BITS - 1)
            a = raw_a - (1 << N_BITS) if raw_a & sign else raw_a
            b = raw_b - (1 << N_BITS) if raw_b & sign else raw_b
        else:
            a, b = raw_a, raw_b
        y = a * b
        vecs.append((
            f"rand_{i}",
            {"a": _bits(a, N_BITS), "b": _bits(b, N_BITS)},
            {"y": _bits(y, 2 * N_BITS)},
        ))
    return vecs


# ---------------------------------------------------------------------------
# Parametrised matrix: {adder, mul} × {operator, structural} × {unsigned, signed}
# Each test runs both on the SpireHDL Simulator AND on the AIG (post
# ``refactor_module_to_aig``). Before this test, only the SpireHDL path was
# covered — which is why the signed AIG zero-extension bug survived.
# ---------------------------------------------------------------------------

ENCODINGS = [Encoding.unsigned, Encoding.twos_complement]
USE_OPERATOR = [True, False]


def _adder_cfg(use_operator: bool, enc: Encoding) -> AdderConfig:
    if use_operator:
        return AdderConfig(use_operator=True, encoding=enc, optim_type="area")
    return AdderConfig(
        use_operator=False,
        encoding=enc,
        optim_type="area",
        fsa_opt=FSAOption.PREFIX_BRENT_KUNG,
        full_output_bit=True,
    )


def _mult_cfg(use_operator: bool, enc: Encoding) -> MultiplierConfig:
    encs = TwoInputAritEncodings.with_enc(enc)
    if use_operator:
        return MultiplierConfig(use_operator=True, encodings=encs, optim_type="area")
    return MultiplierConfig(
        use_operator=False,
        multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
        encodings=encs,
        ppg_opt=PPGOption.BAUGH_WOOLEY if enc == Encoding.twos_complement else PPGOption.AND,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.PREFIX_BRENT_KUNG,
        optim_type="area",
    )


@pytest.mark.parametrize("use_operator", USE_OPERATOR, ids=lambda v: "operator" if v else "structural")
@pytest.mark.parametrize("enc", ENCODINGS, ids=lambda e: e.name)
def test_build_adder_spire_and_aig(use_operator: bool, enc: Encoding):
    """``build_adder`` must produce correct bits in both SpireHDL and AIG sims."""
    reset_shared_cache()
    signed = enc == Encoding.twos_complement
    cfg = _adder_cfg(use_operator, enc)
    comp = _make_adder_component(cfg, signed)
    module = comp.to_module(
        f"Add_{'op' if use_operator else 'struct'}_{enc.name}",
        with_clock=True, with_reset=True,
    )
    vecs = _adder_vectors(signed)

    # SpireHDL-level simulation (already well-covered by the existing suite,
    # but we keep it here so a failure tells you *which* path broke).
    run_vectors(module, vecs, use_signed=False)

    # AIG round-trip — the path that flows into sim_and_switch_count.
    m_aig = refactor_module_to_aig(module)
    run_vectors(m_aig, vecs, use_signed=False)


@pytest.mark.parametrize("use_operator", USE_OPERATOR, ids=lambda v: "operator" if v else "structural")
@pytest.mark.parametrize("enc", ENCODINGS, ids=lambda e: e.name)
def test_build_multiplier_spire_and_aig(use_operator: bool, enc: Encoding):
    """``build_multiplier`` must produce correct bits in both sim paths."""
    reset_shared_cache()
    signed = enc == Encoding.twos_complement
    cfg = _mult_cfg(use_operator, enc)
    comp = _make_multiplier_component(cfg, signed)
    module = comp.to_module(
        f"Mul_{'op' if use_operator else 'struct'}_{enc.name}",
        with_clock=True, with_reset=True,
    )
    vecs = _mul_vectors(signed)

    run_vectors(module, vecs, use_signed=False)
    m_aig = refactor_module_to_aig(module)
    run_vectors(m_aig, vecs, use_signed=False)


# ---------------------------------------------------------------------------
# Focused regression: the exact failure mode from the MMAC baseline sweep
# ---------------------------------------------------------------------------

def test_signed_add_sext_in_aig():
    """Regression for the ``bv_add`` zero-extension bug.

    Minimal reproducer: a signed 8-bit add whose declared result width is
    9 bits. The operand list in ``bv_add`` is length 8; when the inner
    loop runs to bit 8 it used to pad with ``lit_const0()``, giving the
    wrong upper bit for negative sums. The concrete witness ``-1 + 1``
    exposes this — the correct 9-bit result is ``0x000`` but the buggy
    path returns ``0x100`` (carry out of the unsigned add instead of the
    sign-extended zero).
    """
    reset_shared_cache()
    cfg = AdderConfig(use_operator=True, encoding=Encoding.twos_complement, optim_type="area")
    comp = _make_adder_component(cfg, signed=True)
    module = comp.to_module("SignedAddSext", with_clock=True, with_reset=True)

    witnesses = [
        (-1, 1, 0),       # the smoking gun: bit 8 must be 0, not 1
        (-1, -1, -2),     # both sign bits set
        (-128, -128, -256),  # most-neg + most-neg
        (127, 127, 254),     # max-pos + max-pos
        (-128, 127, -1),     # asymmetric
    ]
    vecs = [
        (f"{a}_{b}", {"a": _bits(a, N_BITS), "b": _bits(b, N_BITS)}, {"y": _bits(y, N_BITS + 1)})
        for a, b, y in witnesses
    ]

    # AIG path is where the bug lived — this is the must-pass assertion.
    m_aig = refactor_module_to_aig(module)
    run_vectors(m_aig, vecs, use_signed=False)


def test_signed_sub_sext_in_aig():
    """Same regression pattern for ``-`` (the ``bv_sub`` path routes through
    ``bv_add`` internally, so the same sign-extension gap applied)."""
    reset_shared_cache()

    class SubWrap(Component):
        def __init__(self):
            self.io = StageBasedMultiplierIO(
                a=Signal(typ=SInt(N_BITS), kind="input"),
                b=Signal(typ=SInt(N_BITS), kind="input"),
                y=Signal(typ=UInt(N_BITS + 1), kind="output"),
            )
            self.io.y <<= self.io.a - self.io.b

        def elaborate(self):  # logic built in __init__; no-op to satisfy the abstract method
            pass

    comp = SubWrap()
    module = comp.to_module("SignedSubSext", with_clock=True, with_reset=True)

    witnesses = [
        (1, -1, 2),        # 1 - (-1) = 2, but both sign bits need extension
        (-1, 1, -2),
        (-128, 1, -129),   # underflow past min
        (127, -1, 128),    # overflow past max
        (-128, -128, 0),
    ]
    vecs = [
        (f"{a}_{b}", {"a": _bits(a, N_BITS), "b": _bits(b, N_BITS)}, {"y": _bits(y, N_BITS + 1)})
        for a, b, y in witnesses
    ]

    m_aig = refactor_module_to_aig(module)
    run_vectors(m_aig, vecs, use_signed=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
