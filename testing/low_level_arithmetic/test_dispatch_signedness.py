"""Dispatch-layer signedness contracts.

The auto path never coerces mixed-sign operands into one encoding (subtractor included);
use_operator returns the same type as the structural path; fused dot chains carry the
encoding's signedness so signed refits sign-extend.
"""
import itertools

import pytest

from spire.arithmetic.int_arithmetic_config import (
    ArithmeticAutoConfig, MultiplierConfig, build_multiplier, build_subtractor,
)
from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption, PPAOption, PPGOption,
)
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding
from spire.cores.matmul_accumulate.matmul_accumulate_core_fused import (
    MultiplierConfig as FusedCfg, fused_inner_product,
)
from spire.expr import Const, SInt, UInt, fit_type, reinterpret, reset_shared_cache
from spire.ir import Netlist
from spire.simulator import Simulator


def _decode(p, w, signed):
    return p - (1 << w) if signed and (p >> (w - 1)) & 1 else p


def test_mixed_sign_subtractor_auto_falls_back_exhaustively():
    reset_shared_cache()
    m = Netlist("subq", with_clock=False, with_reset=False)
    a = m.input(UInt(4), "a")
    b = m.input(SInt(4), "b")
    y = m.output(UInt(5), "y")
    y <<= build_subtractor(a, reinterpret(b, SInt(4)), ArithmeticAutoConfig())
    sim = Simulator(m)
    for pa, pb in itertools.product(range(16), range(16)):
        sim.set("a", pa).set("b", pb)
        sim.eval()
        want = (_decode(pa, 4, False) - _decode(pb, 4, True)) & 0x1F
        assert sim.get("y") & 0x1F == want, f"{pa}-{pb}: {sim.get('y') & 0x1F} != {want}"


def test_use_operator_matches_structural_result_type():
    reset_shared_cache()
    m = Netlist("mulq", with_clock=False, with_reset=False)
    a = m.input(SInt(4), "a")
    b = m.input(SInt(4), "b")
    cfg_kw = dict(ppg_opt=PPGOption.BOOTH_UNOPTIMISED, ppa_opt=PPAOption.WALLACE_TREE,
                  fsa_opt=FSAOption.RIPPLE_CARRY)
    from spire.arithmetic.int_arithmetic_config import TwoInputAritEncodings
    from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import MultiplierOption
    enc = TwoInputAritEncodings.with_enc(Encoding.twos_complement)
    structural = build_multiplier(a, b, MultiplierConfig(
        multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER, encodings=enc, **cfg_kw))
    operator = build_multiplier(a, b, MultiplierConfig(encodings=enc, use_operator=True, **cfg_kw))
    assert operator.typ.signed == structural.typ.signed, \
        f"operator path {operator.typ} vs structural {structural.typ}"
    # widened through SInt(12): both paths must sign-extend identically
    ys = m.output(UInt(12), "ys")
    yo = m.output(UInt(12), "yo")
    ys <<= fit_type(structural, SInt(12))
    yo <<= fit_type(operator, SInt(12))
    sim = Simulator(m)
    for pa, pb in itertools.product(range(16), range(16)):
        sim.set("a", pa).set("b", pb)
        sim.eval()
        assert sim.get("ys") == sim.get("yo"), f"a={pa} b={pb}"


def test_fused_inner_product_signed_refit_is_correct():
    reset_shared_cache()
    m = Netlist("fusedq", with_clock=False, with_reset=False)
    ins = [m.input(SInt(3), n) for n in ("a0", "b0", "a1", "b1")]
    a0, b0, a1, b1 = ins
    y = m.output(UInt(12), "y")
    cfg = FusedCfg(ppg_opt=PPGOption.BOOTH_UNOPTIMISED, ppa_opt=PPAOption.WALLACE_TREE,
                   fsa_opt=FSAOption.RIPPLE_CARRY)
    dot = fused_inner_product([a0, a1], [b0, b1], Const(0, UInt(1)), cfg, Encoding.twos_complement)
    assert dot.typ.signed, "signed-encoding fused result must be SInt (refits sign-extend)"
    y <<= fit_type(dot, SInt(12))
    sim = Simulator(m)
    for pa0, pb0, pa1, pb1 in itertools.product(range(8), repeat=4):
        for n, v in zip(("a0", "b0", "a1", "b1"), (pa0, pb0, pa1, pb1)):
            sim.set(n, v)
        sim.eval()
        want = (_decode(pa0, 3, True) * _decode(pb0, 3, True)
                + _decode(pa1, 3, True) * _decode(pb1, 3, True)) & 0xFFF
        assert sim.get("y") == want, f"{pa0},{pb0},{pa1},{pb1}: {sim.get('y')} != {want}"
