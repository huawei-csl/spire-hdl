"""FixedPoint arithmetic: exhaustive differential against an exact-math golden model, plus emission conformance.

Add/sub promote to a signed full type when the operand signs differ or when subtracting unsigned operands
(RealARITH semantics), so every add/sub/mul is lossless at full precision; quantization is floor (Trc) or
add-half-then-floor (Rnd), wrapped into the output width (Wrp).
"""
import math
from fractions import Fraction

import pytest

from differential.harness import diff_sim_vs_verilog, exhaustive
from spire.composite.fixed_point import ARITHQuant, FixedPoint, FixedPointType
from spire.expr import SInt, UInt, reinterpret, reset_shared_cache
from spire.ir import Netlist
from spire.simulator import Simulator


def _decode_int(pattern: int, width: int, signed: bool) -> int:
    return pattern - (1 << width) if signed and (pattern >> (width - 1)) & 1 else pattern


def _golden(pa: int, pb: int, ta: FixedPointType, tb: FixedPointType, op: str,
            out: FixedPointType, q: ARITHQuant) -> int:
    ia = Fraction(_decode_int(pa, ta.width_total, ta.signed), 1 << ta.width_frac)
    ib = Fraction(_decode_int(pb, tb.width_total, tb.signed), 1 << tb.width_frac)
    exact = {"add": ia + ib, "sub": ia - ib, "mul": ia * ib}[op]
    scaled = exact * (1 << out.width_frac)
    if q == ARITHQuant.WrpRnd and scaled.denominator > 1:
        scaled += Fraction(1, 2)
    return math.floor(scaled) % (1 << out.width_total)


def _promoted(op: str, sa: bool, sb: bool) -> bool:
    return sa or sb or (op == "sub")


def _sig(m, name, ftype):
    raw = m.input(UInt(ftype.width_total), name)
    return reinterpret(raw, SInt(ftype.width_total)) if ftype.signed else raw


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
@pytest.mark.parametrize("q", [ARITHQuant.WrpTrc, ARITHQuant.WrpRnd])
def test_ops_match_exact_math(op, q):
    sign_pairs = [(False, False), (True, True)] + ([(False, True), (True, False)] if op != "mul" else [])
    for sa, sb in sign_pairs:
        for wt, wf in ((4, 2), (5, 3), (4, 4)):
            ta = FixedPointType(width_total=wt, width_frac=wf, signed=sa)
            tb = FixedPointType(width_total=wt, width_frac=wf, signed=sb)
            out_signed = _promoted(op, sa, sb)
            outs = [
                FixedPointType(width_total=4, width_frac=1, signed=out_signed),
                FixedPointType(width_total=6, width_frac=4, signed=out_signed),
                FixedPointType(width_total=wt, width_frac=0, signed=out_signed),
            ]
            for out in outs:
                reset_shared_cache()
                m = Netlist(f"fx_{op}_{q.name}", with_clock=False, with_reset=False)
                fa = FixedPoint(ta, bits=_sig(m, "a", ta))
                fb = FixedPoint(tb, bits=_sig(m, "b", tb))
                y = m.output(UInt(out.width_total), "y")
                y <<= getattr(fa, op)(fb, out_type=out, q=q).bits

                bad, text = diff_sim_vs_verilog(m, ["a", "b"], exhaustive(wt, wt))
                assert not bad, f"{op}/{q.name} {ta}/{tb}→{out}: sim/Verilog diverge ({len(bad)})\n{text}"

                sim = Simulator(m)
                for pa, pb in exhaustive(wt, wt):
                    sim.set("a", pa)
                    sim.set("b", pb)
                    sim.eval()
                    got = sim.get("y") % (1 << out.width_total)
                    want = _golden(pa, pb, ta, tb, op, out, q)
                    assert got == want, f"{op}/{q.name} {ta}/{tb}→{out}: a={pa} b={pb} got={got} want={want}"


def test_wrong_out_signedness_rejected():
    ta = FixedPointType(width_total=4, width_frac=2, signed=False)
    reset_shared_cache()
    m = Netlist("fx_sign_err", with_clock=False, with_reset=False)
    fa = FixedPoint(ta, bits=_sig(m, "a", ta))
    fb = FixedPoint(ta, bits=_sig(m, "b", ta))
    with pytest.raises(ValueError, match="promote to signed"):
        fa.sub(fb, out_type=FixedPointType(width_total=4, width_frac=0, signed=False))


def test_mixed_sign_mul_rejected():
    ta = FixedPointType(width_total=4, width_frac=2, signed=False)
    tb = FixedPointType(width_total=4, width_frac=2, signed=True)
    reset_shared_cache()
    m = Netlist("fx_mul_err", with_clock=False, with_reset=False)
    fa = FixedPoint(ta, bits=_sig(m, "a", ta))
    fb = FixedPoint(tb, bits=_sig(m, "b", tb))
    with pytest.raises(ValueError, match="sign mismatch"):
        fa.mul(fb)


@pytest.mark.parametrize("q", [ARITHQuant.WrpTrc, ARITHQuant.WrpRnd])
def test_neg_matches_exact_math(q):
    for signed in (False, True):
        for wt, wf in ((4, 2), (5, 3)):
            ta = FixedPointType(width_total=wt, width_frac=wf, signed=signed)
            outs = [FixedPointType(width_total=4, width_frac=1, signed=True),
                    FixedPointType(width_total=6, width_frac=4, signed=True)]
            for out in outs:
                reset_shared_cache()
                m = Netlist(f"fx_neg_{q.name}", with_clock=False, with_reset=False)
                fa = FixedPoint(ta, bits=_sig(m, "a", ta))
                y = m.output(UInt(out.width_total), "y")
                y <<= fa.neg(out_type=out, q=q).bits

                bad, text = diff_sim_vs_verilog(m, ["a"], exhaustive(wt))
                assert not bad, f"neg/{q.name} {ta}→{out}: sim/Verilog diverge\n{text}"

                sim = Simulator(m)
                for pa in range(1 << wt):
                    sim.set("a", pa)
                    sim.eval()
                    va = Fraction(_decode_int(pa, wt, signed), 1 << wf)
                    scaled = -va * (1 << out.width_frac)
                    if q == ARITHQuant.WrpRnd and scaled.denominator > 1:
                        scaled += Fraction(1, 2)
                    want = math.floor(scaled) % (1 << out.width_total)
                    got = sim.get("y") % (1 << out.width_total)
                    assert got == want, f"neg {ta}→{out}: a={pa} got={got} want={want}"


def test_dunder_neg_full_precision():
    ta = FixedPointType(width_total=4, width_frac=2, signed=True)
    reset_shared_cache()
    m = Netlist("fx_negfp", with_clock=False, with_reset=False)
    fa = FixedPoint(ta, bits=_sig(m, "a", ta))
    r = -fa
    assert r.signed and r.ftype.width_total == 5
    y = m.output(UInt(5), "y")
    y <<= r.bits
    sim = Simulator(m)
    sim.set("a", 0b1000)  # -8/4 = -2.0 → +2.0 = 8/4, needs the extra MSB
    sim.eval()
    assert sim.get("y") % 32 == 8
