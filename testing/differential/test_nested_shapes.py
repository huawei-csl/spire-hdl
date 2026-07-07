"""Nested-shape conformance battery: compound operands in every context the §2 table can put them in.

Each shape drives a compound subexpression into a position where IEEE-1364 context rules could re-size it (operand
of a wider op, comparison island, ternary branch, extension concat, shift, flat emission). The simulator defines the
expected value (docs/README_semantics.md §2); the assertion is zero divergence against the IEEE evaluation of the
emitted text. Known-divergent shapes at the current baseline carry strict xfail marks naming the owning issue.
"""
from __future__ import annotations

import pytest

from spire.expr import Bool, SInt, UInt, cat, flat_emit, mux

from .harness import diff_sim_vs_verilog, format_mismatches, fresh_netlist, sweep

XF = pytest.mark.xfail  # applied per-shape below with strict=True


def shape_sub_plus(m):  # (a - b) + c, unsigned: inner sub must wrap at 5 bits
    a, b, c = m.input(UInt(4), "a"), m.input(UInt(4), "b"), m.input(UInt(8), "c")
    y = m.output(UInt(9), "y"); y <<= (a - b) + c
    return ["a", "b", "c"], sweep(4, 4, 8)


def shape_sub_plus_right(m):  # control: c + (a - b) — the sub is the right operand (shared today)
    a, b, c = m.input(UInt(4), "a"), m.input(UInt(4), "b"), m.input(UInt(8), "c")
    y = m.output(UInt(9), "y"); y <<= c + (a - b)
    return ["a", "b", "c"], sweep(4, 4, 8)


def shape_not_plus_one(m):  # ~a + 1: the two's-complement idiom
    a = m.input(UInt(4), "a")
    y = m.output(UInt(5), "y"); y <<= ~a + 1
    return ["a"], sweep(4)


def shape_mixed_inner(m):  # (s4 + u4) + s8: mixed-sign inner op nested in a signed context
    a, b, c = m.input(SInt(4), "a"), m.input(UInt(4), "b"), m.input(SInt(8), "c")
    y = m.output(UInt(9), "y"); y <<= (a + b) + c
    return ["a", "b", "c"], sweep(4, 4, 8)


def shape_signed_shr(m):  # (s4 >> 1) + s8: shift-then-extend vs extend-then-shift
    a, c = m.input(SInt(4), "a"), m.input(SInt(8), "c")
    y = m.output(UInt(9), "y"); y <<= (a >> 1) + c
    return ["a", "c"], sweep(4, 8)


def shape_var_shl(m):  # (a << k) + c: variable shift must wrap at a's width
    a, k, c = m.input(UInt(4), "a"), m.input(UInt(2), "k"), m.input(UInt(8), "c")
    y = m.output(UInt(9), "y"); y <<= (a << k) + c
    return ["a", "k", "c"], sweep(4, 2, 8)


def shape_cmp_compound_signed(m):  # (s4 - s4) < s5: all operands width-5 → nothing is materialized
    # Conformant at the baseline BY ACCIDENT: the raw emission `((a - b) < c)` with all-signed declarations forms a
    # signed IEEE compare island whose context evaluation equals spire's. Wrapping the compound in `$signed(...)`
    # (the old branch's 0.1 fix) self-determines it and BREAKS this shape (ISSUES2 §1.3) — any compare re-fix must
    # keep this test green, which is why it carries no xfail.
    a, b, c = m.input(SInt(4), "a"), m.input(SInt(4), "b"), m.input(SInt(5), "c")
    y = m.output(Bool(), "y"); y <<= (a - b) < c
    return ["a", "b", "c"], sweep(4, 4, 5)


def shape_cmp_compound_signed_narrow(m):  # (s4 - s4) < s4: c is extended → unsigned concat poisons the island
    a, b, c = m.input(SInt(4), "a"), m.input(SInt(4), "b"), m.input(SInt(4), "c")
    y = m.output(Bool(), "y"); y <<= (a - b) < c
    return ["a", "b", "c"], sweep(4, 4, 4)


def shape_cmp_compound_unsigned(m):  # control: (u4 - u4) < u5
    a, b, c = m.input(UInt(4), "a"), m.input(UInt(4), "b"), m.input(UInt(5), "c")
    y = m.output(Bool(), "y"); y <<= (a - b) < c
    return ["a", "b", "c"], sweep(4, 4, 5)


def shape_flat_eq(m):  # (a + b) == (c + d) under flat_emit: carries dropped on both sides
    a, b = m.input(UInt(4), "a"), m.input(UInt(4), "b")
    c, d = m.input(UInt(4), "c"), m.input(UInt(4), "d")
    y = m.output(Bool(), "y"); y <<= (a + b) == (c + d)
    return ["a", "b", "c", "d"], sweep(4, 4, 4, 4)


def shape_neg_plus(m):  # (-s4) + s8: unary minus is a mixed op feeding a signed context
    a, c = m.input(SInt(4), "a"), m.input(SInt(8), "c")
    y = m.output(UInt(9), "y"); y <<= (-a) + c
    return ["a", "c"], sweep(4, 8)


def shape_mux_operand(m):  # mux(c, s4, s5) + s8: ternary as an arithmetic operand
    c, a, b, d = m.input(Bool(), "c"), m.input(SInt(4), "a"), m.input(SInt(5), "b"), m.input(SInt(8), "d")
    y = m.output(UInt(9), "y"); y <<= mux(c, a, b) + d
    return ["c", "a", "b", "d"], sweep(1, 4, 5, 8)


def shape_sub_times(m):  # (a - b) * c, unsigned: wrapped sub feeding a multiply
    a, b, c = m.input(UInt(4), "a"), m.input(UInt(4), "b"), m.input(UInt(4), "c")
    y = m.output(UInt(9), "y"); y <<= (a - b) * c
    return ["a", "b", "c"], sweep(4, 4, 4)


def shape_controls(m):  # lossless/boundary controls: + nesting, slice of compound, concat of compound
    a, b, c = m.input(UInt(4), "a"), m.input(UInt(4), "b"), m.input(UInt(8), "c")
    y = m.output(UInt(9), "y"); y <<= (a + b) + c
    y2 = m.output(UInt(4), "y2"); y2 <<= (a - b)[0:4]
    y3 = m.output(UInt(9), "y3"); y3 <<= cat(a - b, a[0:4])
    return ["a", "b", "c"], sweep(4, 4, 8)


def shape_masked_sub(m):  # ((a - b) & 7) + c: compound inside a bitwise op
    a, b, c = m.input(UInt(4), "a"), m.input(UInt(4), "b"), m.input(UInt(8), "c")
    y = m.output(UInt(9), "y"); y <<= ((a - b) & 0x7) + c
    return ["a", "b", "c"], sweep(4, 4, 8)


def shape_cmp_in_arith(m):  # (a < b) + c: comparison feeding arithmetic
    a, b, c = m.input(UInt(4), "a"), m.input(UInt(4), "b"), m.input(UInt(4), "c")
    y = m.output(UInt(5), "y"); y <<= (a < b) + c
    return ["a", "b", "c"], sweep(4, 4, 4)


def shape_not_signed_plus(m):  # (~s8) + s8: unsigned ~ result mixed into signed arithmetic
    a, b = m.input(SInt(8), "a"), m.input(SInt(8), "b")
    y = m.output(UInt(9), "y"); y <<= (~a) + b
    return ["a", "b"], sweep(8, 8)


SHAPES = [
    # (builder, flat_emit?, strict-xfail reason or None). Shapes without a reason are conformance guarantees;
    # the §1.1/§1.5/§1.6-family entries flipped green when emission-time width isolation landed.
    (shape_sub_plus, False, None),
    (shape_sub_plus_right, False, None),
    (shape_not_plus_one, False, None),
    (shape_mixed_inner, False, "ISSUES 0.2: mixed-sign op emits an unsigned expression"),
    (shape_signed_shr, False, None),
    (shape_var_shl, False, None),
    (shape_cmp_compound_signed, False, None),  # all-signed island; also guards the ISSUES2 §1.3 $signed trap
    (shape_cmp_compound_signed_narrow, False, None),  # isolation keeps the alignment wire signed
    (shape_cmp_compound_unsigned, False, None),
    (shape_flat_eq, True, None),
    (shape_neg_plus, False, "ISSUES 0.2: unary minus (`0 - a`) is a mixed-sign op"),
    (shape_mux_operand, False, None),
    (shape_sub_times, False, None),
    (shape_controls, False, None),
    (shape_masked_sub, False, None),
    (shape_cmp_in_arith, False, None),
    (shape_not_signed_plus, False, "ISSUES 0.2: unsigned ~ result mixed into signed arithmetic"),
]


def _params():
    for build, flat, reason in SHAPES:
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        yield pytest.param(build, flat, id=build.__name__.removeprefix("shape_"), marks=marks)


@pytest.mark.parametrize("build,flat", list(_params()))
def test_nested_shape(build, flat):
    if flat:
        with flat_emit():
            m = fresh_netlist(build.__name__)
            inputs, vectors = build(m)
            bad, text = diff_sim_vs_verilog(m, inputs, vectors)
    else:
        m = fresh_netlist(build.__name__)
        inputs, vectors = build(m)
        bad, text = diff_sim_vs_verilog(m, inputs, vectors)
    assert not bad, f"{build.__name__}\n{format_mismatches(bad, inputs)}\n{text}"
