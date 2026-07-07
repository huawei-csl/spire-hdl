"""Leaf-operand conformance battery: simulator vs IEEE evaluation of the emitted Verilog.

Every single-operator shape with *leaf* operands (inputs only) and an output typed exactly like the operator's
result, evaluated exhaustively over small widths. This doubles as the calibration battery required by
docs/README_semantics.md §5: on conformant shapes the reference evaluator must agree with the simulator on every
vector, which pins the evaluator itself before its verdicts on composite shapes (test_nested_shapes.py) count.

Shapes that are known-divergent at the current baseline carry strict xfail marks naming the issue; fixing the
emitter flips them and pytest will fail on any unexpected pass, so the marks must be removed in the same change.
"""
from __future__ import annotations

import pytest

from spire.expr import SInt, UInt, cat, mux

from .harness import diff_sim_vs_verilog, exhaustive, format_mismatches, fresh_netlist, output_like

WIDTH_PAIRS = ((1, 1), (2, 4), (4, 2), (4, 4))

BINARY_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "shl_var": lambda a, b: a << b,
    "shr_var": lambda a, b: a >> b,
    "mux": lambda a, b: mux(a[0:1] == 1, a, b),
    "cat": lambda a, b: cat(a, b),
}

UNARY_OPS = {
    "not": lambda a: ~a,
    "neg": lambda a: -a,
    "shl_const": lambda a: a << 3,
    "shr_const": lambda a: a >> 2,
    "slice": lambda a: a[0:1] if a.typ.width == 1 else a[1 : a.typ.width],
}

ORDERED_CMP = {"lt", "le", "gt", "ge"}
ARITH = {"add", "sub", "mul"}


def _baseline_xfail(op: str, sa: bool, sb: bool):
    """Expected-divergent leaf shapes at the current baseline, with the issue that owns them."""
    if op in ARITH and sa != sb:
        return "ISSUES 0.2: mixed signed/unsigned +/-/* emits unsigned Verilog"
    if op in ORDERED_CMP and (sa or sb):
        return "ISSUES 0.1: signed comparisons emit unsigned Verilog"
    return None


def _binary_params():
    for op in BINARY_OPS:
        for sa in (False, True):
            for sb in (False, True):
                reason = _baseline_xfail(op, sa, sb)
                marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
                yield pytest.param(op, sa, sb, id=f"{op}-{'s' if sa else 'u'}{'s' if sb else 'u'}", marks=marks)


@pytest.mark.parametrize("op,sa,sb", list(_binary_params()))
def test_leaf_binary(op, sa, sb):
    build = BINARY_OPS[op]
    for wa, wb in WIDTH_PAIRS:
        m = fresh_netlist(f"leaf_{op}_{int(sa)}{int(sb)}_{wa}{wb}")
        a = m.input(SInt(wa) if sa else UInt(wa), "a")
        b = m.input(SInt(wb) if sb else UInt(wb), "b")
        output_like(m, build(a, b), "y")
        bad, text = diff_sim_vs_verilog(m, ["a", "b"], exhaustive(wa, wb))
        assert not bad, f"{op} w=({wa},{wb}) s=({sa},{sb})\n{format_mismatches(bad, ['a', 'b'])}\n{text}"


def _unary_params():
    for op in UNARY_OPS:
        for sa in (False, True):
            reason = None
            if op == "neg" and sa:
                reason = "ISSUES 0.2: unary minus builds `0 - a` (a mixed-sign op) — unsigned Verilog"
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            yield pytest.param(op, sa, id=f"{op}-{'s' if sa else 'u'}", marks=marks)


@pytest.mark.parametrize("op,sa", list(_unary_params()))
def test_leaf_unary(op, sa):
    build = UNARY_OPS[op]
    for wa in (1, 2, 4):
        if op == "shr_const" and wa < 3:
            continue
        m = fresh_netlist(f"leafu_{op}_{int(sa)}_{wa}")
        a = m.input(SInt(wa) if sa else UInt(wa), "a")
        output_like(m, build(a), "y")
        bad, text = diff_sim_vs_verilog(m, ["a"], exhaustive(wa))
        assert not bad, f"{op} w={wa} s={sa}\n{format_mismatches(bad, ['a'])}\n{text}"
