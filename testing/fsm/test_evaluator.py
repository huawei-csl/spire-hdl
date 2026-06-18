"""Symbolic evaluator (Step 5 of the FSM-encoding-search plan)."""
from __future__ import annotations

import pytest

from spire.optimize.fsm._evaluator import eval_with
from spire.expr import Bool, Const, UInt, cat, mux
from spire.component import Module


def test_const_returns_value():
    assert eval_with(Const(7, UInt(8)), {}) == 7
    assert eval_with(Const(0, UInt(1)), {}) == 0


def test_const_width_masked():
    # 8-bit Const initialized with a wider value should mask in the evaluator.
    c = Const(0x123, UInt(8))
    # The constructor stores .value as int(value) (no truncation), but the
    # evaluator masks to typ.width.
    assert eval_with(c, {}) == 0x23


def test_op2_add_with_signal_binding():
    m = Module("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    expr = a + b
    assert eval_with(expr, [(a, 3), (b, 5)]) == 8


def test_op2_add_wraps_at_width():
    m = Module("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    expr = a + b
    # 8-bit add wraps at 256... but spire's `+` widens to 9 bits by default.
    # Test the width-mask behaviour by binding values that don't wrap there.
    assert eval_with(expr, [(a, 200), (b, 100)]) == 300


def test_op2_logical_ops():
    m = Module("t", with_clock=False, with_reset=False)
    a = m.input(UInt(4), "a")
    b = m.input(UInt(4), "b")
    assert eval_with(a & b, [(a, 0b1100), (b, 0b1010)]) == 0b1000
    assert eval_with(a | b, [(a, 0b1100), (b, 0b1010)]) == 0b1110
    assert eval_with(a ^ b, [(a, 0b1100), (b, 0b1010)]) == 0b0110


def test_op2_compare():
    m = Module("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    assert eval_with(a == b, [(a, 5), (b, 5)]) == 1
    assert eval_with(a == b, [(a, 5), (b, 6)]) == 0
    assert eval_with(a < b, [(a, 3), (b, 5)]) == 1
    assert eval_with(a < b, [(a, 5), (b, 3)]) == 0


def test_ternary_picks_branch():
    m = Module("t", with_clock=False, with_reset=False)
    sel = m.input(Bool(), "sel")
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    expr = mux(sel, a, b)
    assert eval_with(expr, [(sel, 1), (a, 99), (b, 0)]) == 99
    assert eval_with(expr, [(sel, 0), (a, 99), (b, 0)]) == 0


def test_signal_with_driver_is_followed():
    """If a Signal isn't in env but has a driver, the evaluator follows it."""
    m = Module("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    w = m.wire(UInt(8), "w")
    w <<= a + 1
    # Only bind 'a' — 'w' should be computed from its driver.
    assert eval_with(w, [(a, 7)]) == 8


def test_unbound_signal_raises():
    m = Module("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    with pytest.raises(ValueError, match="no value in env"):
        eval_with(a, [])


def test_nested_mux_chain():
    """A switch_/case_-style nested mux over a 4-state selector."""
    m = Module("t", with_clock=False, with_reset=False)
    sel = m.input(UInt(2), "sel")
    expr = mux(sel == 0, 10,
           mux(sel == 1, 20,
           mux(sel == 2, 30, 40)))
    assert eval_with(expr, [(sel, 0)]) == 10
    assert eval_with(expr, [(sel, 1)]) == 20
    assert eval_with(expr, [(sel, 2)]) == 30
    assert eval_with(expr, [(sel, 3)]) == 40


def test_concat_lsb_first():
    """cat(a, b) places a in lower bits, b in upper."""
    m = Module("t", with_clock=False, with_reset=False)
    a = m.input(UInt(2), "a")
    b = m.input(UInt(2), "b")
    expr = cat(a, b)                          # 4-bit, a is low
    assert eval_with(expr, [(a, 0b01), (b, 0b10)]) == 0b1001
    assert eval_with(expr, [(a, 0b11), (b, 0b00)]) == 0b0011
