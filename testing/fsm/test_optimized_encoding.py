"""End-to-end test of ``optimized_encoding``.

We use a custom synthetic cost_fn (no yosys dependency) to verify the
wrapper integrates the snapshot/walker/search/apply pipeline end-to-end
deterministically. The yosys-based path is exercised by
``test_nested_wrappers.py`` (which skips when yosys is unavailable).
"""
from __future__ import annotations

import pytest

from spirehdl.fsm._emit import restore_encoding
from spirehdl.spirehdl import Bool, UInt, mux
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_state import (
    Encoding, State, optimized_encoding, state,
)


class Op(State, encoding=Encoding.BINARY):
    ADD = state(); SUB = state(); AND = state(); OR = state()


@pytest.fixture(autouse=True)
def _restore_op():
    yield
    restore_encoding(Op, {"ADD": 0, "SUB": 1, "AND": 2, "OR": 3})


def test_optimized_encoding_picks_minimum_under_synthetic_cost():
    """Custom cost_fn returns 0 for the target assignment, 1 otherwise.
    With strategy=exhaustive, the wrapper must find and apply that target.
    """
    m = Module("alu_op", with_clock=False, with_reset=False)
    op = m.input(Op.typ, "op")
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(8), "y")

    target = {"ADD": 3, "SUB": 0, "AND": 1, "OR": 2}

    def cost_fn(assignment):
        return 0.0 if assignment == target else 1.0

    with optimized_encoding(Op, module=m, search="exhaustive", cost_fn=cost_fn):
        # Body that *references* Op Consts so the walker confirms work needs doing.
        y <<= mux(op == Op.ADD, a + b,
              mux(op == Op.SUB, a - b,
              mux(op == Op.AND, a & b,
                                a | b)))

    assert Op._values == target


def test_no_state_const_use_is_noop():
    """If the user's with-block doesn't reference the State class, the wrapper
    should detect that and skip the search entirely (leaving the encoding alone)."""
    m = Module("noop", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(8), "y")

    # A cost_fn that would fire if called.
    called = {"n": 0}
    def cost_fn(assignment):
        called["n"] += 1
        return 0.0

    before = dict(Op._values)
    with optimized_encoding(Op, module=m, search="exhaustive", cost_fn=cost_fn):
        y <<= a + b                 # no reference to Op
    # No search performed; Op untouched.
    assert called["n"] == 0
    assert Op._values == before


def test_predefined_strategy():
    """With strategy=predefined, only BINARY and GRAY are tried."""
    m = Module("t", with_clock=False, with_reset=False)
    op = m.input(Op.typ, "op")
    a = m.input(UInt(8), "a")
    y = m.output(UInt(8), "y")

    # GRAY for 4 states = [0, 1, 3, 2]. Make GRAY uniquely optimal.
    gray = {"ADD": 0, "SUB": 1, "AND": 3, "OR": 2}
    def cost_fn(assignment):
        return 0.0 if assignment == gray else 5.0

    with optimized_encoding(Op, module=m, search="predefined", cost_fn=cost_fn):
        y <<= mux(op == Op.ADD, a,
              mux(op == Op.SUB, a,
              mux(op == Op.AND, a, a)))
    assert Op._values == gray
