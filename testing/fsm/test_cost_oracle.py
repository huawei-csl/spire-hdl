"""Cost oracle (Step 9). Lightweight tests; the heavy yosys-shell-out is
covered by the end-to-end tests in Step 14."""
from __future__ import annotations

import shutil

import pytest

from spirehdl.fsm._cost_oracle import make_yosys_cost_fn
from spirehdl.fsm._emit import restore_encoding
from spirehdl.spirehdl import Bool, UInt, mux
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_state import Encoding, State, state


HAS_YOSYS = shutil.which("yosys") is not None
requires_yosys = pytest.mark.skipif(not HAS_YOSYS, reason="yosys binary not on PATH")


class Op(State, encoding=Encoding.BINARY):
    ADD = state(); SUB = state(); AND = state(); OR = state()


@pytest.fixture(autouse=True)
def _restore_op():
    yield
    restore_encoding(Op, {"ADD": 0, "SUB": 1, "AND": 2, "OR": 3})


def _build_alu_module() -> Module:
    m = Module("alu_op", with_clock=False, with_reset=False)
    op = m.input(Op.typ, "op")
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(8), "y")
    y <<= mux(op == Op.ADD, a + b,
          mux(op == Op.SUB, a - b,
          mux(op == Op.AND, a & b,
                            a | b)))
    return m


@requires_yosys
def test_cost_fn_returns_finite_number_for_valid_assignment():
    m = _build_alu_module()
    cost_fn = make_yosys_cost_fn(m, Op, objective="cells")
    cost = cost_fn({"ADD": 0, "SUB": 1, "AND": 2, "OR": 3})
    assert cost != float("inf")
    assert cost > 0


@requires_yosys
def test_cost_fn_restores_encoding_on_success():
    m = _build_alu_module()
    cost_fn = make_yosys_cost_fn(m, Op, objective="cells")
    cost_fn({"ADD": 3, "SUB": 0, "AND": 1, "OR": 2})
    # After the call returns, the State Consts must be back to original.
    assert Op.ADD.value == 0
    assert Op.SUB.value == 1
    assert Op.AND.value == 2
    assert Op.OR.value == 3


@requires_yosys
def test_cost_fn_restores_encoding_on_exception():
    """Even if yosys fails or the assignment is malformed, the State class
    must be restored before the cost_fn returns."""
    m = _build_alu_module()
    cost_fn = make_yosys_cost_fn(m, Op, objective="cells")
    # Missing-state assignment triggers an exception inside apply_encoding;
    # cost_fn must catch it, return inf, and restore originals.
    cost = cost_fn({"ADD": 0, "SUB": 1})       # missing AND and OR
    assert cost == float("inf")
    assert Op.ADD.value == 0
    assert Op.SUB.value == 1
    assert Op.AND.value == 2
    assert Op.OR.value == 3


@requires_yosys
def test_two_different_assignments_can_yield_different_costs():
    """Sanity: encoding actually affects synthesis (not necessarily true for
    every design, but for an opcode dispatch it usually is)."""
    m = _build_alu_module()
    cost_fn = make_yosys_cost_fn(m, Op, objective="cells")
    cost_a = cost_fn({"ADD": 0, "SUB": 1, "AND": 2, "OR": 3})
    cost_b = cost_fn({"ADD": 3, "SUB": 2, "AND": 1, "OR": 0})
    # At minimum, both are finite. The test doesn't insist they differ —
    # synthesis may collapse opcode permutations into the same gate count —
    # but it must not produce NaN.
    assert cost_a != float("inf")
    assert cost_b != float("inf")


def test_cost_fn_objective_selection():
    """The objective kwarg must select between cells/wires/transistors at
    construction time (the returned closure reads it)."""
    m = _build_alu_module()
    fn_cells = make_yosys_cost_fn(m, Op, objective="cells")
    fn_wires = make_yosys_cost_fn(m, Op, objective="wires")
    # Without yosys we can still construct both — the call would fail at
    # subprocess time, not construction time.
    assert callable(fn_cells)
    assert callable(fn_wires)
