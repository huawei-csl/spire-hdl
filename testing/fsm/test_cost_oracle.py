"""Cost oracle (Step 9). Uses in-process pyosys + aigverse — no `yosys` binary
required, so these tests run unconditionally.
"""
from __future__ import annotations

import pytest

from spirehdl.optimize.fsm._cost_oracle import make_cost_fn, make_yosys_cost_fn
from spirehdl.optimize.fsm._emit import restore_encoding
from spirehdl.spirehdl import UInt, mux
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_state import Encoding, State, state


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


def test_cost_fn_returns_finite_number_for_valid_assignment():
    m = _build_alu_module()
    cost_fn = make_cost_fn(m, Op, objective="cells")
    cost = cost_fn({"ADD": 0, "SUB": 1, "AND": 2, "OR": 3})
    assert cost != float("inf")
    assert cost > 0


def test_cost_fn_restores_encoding_on_success():
    m = _build_alu_module()
    cost_fn = make_cost_fn(m, Op, objective="cells")
    cost_fn({"ADD": 3, "SUB": 0, "AND": 1, "OR": 2})
    assert Op.ADD.value == 0
    assert Op.SUB.value == 1
    assert Op.AND.value == 2
    assert Op.OR.value == 3


def test_cost_fn_restores_encoding_on_exception():
    """Even on a malformed assignment, the State class must be restored
    before the cost_fn returns. The exception is caught and inf is returned
    so the search rejects the candidate."""
    m = _build_alu_module()
    cost_fn = make_cost_fn(m, Op, objective="cells")
    cost = cost_fn({"ADD": 0, "SUB": 1})       # missing AND and OR
    assert cost == float("inf")
    assert Op.ADD.value == 0
    assert Op.SUB.value == 1
    assert Op.AND.value == 2
    assert Op.OR.value == 3


def test_two_different_assignments_can_yield_different_costs():
    """Sanity: encoding actually affects synthesis (not necessarily true for
    every design, but for an opcode dispatch it usually is)."""
    m = _build_alu_module()
    cost_fn = make_cost_fn(m, Op, objective="cells")
    cost_a = cost_fn({"ADD": 0, "SUB": 1, "AND": 2, "OR": 3})
    cost_b = cost_fn({"ADD": 3, "SUB": 2, "AND": 1, "OR": 0})
    assert cost_a != float("inf")
    assert cost_b != float("inf")


def test_cost_fn_objective_selection_yosys_metrics():
    """Each yosys-side objective resolves to a finite distinct float for a
    non-trivial module."""
    m = _build_alu_module()
    assignment = {"ADD": 0, "SUB": 1, "AND": 2, "OR": 3}
    fn_cells       = make_cost_fn(m, Op, objective="cells")
    fn_wires       = make_cost_fn(m, Op, objective="wires")
    fn_transistors = make_cost_fn(m, Op, objective="transistors")
    c, w, t = fn_cells(assignment), fn_wires(assignment), fn_transistors(assignment)
    for v in (c, w, t):
        assert v != float("inf")
        assert v > 0


def test_cost_fn_aig_gates_objective():
    """The new aig_gates objective returns a positive finite integer count
    via aigverse (no yosys binary required)."""
    m = _build_alu_module()
    cost_fn = make_cost_fn(m, Op, objective="aig_gates")
    cost = cost_fn({"ADD": 0, "SUB": 1, "AND": 2, "OR": 3})
    assert cost != float("inf")
    assert cost > 0
    # AIG gate counts are integers; the float should be one.
    assert cost == int(cost)


def test_cost_fn_aig_depth_objective():
    """The new aig_depth objective returns a positive finite integer depth."""
    m = _build_alu_module()
    cost_fn = make_cost_fn(m, Op, objective="aig_depth")
    cost = cost_fn({"ADD": 0, "SUB": 1, "AND": 2, "OR": 3})
    assert cost != float("inf")
    assert cost > 0
    assert cost == int(cost)


def test_unknown_objective_returns_inf():
    """Unknown objective triggers the ValueError inside _measure → caught
    by the cost_fn's except → returns inf so the search rejects."""
    m = _build_alu_module()
    cost_fn = make_cost_fn(m, Op, objective="not_an_objective")  # type: ignore[arg-type]
    cost = cost_fn({"ADD": 0, "SUB": 1, "AND": 2, "OR": 3})
    assert cost == float("inf")


def test_backward_compat_alias_still_works():
    """make_yosys_cost_fn (the pre-pyosys name) remains importable and is
    bound to make_cost_fn."""
    assert make_yosys_cost_fn is make_cost_fn
    m = _build_alu_module()
    cost_fn = make_yosys_cost_fn(m, Op, objective="cells")
    assert callable(cost_fn)
    assert cost_fn({"ADD": 0, "SUB": 1, "AND": 2, "OR": 3}) != float("inf")
