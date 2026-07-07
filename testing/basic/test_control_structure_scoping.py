"""Scoping of if_/elif_/else_ chains and condition state across branches, switches, and components."""
import pytest

from spire import Component, IORecord, Input, Output
from spire.expr import UInt, Wire, reset_shared_cache
from spire.control_structures import case_, else_, if_, switch_
from spire.ir import Netlist
from spire.simulator import Simulator


class _PassThrough(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(4)), z=Output(UInt(4)))
        self.elaborate()

    def elaborate(self):
        self.io.z <<= self.io.a


class _TrailingIf(Component):
    """Leaves a pending if_ chain behind inside elaborate()."""

    def __init__(self):
        self.io = IORecord(a=Input(UInt(4)), z=Output(UInt(4)))
        self.elaborate()

    def elaborate(self):
        self.io.z <<= 0
        with if_(self.io.a == 1):
            self.io.z <<= self.io.a


def _sim(m):
    return Simulator(m)


def test_else_cannot_cross_component_boundary():
    reset_shared_cache()
    _TrailingIf()  # its trailing chain must not escape
    with pytest.raises(RuntimeError, match="must follow an if_"):
        with else_():
            pass


def test_component_between_if_and_else_works():
    # A component constructed between if_ and else_ neither swallows nor corrupts the pending chain.
    reset_shared_cache()
    m = Netlist("scope1", with_clock=False, with_reset=False)
    c = m.input(UInt(1), "c")
    x = m.input(UInt(4), "x")
    y = m.output(UInt(4), "y")
    y <<= 0
    with if_(c == 1):
        y <<= x
    sub = _PassThrough()
    sub.io.a <<= x
    with else_():
        y <<= 15

    sim = _sim(m)
    for cv, xv, expect in [(1, 5, 5), (0, 5, 15)]:
        sim.set("c", cv)
        sim.set("x", xv)
        sim.eval()
        assert sim.get("y") == expect


def test_component_under_if_elaborates_unconditionally():
    # Elaboration is structural: the sub-component's internals are not gated by the enclosing condition,
    # and the caller's own conditional assignment still works after construction.
    reset_shared_cache()
    m = Netlist("scope2", with_clock=False, with_reset=False)
    c = m.input(UInt(1), "c")
    x = m.input(UInt(4), "x")
    y = m.output(UInt(4), "y")
    z = m.output(UInt(4), "z")
    y <<= 0
    with if_(c == 1):
        sub = _PassThrough()
        y <<= sub.io.z
    sub.io.a <<= x  # inputs are wired unconditionally (a gated input drive would need a fallback)
    z <<= sub.io.z

    sim = _sim(m)
    for cv, xv, y_expect in [(1, 9, 9), (0, 9, 0)]:
        sim.set("c", cv)
        sim.set("x", xv)
        sim.eval()
        assert sim.get("y") == y_expect
        assert sim.get("z") == xv  # never gated


def test_else_cannot_cross_case_scopes():
    reset_shared_cache()
    m = Netlist("scope3", with_clock=False, with_reset=False)
    sel = m.input(UInt(2), "sel")
    c = m.input(UInt(1), "c")
    y = m.output(UInt(4), "y")
    y <<= 0
    with switch_(sel):
        with case_(0):
            with if_(c == 1):
                y <<= 1
        with case_(1):
            with pytest.raises(RuntimeError, match="different branch or switch case"):
                with else_():
                    y <<= 2


def test_helper_chain_closed_by_caller_still_works():
    reset_shared_cache()
    m = Netlist("scope4", with_clock=False, with_reset=False)
    c = m.input(UInt(1), "c")
    y = m.output(UInt(4), "y")
    y <<= 0

    def helper(sig):
        with if_(c == 1):
            sig <<= 3

    helper(y)
    with else_():
        y <<= 4

    sim = _sim(m)
    for cv, expect in [(1, 3), (0, 4)]:
        sim.set("c", cv)
        sim.eval()
        assert sim.get("y") == expect


def test_composite_rhs_under_if():
    reset_shared_cache()
    m = Netlist("scope5", with_clock=False, with_reset=False)
    c = m.input(UInt(1), "c")
    a = m.input(UInt(4), "a")
    b = m.input(UInt(4), "b")
    y = m.output(UInt(8), "y")
    rec = IORecord(lo=Wire(UInt(4)), hi=Wire(UInt(4)))
    rec.lo <<= a
    rec.hi <<= b
    y <<= 0
    with if_(c == 1):
        y <<= rec  # composite RHS packs to bits under a condition

    sim = _sim(m)
    sim.set("a", 5)
    sim.set("b", 2)
    for cv, expect in [(1, 5 | (2 << 4)), (0, 0)]:
        sim.set("c", cv)
        sim.eval()
        assert sim.get("y") == expect


def test_direct_assign_respects_conditions():
    reset_shared_cache()
    m = Netlist("scope6", with_clock=False, with_reset=False)
    c = m.input(UInt(1), "c")
    x = m.input(UInt(4), "x")
    y = m.output(UInt(4), "y")
    y <<= 0
    with if_(c == 1):
        y.assign(x)  # the documented primitive must be gated like <<=

    sim = _sim(m)
    sim.set("x", 9)
    for cv, expect in [(1, 9), (0, 0)]:
        sim.set("c", cv)
        sim.eval()
        assert sim.get("y") == expect


def test_case_value_out_of_selector_range_rejected():
    reset_shared_cache()
    m = Netlist("scope7", with_clock=False, with_reset=False)
    sel = m.input(UInt(2), "sel")
    y = m.output(UInt(4), "y")
    y <<= 0
    with switch_(sel):
        with pytest.raises(ValueError, match="can never match"):
            with case_(7):
                y <<= 1


def test_unbalanced_exit_detected():
    reset_shared_cache()
    m = Netlist("scope8", with_clock=False, with_reset=False)
    c = m.input(UInt(1), "c")
    ctx = if_(c == 1)
    with ctx:
        pass
    with pytest.raises(RuntimeError, match="not active"):
        ctx.__exit__(None, None, None)
