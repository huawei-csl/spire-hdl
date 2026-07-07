"""Sharing machinery: visitor-cache pinning, CSE emit-once, and force-share reliability."""
import gc
import weakref

from spire.expr import Op2, Signal, UInt, flat_emit, reset_shared_cache
from spire.ir import Netlist
from spire.simulator import Simulator
from spire.visitor import ExprVisitor


class _KeyVisitor(ExprVisitor):
    """Minimal structural-key visitor with no state beyond the base cache."""

    def visit_signal(self, e):
        return ("sig", id(e))

    def visit_op2(self, e):
        return ("op2", e.op, self.visit(e.a), self.visit(e.b))


def test_visitor_cache_pins_nodes():
    # The base-class cache must pin visited nodes: a freed node's id could otherwise be recycled by a new
    # node, which would inherit the stale cached key and cause false merges downstream.
    reset_shared_cache()
    m = Netlist("pin", with_clock=False, with_reset=False)
    a = m.input(UInt(4), "a")
    b = m.input(UInt(4), "b")
    e = a & b
    visitor = _KeyVisitor()
    visitor.visit(e)
    ref = weakref.ref(e)
    del e
    gc.collect()
    assert ref() is not None, "visited node was garbage-collected; its id is free to recycle"


def test_structural_duplicates_emit_once():
    # Two structurally identical trees built as distinct instances must emit the shared logic exactly once.
    reset_shared_cache()
    m = Netlist("cse_once", with_clock=False, with_reset=False)
    a = m.input(UInt(4), "a")
    b = m.input(UInt(4), "b")
    c = m.input(UInt(4), "c")
    y0 = m.output(UInt(4), "y0")
    y1 = m.output(UInt(4), "y1")
    with flat_emit():  # keep construction-time sharing out of the way; CSE must do the collapsing
        y0 <<= (a & b) | c
        y1 <<= (a & b) | c

    text = m.to_verilog()
    # The shared tree must emit exactly once (as one wire's driver), not inline at the first use too.
    assert text.count("| c)") == 1, text

    sim = Simulator(m)
    sim.set("a", 12)
    sim.set("b", 10)
    sim.set("c", 1)
    sim.eval()
    assert sim.get("y0") == sim.get("y1") == (12 & 10) | 1


def test_force_share_after_prior_references():
    # A bit-select on an already-referenced compound must still get its named base wire.
    reset_shared_cache()
    m = Netlist("fs", with_clock=False, with_reset=False)
    a = m.input(UInt(4), "a")
    b = m.input(UInt(4), "b")
    y0 = m.output(UInt(4), "y0")
    y1 = m.output(UInt(1), "y1")
    with flat_emit():
        e = a & b
        y0 <<= e        # first reference; no wire under flat build
        sel = e[0]      # __getitem__ force-shares the base — must succeed despite the prior reference
    assert isinstance(sel.a, Signal), "force_share failed: slice base is an inline compound"
    y1 <<= sel

    sim = Simulator(m)
    sim.set("a", 3)
    sim.set("b", 1)
    sim.eval()
    assert sim.get("y0") == 1
    assert sim.get("y1") == 1
