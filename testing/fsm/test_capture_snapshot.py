"""SharedCacheSnapshot (Step 3 of the FSM-encoding-search plan)."""
from __future__ import annotations

from spire.optimize.fsm._capture import SharedCacheSnapshot
from spire.expr import UInt
from spire.component import Module


def test_snapshot_records_no_wires_for_empty_block():
    m = Module("t", with_clock=False, with_reset=False)
    snap = SharedCacheSnapshot()
    with snap:
        pass
    assert snap.new_wires == []


def test_snapshot_captures_wires_added_inside_with():
    m = Module("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    snap = SharedCacheSnapshot()
    with snap:
        # Each non-trivial Expr is auto-shared via _maybe_share, appending to
        # _SharedCache.wires. Bind to an output so the framework computes a
        # full Expr tree.
        y = m.output(UInt(8), "y")
        y <<= (a & b) | (a + 1)
    # At least the AND and OR wires above should appear.
    assert len(snap.new_wires) >= 2


def test_snapshot_excludes_wires_added_before_with():
    m = Module("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")

    # Build an expression BEFORE the snapshot.
    pre_out = m.output(UInt(8), "pre_out")
    pre_out <<= a + b

    snap = SharedCacheSnapshot()
    with snap:
        post_out = m.output(UInt(8), "post_out")
        post_out <<= a & b
    # Only wires created inside the with-block should be in new_wires.
    # Check by object identity (Signal.__eq__ builds an Expr, so the usual
    # `in` / `.index()` operations don't work).
    from spire.expr import _SharedCache
    new_ids = {id(w) for w in snap.new_wires}
    pre_ids = {id(w) for w in _SharedCache.wires[:snap._start_idx]}
    assert new_ids.isdisjoint(pre_ids)


def test_two_consecutive_snapshots_dont_overlap():
    m = Module("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")

    snap1 = SharedCacheSnapshot()
    with snap1:
        out1 = m.output(UInt(8), "y1"); out1 <<= a + 1
    n1 = len(snap1.new_wires)

    snap2 = SharedCacheSnapshot()
    with snap2:
        out2 = m.output(UInt(8), "y2"); out2 <<= a + 2
    n2 = len(snap2.new_wires)

    # No wire should appear in both diffs (disjoint id sets).
    s1_ids = {id(w) for w in snap1.new_wires}
    s2_ids = {id(w) for w in snap2.new_wires}
    assert s1_ids.isdisjoint(s2_ids)
    # Each block independently added something.
    assert n1 >= 1
    assert n2 >= 1


def test_nested_snapshots():
    m = Module("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    outer = SharedCacheSnapshot()
    inner = SharedCacheSnapshot()
    with outer:
        out0 = m.output(UInt(8), "y0"); out0 <<= a + 1
        with inner:
            outx = m.output(UInt(8), "y1"); outx <<= a + 2
        # Outer's new_wires is queried after the inner finishes.
    # Outer sees a superset; inner sees only its own.
    inner_ids = {id(w) for w in inner.new_wires}
    outer_ids = {id(w) for w in outer.new_wires}
    assert inner_ids.issubset(outer_ids)
    assert len(outer_ids) > len(inner_ids)


def test_new_wires_live_query_during_block():
    m = Module("t", with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    snap = SharedCacheSnapshot()
    with snap:
        out0 = m.output(UInt(8), "y0"); out0 <<= a + 1
        midpoint = len(snap.new_wires)
        outx = m.output(UInt(8), "y1"); outx <<= a + 2
    assert len(snap.new_wires) >= midpoint
