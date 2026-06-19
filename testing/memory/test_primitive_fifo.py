"""Tests for ``FIFOPrimitive`` — sync FIFO with one-cycle read latency.

Convention:
  * After ``deassert_reset()``, ``empty=1`` and ``count=0``.
  * A push captured at clock edge T puts the value into ``mem[wr_ptr]`` and advances
    ``wr_ptr`` / increments ``count``.
  * A pop captured at clock edge T captures ``mem[rd_ptr]`` into ``dout`` (visible
    on cycle T+1), advances ``rd_ptr``, decrements ``count``.
  * Simultaneous push + pop on a non-empty/non-full FIFO leaves ``count`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from spire.expr import (
    Bool,
    Signal,
    UInt,
    Wire,
    reset_shared_cache,
)
from spire.component import Component
from spire.simulator import Simulator
from spire.primitives import FIFOPrimitive
from spire.composite.record import CompositeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_fifo(*, depth: int = 4, elem_w: int = 8, name: str = "myfifo"):
    """Wraps a FIFOPrimitive in a parent Component so it sits as the top of a Netlist."""
    count_w = (depth - 1).bit_length() + 1

    class TopFifo(Component):
        def __init__(self):
            @dataclass
            class IO:
                push:  Signal
                pop:   Signal
                din:   Signal
                dout:  Signal
                full:  Signal
                empty: Signal
                count: Signal
            self.io = IO(
                push  = Signal(typ=Bool(), kind="input"),
                pop   = Signal(typ=Bool(), kind="input"),
                din   = Signal(typ=UInt(elem_w), kind="input"),
                dout  = Signal(typ=UInt(elem_w), kind="output"),
                full  = Signal(typ=Bool(), kind="output"),
                empty = Signal(typ=Bool(), kind="output"),
                count = Signal(typ=UInt(count_w), kind="output"),
            )
            self.elaborate()

        def elaborate(self):
            fifo = FIFOPrimitive(UInt(elem_w), depth=depth, name=name).make_internal()
            fifo.io.push <<= self.io.push
            fifo.io.pop  <<= self.io.pop
            fifo.io.din  <<= self.io.din
            self.io.dout  <<= fifo.io.dout
            self.io.full  <<= fifo.io.full
            self.io.empty <<= fifo.io.empty
            self.io.count <<= fifo.io.count

    reset_shared_cache()
    return TopFifo().to_module(name="fifotop", with_clock=True, with_reset=True)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_depth_too_small_raises():
    with pytest.raises(ValueError, match="depth must be >= 2"):
        FIFOPrimitive(UInt(8), depth=1)


def test_depth_not_power_of_two_raises():
    with pytest.raises(ValueError, match="power of two"):
        FIFOPrimitive(UInt(8), depth=6)


# ---------------------------------------------------------------------------
# Verilog emission
# ---------------------------------------------------------------------------

def test_emits_fifo_block():
    m = _build_fifo(depth=4, elem_w=8, name="myfifo")
    v = m.to_verilog()
    # Storage + pointer + count regs all appear inside the custom block.
    assert "reg [7:0] myfifo__mem[0:3];" in v
    assert "reg [1:0] myfifo__wr;" in v
    assert "reg [1:0] myfifo__rd;" in v
    assert "reg [2:0] myfifo__cnt;" in v
    assert "reg [7:0] myfifo__dout_r;" in v
    # Push / pop gating wires.
    assert "myfifo__empty_w = (myfifo__cnt == 0);" in v
    assert "myfifo__full_w = (myfifo__cnt == 4);" in v
    # Reset arm and write/read inside the always block.
    assert "myfifo__mem[myfifo__wr] <= " in v
    assert "myfifo__dout_r <= myfifo__mem[myfifo__rd];" in v


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def test_sim_push_pop_basic():
    """Push N items, pop them, verify FIFO order."""
    m = _build_fifo(depth=4, elem_w=8)
    sim = Simulator(m)
    sim.deassert_reset()
    assert sim.get("empty") == 1
    assert sim.get("count") == 0

    # Push 1, 2, 3, 4
    for v in (1, 2, 3, 4):
        sim.set("push", 1).set("din", v).step()
    sim.set("push", 0).eval()
    assert sim.get("count") == 4
    assert sim.get("full") == 1
    assert sim.get("empty") == 0

    # Pop them; dout is registered (one-cycle latency).
    expected = [1, 2, 3, 4]
    for v in expected:
        sim.set("pop", 1).step()
        assert sim.get("dout") == v
    sim.set("pop", 0).eval()
    assert sim.get("count") == 0
    assert sim.get("empty") == 1


def test_sim_full_and_empty_flags():
    m = _build_fifo(depth=4, elem_w=8)
    sim = Simulator(m)
    sim.deassert_reset()

    # Push 4 → full.
    for v in (10, 20, 30, 40):
        sim.set("push", 1).set("din", v).step()
    sim.set("push", 0).eval()
    assert sim.get("full") == 1
    assert sim.get("empty") == 0
    assert sim.get("count") == 4

    # Pop until empty.
    for _ in range(4):
        sim.set("pop", 1).step()
    sim.set("pop", 0).eval()
    assert sim.get("empty") == 1
    assert sim.get("full") == 0
    assert sim.get("count") == 0


def test_sim_simultaneous_push_pop():
    """Push and pop on the same cycle while non-empty/non-full: count unchanged,
    new data flows through."""
    m = _build_fifo(depth=4, elem_w=8)
    sim = Simulator(m)
    sim.deassert_reset()

    # Prime with two values.
    sim.set("push", 1).set("din", 0x11).step()
    sim.set("din", 0x22).step()
    sim.set("push", 0).eval()
    assert sim.get("count") == 2

    # Simultaneous push + pop: count stays at 2, head moves to 0x22.
    sim.set("push", 1).set("pop", 1).set("din", 0x33).step()
    assert sim.get("dout") == 0x11   # popped value
    assert sim.get("count") == 2

    # Pop again; head is the second pre-loaded value.
    sim.set("push", 0).set("pop", 1).step()
    assert sim.get("dout") == 0x22
    assert sim.get("count") == 1

    # Pop the new push.
    sim.step()
    assert sim.get("dout") == 0x33
    assert sim.get("count") == 0


def test_sim_underflow_and_overflow_safety():
    """Pop while empty does nothing; push while full does nothing."""
    m = _build_fifo(depth=2, elem_w=8)
    sim = Simulator(m)
    sim.deassert_reset()

    # Underflow: pop on empty.
    sim.set("pop", 1).step()
    assert sim.get("count") == 0
    assert sim.get("empty") == 1
    sim.set("pop", 0).eval()

    # Fill it.
    sim.set("push", 1).set("din", 0xAA).step()
    sim.set("din", 0xBB).step()
    sim.set("push", 0).eval()
    assert sim.get("full") == 1
    assert sim.get("count") == 2

    # Overflow: push on full.
    sim.set("push", 1).set("din", 0xFF).step()
    assert sim.get("count") == 2
    assert sim.get("full") == 1
    sim.set("push", 0).eval()

    # Pop; should get the first two pushed values, *not* the overflow value.
    sim.set("pop", 1).step()
    assert sim.get("dout") == 0xAA
    sim.step()
    assert sim.get("dout") == 0xBB
    sim.set("pop", 0).eval()
    assert sim.get("empty") == 1


# ---------------------------------------------------------------------------
# Composite element type
# ---------------------------------------------------------------------------

class _Bus(CompositeRecord):
    data  = Wire(UInt(8))
    valid = Wire(UInt(1))


def test_sim_composite_elem_type_fifo():
    """Element type is an HDLComposite. User packs / unpacks at the port boundary."""

    class TopAggFifo(Component):
        def __init__(self):
            @dataclass
            class IO:
                push:       Signal
                pop:        Signal
                din_data:   Signal
                din_valid:  Signal
                dout_data:  Signal
                dout_valid: Signal
            self.io = IO(
                push       = Signal(typ=Bool(), kind="input"),
                pop        = Signal(typ=Bool(), kind="input"),
                din_data   = Signal(typ=UInt(8), kind="input"),
                din_valid  = Signal(typ=Bool(), kind="input"),
                dout_data  = Signal(typ=UInt(8), kind="output"),
                dout_valid = Signal(typ=Bool(), kind="output"),
            )
            self.elaborate()

        def elaborate(self):
            fifo = FIFOPrimitive(_Bus, depth=4, name="bfifo").make_internal()
            bus_in = _Bus()
            bus_in.data  <<= self.io.din_data
            bus_in.valid <<= self.io.din_valid
            fifo.io.push <<= self.io.push
            fifo.io.pop  <<= self.io.pop
            fifo.io.din  <<= bus_in.to_bits()
            out_bus = _Bus()
            out_bus <<= fifo.io.dout
            self.io.dout_data  <<= out_bus.data
            self.io.dout_valid <<= out_bus.valid

    reset_shared_cache()
    m = TopAggFifo().to_module(name="aggfifo", with_clock=True, with_reset=True)
    # Sanity: storage width matches the packed composite.
    v = m.to_verilog()
    assert "reg [8:0] bfifo__mem[0:3];" in v

    sim = Simulator(m)
    sim.deassert_reset()

    # Push (0x42, valid=1) then (0x7E, valid=0).
    sim.set("push", 1).set("din_data", 0x42).set("din_valid", 1).step()
    sim.set("din_data", 0x7E).set("din_valid", 0).step()
    sim.set("push", 0).eval()

    # Pop and observe.
    sim.set("pop", 1).step()
    assert sim.get("dout_data") == 0x42
    assert sim.get("dout_valid") == 1

    sim.step()
    assert sim.get("dout_data") == 0x7E
    assert sim.get("dout_valid") == 0
