"""Tests for ``MemoryPrimitive`` — the Component-based memory built via ``custom_verilog``.

Mirrors ``testing/test_memory.py`` (built-in ``Memory``) so the two can be compared
side-by-side, plus an aggregate-element-type test that the built-in ``Memory``
cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from spirehdl.spirehdl import (
    Bool,
    Const,
    Signal,
    UInt,
    Wire,
    reset_shared_cache,
)
from spirehdl.spirehdl_module import Component, Module
from spirehdl.spirehdl_simulator import Simulator
from spirehdl.primitives import MemoryPrimitive
from spirehdl.aggregate.aggregate_record import AggregateRecord


# ---------------------------------------------------------------------------
# Helpers: parent Components that expose a MemoryPrimitive's ports
# ---------------------------------------------------------------------------

def _build_fifo16(*, registered_read: bool = False, with_reset_arm: bool = True, init=None):
    """Reset-armed, async-read RAM, depth=16, width=9.

    Returns the constructed ``Module`` and the inner ``MemoryPrimitive`` instance
    name (always ``"fifo"``).
    """

    class TopRam(Component):
        def __init__(self):
            @dataclass
            class IO:
                we: Signal
                clr: Signal
                addr_w: Signal
                addr_r: Signal
                din: Signal
                dout: Signal
            self.io = IO(
                we     = Signal("we",     Bool(),   "input"),
                clr    = Signal("clr",    Bool(),   "input"),
                addr_w = Signal("addr_w", UInt(4), "input"),
                addr_r = Signal("addr_r", UInt(4), "input"),
                din    = Signal("din",    UInt(9), "input"),
                dout   = Signal("dout",   UInt(9), "output"),
            )
            self.elaborate()

        def elaborate(self):
            mem = MemoryPrimitive(
                UInt(9), depth=16,
                registered_read=registered_read,
                with_reset_arm=with_reset_arm,
                init=init,
                name="fifo",
            ).make_internal()
            mem.io.write_addr   <<= self.io.addr_w
            mem.io.write_data   <<= self.io.din
            mem.io.write_enable <<= self.io.we
            if with_reset_arm:
                mem.io.reset_enable <<= self.io.clr
            mem.io.read_addr <<= self.io.addr_r
            self.io.dout     <<= mem.io.read_data

    reset_shared_cache()
    return TopRam().to_module(name="fifo16", with_clock=True, with_reset=True)


def _build_rom8(init):
    """Registered-read ROM, depth=8, width=8. init is a list of 8 ints."""

    class TopRom(Component):
        def __init__(self):
            @dataclass
            class IO:
                addr: Signal
                re: Signal
                dout: Signal
            self.io = IO(
                addr = Signal("addr", UInt(3), "input"),
                re   = Signal("re",   Bool(),  "input"),
                dout = Signal("dout", UInt(8), "output"),
            )
            self.elaborate()

        def elaborate(self):
            rom = MemoryPrimitive(
                UInt(8), depth=8,
                registered_read=True,
                init=init,
                name="rom",
            ).make_internal()
            rom.io.read_addr   <<= self.io.addr
            rom.io.read_enable <<= self.io.re
            # write port is required at the boundary but tied off here (ROM behavior).
            rom.io.write_addr   <<= Const(0, UInt(3))
            rom.io.write_data   <<= Const(0, UInt(8))
            rom.io.write_enable <<= Const(0, Bool())
            self.io.dout <<= rom.io.read_data

    reset_shared_cache()
    return TopRom().to_module(name="rom8", with_clock=True, with_reset=True)


# ---------------------------------------------------------------------------
# Verilog emission
# ---------------------------------------------------------------------------

def test_emits_memory_array_declaration():
    m = _build_fifo16()
    v = m.to_verilog()
    # The whole point of a memory primitive: storage emits as a verilog array.
    assert "reg [8:0] fifo[0:15];" in v


def test_emits_reset_then_write_else_block():
    m = _build_fifo16()
    v = m.to_verilog()
    # Reset arm clears all entries; write arm gated by write_enable.
    assert "if (reset_enable) begin" in v
    assert "fifo[0] <= 9'd0;" in v
    assert "fifo[15] <= 9'd0;" in v
    assert "end else begin" in v
    assert "if (write_enable) fifo[write_addr] <= write_data;" in v


def test_combinational_read_emits_array_index():
    m = _build_fifo16()
    v = m.to_verilog()
    # Async read: a single `assign read_data = fifo[read_addr];` line.
    assert "assign read_data = fifo[read_addr];" in v


def test_registered_read_emits_in_own_always_block():
    init = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]
    m = _build_rom8(init)
    v = m.to_verilog()
    # Storage + internal rdata register declared inside the custom block.
    assert "reg [7:0] rom[0:7];" in v
    assert "reg [7:0] rom__rd0;" in v
    # Init block populated from the list.
    assert "initial begin" in v
    assert "rom[0] = 8'd16;" in v
    assert "rom[7] = 8'd128;" in v
    # Clock-only sensitivity (yosys-recognised memory idiom).
    assert "always @(posedge clk) begin" in v
    assert "if (read_enable) rom__rd0 <= rom[read_addr];" in v
    # No async-rst sensitivity on the memory always block.
    assert "posedge clk or posedge rst" not in v.split("// --- MemoryPrimitive")[1]


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_init_length_mismatch_raises():
    with pytest.raises(ValueError, match="init must have length"):
        MemoryPrimitive(UInt(8), depth=4, init=[1, 2, 3], name="bad")


def test_depth_zero_raises():
    with pytest.raises(ValueError, match="depth must be > 0"):
        MemoryPrimitive(UInt(8), depth=0, name="bad")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def test_sim_write_then_read():
    m = _build_fifo16()
    sim = Simulator(m)
    sim.deassert_reset()
    sim.set("we", 1).set("addr_w", 3).set("din", 0xAB).step()
    sim.set("addr_w", 5).set("din", 0xCD).step()
    sim.set("we", 0)
    sim.set("addr_r", 3).eval()
    assert sim.get("dout") == 0xAB
    sim.set("addr_r", 5).eval()
    assert sim.get("dout") == 0xCD
    # Unwritten address reads 0.
    sim.set("addr_r", 7).eval()
    assert sim.get("dout") == 0


def test_sim_reset_clears_all_entries():
    m = _build_fifo16()
    sim = Simulator(m)
    sim.deassert_reset()
    sim.set("we", 1).set("addr_w", 3).set("din", 0xAB).step()
    sim.set("we", 0)
    sim.set("clr", 1).step()
    sim.set("clr", 0)
    sim.set("addr_r", 3).eval()
    assert sim.get("dout") == 0


def test_sim_rom_init_and_registered_read():
    init = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]
    m = _build_rom8(init)
    sim = Simulator(m)
    sim.deassert_reset()
    sim.set("re", 1).set("addr", 0).step()
    assert sim.get("dout") == 0x10
    sim.set("addr", 5).step()
    assert sim.get("dout") == 0x60
    # With read-enable low the rdata register holds its last value.
    sim.set("re", 0).set("addr", 2).step()
    assert sim.get("dout") == 0x60


# ---------------------------------------------------------------------------
# Aggregate element type — the capability the built-in Memory lacks.
# ---------------------------------------------------------------------------

class _Bus(AggregateRecord):
    data  = Wire(UInt(8))
    valid = Wire(UInt(1))


def test_sim_aggregate_elem_type():
    """Element type is an HDLAggregate. User packs / unpacks at the port boundary."""

    bus_width = _Bus().width  # 9 bits

    class TopAgg(Component):
        def __init__(self):
            @dataclass
            class IO:
                we: Signal
                addr_w: Signal
                addr_r: Signal
                din_data: Signal
                din_valid: Signal
                dout_data: Signal
                dout_valid: Signal
            self.io = IO(
                we         = Signal("we",         Bool(),   "input"),
                addr_w     = Signal("addr_w",     UInt(2), "input"),
                addr_r     = Signal("addr_r",     UInt(2), "input"),
                din_data   = Signal("din_data",   UInt(8), "input"),
                din_valid  = Signal("din_valid",  Bool(),  "input"),
                dout_data  = Signal("dout_data",  UInt(8), "output"),
                dout_valid = Signal("dout_valid", Bool(),  "output"),
            )
            self.elaborate()

        def elaborate(self):
            mem = MemoryPrimitive(_Bus, depth=4, name="busmem").make_internal()
            # Pack input bus into UInt(9) for the write port.
            bus_in = _Bus()
            bus_in.data  <<= self.io.din_data
            bus_in.valid <<= self.io.din_valid
            mem.io.write_data   <<= bus_in.to_bits()
            mem.io.write_addr   <<= self.io.addr_w
            mem.io.write_enable <<= self.io.we
            mem.io.read_addr    <<= self.io.addr_r
            # Unpack read port into a structured bus view.
            out_bus = _Bus()
            out_bus <<= mem.io.read_data
            self.io.dout_data  <<= out_bus.data
            self.io.dout_valid <<= out_bus.valid

    reset_shared_cache()
    m = TopAgg().to_module(name="aggmem", with_clock=True, with_reset=True)
    # Sanity: width is what we expect.
    assert bus_width == 9
    # Storage array width matches the packed aggregate.
    v = m.to_verilog()
    assert "reg [8:0] busmem[0:3];" in v

    sim = Simulator(m)
    sim.deassert_reset()

    sim.set("we", 1).set("addr_w", 0).set("din_data", 0x42).set("din_valid", 1).step()
    sim.set("addr_w", 2).set("din_data", 0x7E).set("din_valid", 0).step()
    sim.set("we", 0)

    sim.set("addr_r", 0).eval()
    assert sim.get("dout_data") == 0x42
    assert sim.get("dout_valid") == 1

    sim.set("addr_r", 2).eval()
    assert sim.get("dout_data") == 0x7E
    assert sim.get("dout_valid") == 0
