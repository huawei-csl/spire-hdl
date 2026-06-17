"""Tests for ``RomPrimitive`` — read-only memory (init-backed), single read port.

Covers async and registered reads (behaviour via simulation) plus emission: the array +
``initial`` block, the async ``assign`` with *no* clocked always block, and the registered
clock-only rdata capture.
"""

from __future__ import annotations

from dataclasses import make_dataclass

import pytest

from spirehdl.spirehdl import Bool, Signal, UInt, Wire, reset_shared_cache
from spirehdl.spirehdl_module import Component
from spirehdl.spirehdl_simulator import Simulator
from spirehdl.primitives import RomPrimitive
from spirehdl.aggregate.aggregate_record import AggregateRecord


def _build_rom(init, *, depth, width, registered=False, name="rom"):
    addr_w = max(1, (depth - 1).bit_length())

    class Top(Component):
        def __init__(self):
            fields = {
                "addr": Signal(typ=UInt(addr_w), kind="input", name="addr"),
                "dout": Signal(typ=UInt(width), kind="output", name="dout"),
            }
            if registered:
                fields["re"] = Signal(typ=Bool(), kind="input", name="re")
            IO = make_dataclass("TopIO", [(k, Signal) for k in fields])
            self.io = IO(**fields)
            self.elaborate()

        def elaborate(self):
            rom = RomPrimitive(UInt(width), depth, init=init,
                               registered_read=registered, name=name).make_internal()
            rom.io.read_addr <<= self.io.addr
            if registered:
                rom.io.read_enable <<= self.io.re
            self.io.dout <<= rom.io.read_data

    reset_shared_cache()
    return Top().to_module(name="rom_top", with_clock=True, with_reset=True)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_init_length_mismatch_raises():
    with pytest.raises(ValueError, match="length == depth"):
        RomPrimitive(UInt(8), depth=4, init=[1, 2, 3])


def test_depth_zero_raises():
    with pytest.raises(ValueError, match="depth must be > 0"):
        RomPrimitive(UInt(8), depth=0, init=[])


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def test_emits_array_and_initial_block():
    m = _build_rom([0x10, 0x20, 0x30, 0x40], depth=4, width=8)
    v = m.to_verilog()
    assert "reg [7:0] rom[0:3];" in v
    assert "initial begin" in v
    assert "rom[0] = 8'd16;" in v
    assert "rom[3] = 8'd64;" in v


def test_async_rom_has_no_always_block():
    m = _build_rom([1, 2, 3, 4], depth=4, width=8)
    v = m.to_verilog()
    rom_block = v.split("// --- RomPrimitive")[1]
    # Pure async ROM: combinational read assign, no clocked always block at all.
    assert "assign read_data = rom[read_addr];" in v
    assert "always" not in rom_block


def test_registered_rom_emits_clocked_capture():
    m = _build_rom([1, 2, 3, 4], depth=4, width=8, registered=True)
    v = m.to_verilog()
    assert "reg [7:0] rom__rd0;" in v
    assert "always @(posedge clk) begin" in v
    assert "if (read_enable) rom__rd0 <= rom[read_addr];" in v
    assert "assign read_data = rom__rd0;" in v
    # No async-reset on the rom capture block (yosys sync-read idiom).
    assert "posedge clk or posedge rst" not in v.split("// --- RomPrimitive")[1]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def test_async_rom_reads_init():
    init = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
    m = _build_rom(init, depth=8, width=8)
    sim = Simulator(m)
    sim.deassert_reset()
    for a, exp in enumerate(init):
        sim.set("addr", a).eval()          # async: eval only, no step
        assert sim.get("dout") == exp


def test_registered_rom_latency_and_hold():
    init = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]
    m = _build_rom(init, depth=8, width=8, registered=True)
    sim = Simulator(m)
    sim.deassert_reset()
    sim.set("re", 1).set("addr", 0).step()
    assert sim.get("dout") == 0x10
    sim.set("addr", 5).step()
    assert sim.get("dout") == 0x60
    # read-enable low → registered output holds.
    sim.set("re", 0).set("addr", 2).step()
    assert sim.get("dout") == 0x60


def test_rom_contents_are_immutable():
    """No write port exists; stepping the clock must never change ROM contents."""
    init = [5, 6, 7, 8]
    m = _build_rom(init, depth=4, width=8)
    sim = Simulator(m)
    sim.deassert_reset()
    for _ in range(5):
        sim.step()
    for a, exp in enumerate(init):
        sim.set("addr", a).eval()
        assert sim.get("dout") == exp


# ---------------------------------------------------------------------------
# Aggregate element type
# ---------------------------------------------------------------------------

class _Bus(AggregateRecord):
    data  = Wire(UInt(8))
    valid = Wire(UInt(1))


def test_aggregate_rom():
    """Element type is an HDLAggregate; init entries are packed bit-patterns."""
    bus_w = _Bus().width  # 9 bits
    # pack (data, valid) → bits: valid is the MSB given field order (data low, valid high).
    init = [0x42 | (1 << 8), 0x7E | (0 << 8), 0x00, 0xFF | (1 << 8)]

    class Top(Component):
        def __init__(self):
            fields = {
                "addr": Signal(typ=UInt(2), kind="input", name="addr"),
                "dout_data": Signal(typ=UInt(8), kind="output", name="dout_data"),
                "dout_valid": Signal(typ=Bool(), kind="output", name="dout_valid"),
            }
            IO = make_dataclass("TopAggIO", [(k, Signal) for k in fields])
            self.io = IO(**fields)
            self.elaborate()

        def elaborate(self):
            rom = RomPrimitive(_Bus, depth=4, init=init, name="brom").make_internal()
            rom.io.read_addr <<= self.io.addr
            out = _Bus()
            out <<= rom.io.read_data
            self.io.dout_data <<= out.data
            self.io.dout_valid <<= out.valid

    reset_shared_cache()
    m = Top().to_module(name="aggrom", with_clock=True, with_reset=True)
    assert "reg [8:0] brom[0:3];" in m.to_verilog()
    sim = Simulator(m)
    sim.deassert_reset()
    sim.set("addr", 0).eval()
    assert sim.get("dout_data") == 0x42 and sim.get("dout_valid") == 1
    sim.set("addr", 1).eval()
    assert sim.get("dout_data") == 0x7E and sim.get("dout_valid") == 0
