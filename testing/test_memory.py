"""Tests for the Memory primitive — verilog emission, structural CSE, and simulation.

Ports are wired with ``<<=`` like any other Signal; there are no ``write/reset/
registered_read`` methods. The verilog idiom keeps yosys-recognisable shapes
(clock-only always block, no async-rst on memory state) — the per-port wires
add one identity ``assign`` per port that yosys folds away during synthesis.
"""

from __future__ import annotations

import re

import pytest

from spirehdl.spirehdl import (
    Bool,
    Memory,
    UInt,
    reset_shared_cache,
)
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator


# ---------------------------------------------------------------------------
# Verilog emission
# ---------------------------------------------------------------------------

def _build_fifo16():
    reset_shared_cache()
    m = Module("fifo16", with_reset=True)
    we = m.input(Bool(), "we")
    clr = m.input(Bool(), "clr")
    addr_w = m.input(UInt(4), "addr_w")
    addr_r = m.input(UInt(4), "addr_r")
    din = m.input(UInt(9), "din")
    dout = m.output(UInt(9), "dout")
    mem = Memory(UInt(9), depth=16, name="fifo")
    mem.write_addr   <<= addr_w
    mem.write_data   <<= din
    mem.write_enable <<= we
    mem.reset_enable <<= clr        # reset_value defaults to 0
    mem.read_addr    <<= addr_r
    dout             <<= mem.read_data
    return m, mem


def test_emits_memory_array_declaration():
    m, _ = _build_fifo16()
    v = m.to_verilog()
    # The whole point of Memory: storage emits as a verilog array, not as N
    # separate `reg` declarations (which yosys's memory pass would refuse to
    # merge back).
    assert "reg [8:0] fifo[0:15];" in v
    assert not re.search(r"reg \[8:0\] fifo_\d+;", v)


def test_emits_reset_then_write_else_block():
    m, _ = _build_fifo16()
    v = m.to_verilog()
    # Reset arm uses the memory's reset_enable port wire (driven by `clr`).
    # Write arm uses the memory's write_enable port wire (driven by `we`).
    # The yosys-recognised idiom: if-reset-then-clear-else-if-we-write.
    assert "if (fifo__rstn) begin" in v
    assert "fifo[0] <= fifo__rv;" in v
    assert "fifo[15] <= fifo__rv;" in v
    assert "end else if (fifo__we) begin" in v
    assert "fifo[fifo__waddr] <= fifo__wdata;" in v
    # And the identity assigns connect the user's signals through the port wires.
    assert "assign fifo__rstn = clr;" in v
    assert "assign fifo__we = we;" in v
    assert "assign fifo__waddr = addr_w;" in v


def test_combinational_read_emits_array_index():
    m, _ = _build_fifo16()
    v = m.to_verilog()
    # Async read: read_data wire's driver is an array-index leaf that emits as
    # `mem[raddr_wire]`.
    assert "assign fifo__rdata = fifo[fifo__raddr];" in v
    # And dout consumes read_data via the normal output-assign loop.
    assert "assign dout = fifo__rdata;" in v


def test_registered_read_emits_in_own_always_block():
    reset_shared_cache()
    m = Module("rom8")
    addr = m.input(UInt(3), "addr")
    re = m.input(Bool(), "re")
    dout = m.output(UInt(8), "dout")
    rom = Memory(UInt(8), depth=8, name="rom", registered_read=True,
                 init=[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80])
    rom.read_addr   <<= addr
    rom.read_enable <<= re
    dout            <<= rom.read_data
    v = m.to_verilog()
    # rdata wire is a reg, declared at module scope; clock-only always block
    # captures it inside the memory's own always block — yosys idiom intact.
    assert "reg [7:0] rom[0:7];" in v
    assert "reg [7:0] rom__rdata;" in v
    assert "initial begin" in v
    assert "rom[0] = 8'd16;" in v
    assert "rom[7] = 8'd128;" in v
    assert "always @(posedge clk) begin" in v
    assert "rom__rdata <= rom[rom__raddr];" in v
    # No async-rst sensitivity on the memory always block.
    assert "posedge clk or posedge rst" not in v


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_init_length_mismatch_raises():
    with pytest.raises(ValueError, match="init must have length"):
        Memory(UInt(8), depth=4, init=[1, 2, 3], name="m")


def test_partial_write_port_raises():
    """Connecting write_addr but not write_data (or vice versa) should fail validation."""
    reset_shared_cache()
    m = Module("partial")
    addr = m.input(UInt(2), "addr")
    dout = m.output(UInt(4), "dout")
    mem = Memory(UInt(4), depth=4, name="m1")
    mem.write_addr <<= addr
    mem.read_addr  <<= addr
    dout           <<= mem.read_data
    # write_data NOT connected → partial write port
    with pytest.raises(ValueError, match="write_addr and write_data"):
        m.to_verilog()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def test_sim_write_then_read():
    m, _ = _build_fifo16()
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
    m, _ = _build_fifo16()
    sim = Simulator(m)
    sim.deassert_reset()
    sim.set("we", 1).set("addr_w", 3).set("din", 0xAB).step()
    sim.set("we", 0)
    sim.set("clr", 1).step()
    sim.set("clr", 0)
    sim.set("addr_r", 3).eval()
    assert sim.get("dout") == 0


def test_sim_rom_init_and_registered_read():
    reset_shared_cache()
    m = Module("rom8")
    addr = m.input(UInt(3), "addr")
    re = m.input(Bool(), "re")
    dout = m.output(UInt(8), "dout")
    rom = Memory(UInt(8), depth=8, name="rom", registered_read=True,
                 init=[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80])
    rom.read_addr   <<= addr
    rom.read_enable <<= re
    dout            <<= rom.read_data

    sim = Simulator(m)
    sim.set("re", 1).set("addr", 0).step()
    assert sim.get("dout") == 0x10
    sim.set("addr", 5).step()
    assert sim.get("dout") == 0x60
    # With read-enable low the rdata register holds its last value.
    sim.set("re", 0).set("addr", 2).step()
    assert sim.get("dout") == 0x60
