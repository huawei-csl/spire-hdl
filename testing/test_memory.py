"""Tests for the Memory primitive — verilog emission, structural CSE, and simulation."""

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
    mem.write(addr=addr_w, data=din, enable=we)
    mem.reset(enable=clr, value=0)
    dout <<= mem[addr_r]
    return m, mem


def test_emits_memory_array_declaration():
    m, _ = _build_fifo16()
    v = m.to_verilog()
    # The whole point of Memory: the storage must emit as a verilog array,
    # not as N separate `reg` declarations (which yosys's memory pass would
    # then refuse to merge back).
    assert "reg [8:0] fifo[0:15];" in v
    # And NOT 16 scalar regs:
    assert not re.search(r"reg \[8:0\] fifo_\d+;", v)


def test_emits_reset_then_write_else_block():
    m, _ = _build_fifo16()
    v = m.to_verilog()
    # The verilog idiom we want yosys to see (matches benchmarks/dr_rtl/router):
    #   if (clr) begin fifo[0]<=0; ...; fifo[15]<=0; end else if (we) fifo[addr_w]<=din;
    assert "if (clr) begin" in v
    assert "fifo[0] <= 9'd0;" in v
    assert "fifo[15] <= 9'd0;" in v
    assert "end else if (we) begin" in v
    assert "fifo[addr_w] <= din;" in v


def test_combinational_read_emits_array_index():
    m, _ = _build_fifo16()
    v = m.to_verilog()
    # Async read: dout <<= mem[addr_r]; should emit `fifo[addr_r]`.
    assert "fifo[addr_r]" in v


def test_registered_read_emits_in_own_always_block():
    reset_shared_cache()
    m = Module("rom8")
    addr = m.input(UInt(3), "addr")
    re = m.input(Bool(), "re")
    dout = m.output(UInt(8), "dout")
    rom = Memory(UInt(8), depth=8, name="rom",
                 init=[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80])
    dout_reg = rom.registered_read(addr=addr, enable=re)
    dout <<= dout_reg
    v = m.to_verilog()
    # Decl + initial + clock-only always block emitting rdata <= mem[addr]
    assert "reg [7:0] rom[0:7];" in v
    assert "reg [7:0] rom_rdata;" in v
    assert "initial begin" in v
    assert "rom[0] = 8'd16;" in v
    assert "rom[7] = 8'd128;" in v
    assert "always @(posedge clk) begin" in v
    assert "rom_rdata <= rom[addr];" in v
    # No async-rst sensitivity on the memory always block (would prevent
    # yosys memory inference from firing cleanly).
    assert "posedge clk or posedge rst" not in v


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_init_length_mismatch_raises():
    with pytest.raises(ValueError, match="init must have length"):
        Memory(UInt(8), depth=4, init=[1, 2, 3], name="m")


def test_double_write_raises():
    reset_shared_cache()
    m = Module("dbl")
    we = m.input(Bool(), "we")
    addr = m.input(UInt(2), "addr")
    din = m.input(UInt(4), "din")
    mem = Memory(UInt(4), depth=4, name="m1")
    mem.write(addr=addr, data=din, enable=we)
    with pytest.raises(ValueError, match="single write port"):
        mem.write(addr=addr, data=din, enable=we)


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
    rom = Memory(UInt(8), depth=8, name="rom",
                 init=[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80])
    dout_reg = rom.registered_read(addr=addr, enable=re)
    dout <<= dout_reg

    sim = Simulator(m)
    sim.set("re", 1).set("addr", 0).step()
    assert sim.get("dout") == 0x10
    sim.set("addr", 5).step()
    assert sim.get("dout") == 0x60
    # With read-enable low the rdata register holds its last value.
    sim.set("re", 0).set("addr", 2).step()
    assert sim.get("dout") == 0x60


# ---------------------------------------------------------------------------
# Structural CSE — two MemReads at the same (mem, addr) collapse to one wire
# ---------------------------------------------------------------------------

def test_memread_canonical_key_groups_identical_reads():
    """Two MemReads of the same Memory at the same addr should land in the same CSE
    equivalence class — i.e., have identical canonical keys. (Whether the CSE pass
    emits one or two ``store[addr]`` lines is a separate wire-level concern; yosys's
    memory pass merges them into a single read port regardless.)
    """
    from collections import defaultdict
    from spirehdl.spirehdl import Expr
    from spirehdl.spirehdl_cse import _CseWalker

    reset_shared_cache()
    m = Module("dup_read")
    addr = m.input(UInt(3), "addr")
    a = m.output(UInt(8), "a")
    b = m.output(UInt(8), "b")
    mem = Memory(UInt(8), depth=8, name="store")
    a <<= mem[addr]
    b <<= mem[addr]
    m.collect_signals()

    walker = _CseWalker()
    for s in m._signals:
        if isinstance(getattr(s, "_driver", None), Expr):
            walker.visit(s._driver)

    by_key = defaultdict(list)
    for e in walker.all_ops:
        by_key[walker._cache[id(e)]].append(e)
    # Exactly one canonical-key class, containing both MemReads.
    assert len(by_key) == 1
    insts = next(iter(by_key.values()))
    assert len(insts) == 2
