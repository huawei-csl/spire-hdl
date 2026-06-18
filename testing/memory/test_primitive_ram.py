"""Tests for ``RamPrimitive`` — multi-port / true-dual-port / masked / read-under-write.

These exercise the shapes the single-port ``MemoryPrimitive`` cannot express, proving the
shape-agnostic ``_MemoryArray`` port factory: two read/write ports over one array (2RW),
per-chunk write masks, and writeFirst read-under-write forwarding.
"""

from __future__ import annotations

from dataclasses import make_dataclass

import pytest

from spire.expr import Bool, Signal, UInt, reset_shared_cache
from spire.component import Component
from spire.simulator import Simulator
from spire.primitives import RamPrimitive


def _wire(dst, src):
    """`dst <<= src` for dynamically-fetched Signal attributes (avoids function-call LHS)."""
    dst <<= src


def _top_from(ram_factory, io_fields, wiring):
    """Build a parent Component that exposes a RamPrimitive's ports and returns its Module."""
    class Top(Component):
        def __init__(self):
            IO = make_dataclass("TopIO", [(n, Signal) for n, _ in io_fields])
            self.io = IO(**{n: Signal(typ=t, kind=d, name=n) for n, (t, d) in
                            ((n, (t, d)) for n, (t, d) in io_fields)})
            self.elaborate()

        def elaborate(self):
            ram = ram_factory().make_internal()
            wiring(self, ram)

    reset_shared_cache()
    return Top().to_module(name="ram_top", with_clock=True, with_reset=True)


# ---------------------------------------------------------------------------
# True dual-port (2RW)
# ---------------------------------------------------------------------------

def _build_2rw(depth=4, elem_w=8, ruw="readFirst"):
    io_fields = []
    for io in ("a", "b"):
        io_fields += [
            (f"{io}_addr",  (UInt(2), "input")),
            (f"{io}_din",   (UInt(elem_w), "input")),
            (f"{io}_write", (Bool(), "input")),
            (f"{io}_en",    (Bool(), "input")),
            (f"{io}_dout",  (UInt(elem_w), "output")),
        ]

    def wiring(top, ram):
        for p, io in (("rw0", "a"), ("rw1", "b")):
            _wire(getattr(ram.io, f"{p}_addr"),  getattr(top.io, f"{io}_addr"))
            _wire(getattr(ram.io, f"{p}_din"),   getattr(top.io, f"{io}_din"))
            _wire(getattr(ram.io, f"{p}_write"), getattr(top.io, f"{io}_write"))
            _wire(getattr(ram.io, f"{p}_en"),    getattr(top.io, f"{io}_en"))
            _wire(getattr(top.io, f"{io}_dout"), getattr(ram.io, f"{p}_dout"))

    return _top_from(
        lambda: RamPrimitive(UInt(elem_w), depth=depth, rw_ports=2,
                             num_read_ports=0, num_write_ports=0,
                             read_under_write=ruw, name="dp"),
        io_fields, wiring)


def test_2rw_emits_single_array():
    m = _build_2rw()
    v = m.to_verilog()
    assert "reg [7:0] dp[0:3];" in v          # one shared array
    # Two write arms over the same array, each gated by en & write.
    assert "if ((rw0_en && rw0_write)) dp[rw0_addr] <= rw0_din;" in v
    assert "if ((rw1_en && rw1_write)) dp[rw1_addr] <= rw1_din;" in v
    assert "assign rw0_dout = dp[rw0_addr];" in v
    assert "assign rw1_dout = dp[rw1_addr];" in v


def test_2rw_independent_ports():
    m = _build_2rw()
    sim = Simulator(m)
    sim.deassert_reset()
    # A writes addr1=0xAB; B writes addr2=0xCD, same cycle.
    sim.set("a_en", 1).set("a_write", 1).set("a_addr", 1).set("a_din", 0xAB)
    sim.set("b_en", 1).set("b_write", 1).set("b_addr", 2).set("b_din", 0xCD).step()
    sim.set("a_write", 0).set("b_write", 0)
    sim.set("a_addr", 1).set("b_addr", 2).eval()
    assert sim.get("a_dout") == 0xAB
    assert sim.get("b_dout") == 0xCD
    # Cross-read: A reads what B wrote, B reads what A wrote.
    sim.set("a_addr", 2).set("b_addr", 1).eval()
    assert sim.get("a_dout") == 0xCD
    assert sim.get("b_dout") == 0xAB


# ---------------------------------------------------------------------------
# Write mask (per-chunk byte enable)
# ---------------------------------------------------------------------------

def _build_masked(depth=4, elem_w=16, mask_chunks=2):
    io_fields = [
        ("w_addr", (UInt(2), "input")),
        ("w_data", (UInt(elem_w), "input")),
        ("w_en",   (Bool(), "input")),
        ("w_mask", (UInt(mask_chunks), "input")),
        ("r_addr", (UInt(2), "input")),
        ("r_data", (UInt(elem_w), "output")),
    ]

    def wiring(top, ram):
        _wire(ram.io.w0_addr, top.io.w_addr)
        _wire(ram.io.w0_data, top.io.w_data)
        _wire(ram.io.w0_en,   top.io.w_en)
        _wire(ram.io.w0_mask, top.io.w_mask)
        _wire(ram.io.r0_addr, top.io.r_addr)
        _wire(top.io.r_data,  ram.io.r0_data)

    return _top_from(
        lambda: RamPrimitive(UInt(elem_w), depth=depth, num_write_ports=1,
                             num_read_ports=1, mask_chunks=mask_chunks, name="mr"),
        io_fields, wiring)


def test_masked_write_emits_per_chunk():
    m = _build_masked()
    v = m.to_verilog()
    assert "if (w0_mask[0]) mr[w0_addr][7:0] <= w0_data[7:0];" in v
    assert "if (w0_mask[1]) mr[w0_addr][15:8] <= w0_data[15:8];" in v


def test_masked_write_rmw():
    m = _build_masked()
    sim = Simulator(m)
    sim.deassert_reset()
    # Write only low byte (mask=0b01) of 0xFFFF.
    sim.set("w_en", 1).set("w_addr", 0).set("w_data", 0xFFFF).set("w_mask", 0b01).step()
    sim.set("w_en", 0).set("r_addr", 0).eval()
    assert sim.get("r_data") == 0x00FF
    # Now write only high byte (mask=0b10) of 0xAAAA — low byte must survive.
    sim.set("w_en", 1).set("w_data", 0xAAAA).set("w_mask", 0b10).step()
    sim.set("w_en", 0).set("r_addr", 0).eval()
    assert sim.get("r_data") == 0xAAFF


# ---------------------------------------------------------------------------
# Read-under-write: writeFirst forwarding on an rw port
# ---------------------------------------------------------------------------

def _build_rw_writefirst(depth=4, elem_w=8):
    io_fields = [
        ("addr",  (UInt(2), "input")),
        ("din",   (UInt(elem_w), "input")),
        ("write", (Bool(), "input")),
        ("en",    (Bool(), "input")),
        ("dout",  (UInt(elem_w), "output")),
    ]

    def wiring(top, ram):
        _wire(ram.io.rw0_addr,  top.io.addr)
        _wire(ram.io.rw0_din,   top.io.din)
        _wire(ram.io.rw0_write, top.io.write)
        _wire(ram.io.rw0_en,    top.io.en)
        _wire(top.io.dout,      ram.io.rw0_dout)

    return _top_from(
        lambda: RamPrimitive(UInt(elem_w), depth=depth, rw_ports=1,
                             num_read_ports=0, num_write_ports=0,
                             read_under_write="writeFirst", name="wf"),
        io_fields, wiring)


def test_writefirst_emits_forwarding_mux():
    m = _build_rw_writefirst()
    v = m.to_verilog()
    assert "assign rw0_dout = ((rw0_en && rw0_write) && (rw0_addr == rw0_addr)) ? rw0_din : wf[rw0_addr];" in v


def test_writefirst_returns_new_data_same_cycle():
    m = _build_rw_writefirst()
    sim = Simulator(m)
    sim.deassert_reset()
    # Same-cycle write+read at addr 1: writeFirst → dout shows the new data immediately.
    sim.set("en", 1).set("write", 1).set("addr", 1).set("din", 0x55).eval()
    assert sim.get("dout") == 0x55
    # readFirst control: a plain readback after committing.
    sim.step()
    sim.set("write", 0).set("addr", 1).eval()
    assert sim.get("dout") == 0x55
