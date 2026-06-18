"""Memory behaviour tests, exercised through the memory *primitives* as DUT.

The built-in ``Memory`` class was removed when the core went lean (Middle path B): memory is
now a sim-only ``_MemoryArray`` wrapped by Component primitives. This suite checks the
classic memory semantics (async read/write, registered read with enable-hold, broadcast
reset, ROM init, and a multi-port RAM) using ``MemoryPrimitive`` / ``RamPrimitive`` as the
device under test. Emission-detail tests live in ``test_primitive_memory.py`` /
``test_primitive_ram.py``; this file focuses on behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

from spire.expr import Bool, Const, Signal, UInt, reset_shared_cache
from spire.component import Component
from spire.simulator import Simulator
from spire.primitives import MemoryPrimitive, RamPrimitive


# ---------------------------------------------------------------------------
# Async read/write RAM (MemoryPrimitive)
# ---------------------------------------------------------------------------

def _ram(depth=16, width=9, *, registered_read=False, with_reset_arm=True, init=None):
    class Top(Component):
        def __init__(self):
            @dataclass
            class IO:
                we: Signal
                clr: Signal
                aw: Signal
                ar: Signal
                din: Signal
                re: Signal
                dout: Signal
            addr_w = max(1, (depth - 1).bit_length())
            self.io = IO(
                we   = Signal(typ=Bool(), kind="input"),
                clr  = Signal(typ=Bool(), kind="input"),
                aw   = Signal(typ=UInt(addr_w), kind="input"),
                ar   = Signal(typ=UInt(addr_w), kind="input"),
                din  = Signal(typ=UInt(width), kind="input"),
                re   = Signal(typ=Bool(), kind="input"),
                dout = Signal(typ=UInt(width), kind="output"),
            )
            self.elaborate()

        def elaborate(self):
            mem = MemoryPrimitive(
                UInt(width), depth=depth, registered_read=registered_read,
                with_reset_arm=with_reset_arm, init=init, name="mem",
            ).make_internal()
            mem.io.write_addr   <<= self.io.aw
            mem.io.write_data   <<= self.io.din
            mem.io.write_enable <<= self.io.we
            if with_reset_arm:
                mem.io.reset_enable <<= self.io.clr
            mem.io.read_addr <<= self.io.ar
            if registered_read:
                mem.io.read_enable <<= self.io.re
            self.io.dout <<= mem.io.read_data

    reset_shared_cache()
    return Top().to_module(name="mem_top", with_clock=True, with_reset=True)


def test_write_then_read_back():
    sim = Simulator(_ram())
    sim.deassert_reset()
    sim.set("we", 1).set("aw", 3).set("din", 0xAB).step()
    sim.set("aw", 5).set("din", 0xCD).step()
    sim.set("we", 0)
    sim.set("ar", 3).eval(); assert sim.get("dout") == 0xAB
    sim.set("ar", 5).eval(); assert sim.get("dout") == 0xCD
    sim.set("ar", 7).eval(); assert sim.get("dout") == 0       # unwritten reads 0


def test_reset_arm_clears_all():
    sim = Simulator(_ram())
    sim.deassert_reset()
    sim.set("we", 1).set("aw", 3).set("din", 0xAB).step()
    sim.set("we", 0).set("clr", 1).step()
    sim.set("clr", 0).set("ar", 3).eval()
    assert sim.get("dout") == 0


# ---------------------------------------------------------------------------
# Registered-read ROM (MemoryPrimitive)
# ---------------------------------------------------------------------------

def test_rom_init_and_registered_read_hold():
    init = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]
    sim = Simulator(_ram(depth=8, width=8, registered_read=True, with_reset_arm=False, init=init))
    sim.deassert_reset()
    sim.set("re", 1).set("ar", 0).step(); assert sim.get("dout") == 0x10
    sim.set("ar", 5).step();              assert sim.get("dout") == 0x60
    # read-enable low → registered output holds.
    sim.set("re", 0).set("ar", 2).step(); assert sim.get("dout") == 0x60


# ---------------------------------------------------------------------------
# Multi-port: simple dual-port (1W + 2R) via RamPrimitive
# ---------------------------------------------------------------------------

def test_one_write_two_read_ports():
    class Top(Component):
        def __init__(self):
            @dataclass
            class IO:
                we: Signal
                wa: Signal
                wd: Signal
                ra0: Signal
                ra1: Signal
                d0: Signal
                d1: Signal
            self.io = IO(
                we  = Signal(typ=Bool(), kind="input"),
                wa  = Signal(typ=UInt(2), kind="input"),
                wd  = Signal(typ=UInt(8), kind="input"),
                ra0 = Signal(typ=UInt(2), kind="input"),
                ra1 = Signal(typ=UInt(2), kind="input"),
                d0  = Signal(typ=UInt(8), kind="output"),
                d1  = Signal(typ=UInt(8), kind="output"),
            )
            self.elaborate()

        def elaborate(self):
            ram = RamPrimitive(UInt(8), depth=4, num_write_ports=1, num_read_ports=2,
                               name="sdp").make_internal()
            ram.io.w0_addr <<= self.io.wa
            ram.io.w0_data <<= self.io.wd
            ram.io.w0_en   <<= self.io.we
            ram.io.r0_addr <<= self.io.ra0
            ram.io.r1_addr <<= self.io.ra1
            self.io.d0 <<= ram.io.r0_data
            self.io.d1 <<= ram.io.r1_data

    reset_shared_cache()
    m = Top().to_module(name="sdp_top", with_clock=True, with_reset=True)
    assert "reg [7:0] sdp[0:3];" in m.to_verilog()    # single shared array, two read assigns
    sim = Simulator(m)
    sim.deassert_reset()
    sim.set("we", 1).set("wa", 1).set("wd", 0x11).step()
    sim.set("wa", 2).set("wd", 0x22).step()
    sim.set("we", 0)
    sim.set("ra0", 1).set("ra1", 2).eval()
    assert sim.get("d0") == 0x11
    assert sim.get("d1") == 0x22
