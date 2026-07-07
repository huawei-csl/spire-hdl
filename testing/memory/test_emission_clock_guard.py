"""Emission clock guard: sequential primitives cannot emit clockless RTL; async ROMs may.

Custom-verilog primitives tag their internal registers `_no_emit_drive`, so the guard must count
ALL registers and clock-needing memories — while a write-less, unregistered ROM (`initial` +
`assign` only) legitimately emits without a clock.
"""
import pytest

from spire import Component, Input, IORecord, Output, UInt, Bool
from spire.expr import Register, reset_shared_cache
from spire.ir import Netlist
from spire.primitives import FIFOPrimitive, RomPrimitive


class _FifoTop(Component):
    def __init__(self):
        self.io = IORecord(push=Input(Bool()), pop=Input(Bool()), din=Input(UInt(8)),
                           dout=Output(UInt(8)), full=Output(Bool()), empty=Output(Bool()))
        self.elaborate()

    def elaborate(self):
        fifo = FIFOPrimitive(UInt(8), depth=4)
        fifo.io.push <<= self.io.push
        fifo.io.pop <<= self.io.pop
        fifo.io.din <<= self.io.din
        self.io.dout <<= fifo.io.dout
        self.io.full <<= fifo.io.full
        self.io.empty <<= fifo.io.empty


class _RomTop(Component):
    def __init__(self, registered: bool):
        self._registered = registered
        fields = dict(addr=Input(UInt(2)), dout=Output(UInt(8)))
        if registered:
            fields["re"] = Input(Bool())
        self.io = IORecord(**fields)
        self.elaborate()

    def elaborate(self):
        rom = RomPrimitive(UInt(8), depth=4, init=[10, 20, 30, 40], registered_read=self._registered)
        rom.io.read_addr <<= self.io.addr
        if self._registered:
            rom.io.read_enable <<= self.io.re
        self.io.dout <<= rom.io.read_data


def test_fifo_default_emit_raises_instead_of_clockless_rtl():
    reset_shared_cache()
    with pytest.raises(ValueError, match="with_clock=True"):
        _FifoTop().to_verilog("fifo_top")  # default: with_clock=False


def test_fifo_with_clock_emits_clocked_rtl():
    reset_shared_cache()
    v = _FifoTop().to_verilog("fifo_top", with_clock=True, with_reset=True)
    assert "posedge clk" in v and " clk;" in v


def test_async_rom_emits_clockless():
    reset_shared_cache()
    v = _RomTop(registered=False).to_verilog("rom_top")  # no clock needed at all
    assert "always @" not in v and " clk" not in v
    assert "initial begin" in v or "initial " in v


def test_registered_rom_requires_clock():
    reset_shared_cache()
    with pytest.raises(ValueError, match="with_clock=True"):
        _RomTop(registered=True).to_verilog("rom_top")
    v = _RomTop(registered=True).to_verilog("rom_top", with_clock=True, with_reset=True)
    assert "posedge clk" in v


def test_plain_register_still_guarded():
    reset_shared_cache()
    m = Netlist("plain", with_clock=False, with_reset=False)
    d = m.input(UInt(4), "d")
    q = m.output(UInt(4), "q")
    r = Register(UInt(4), init=0, name="r")
    r <<= d
    q <<= r
    with pytest.raises(ValueError, match="no clock"):
        m.to_verilog()
