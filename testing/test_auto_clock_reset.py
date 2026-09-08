"""Auto clock/reset: Component.to_netlist()/to_verilog() flags left at None follow the design — clk/rst are
added iff it holds registers or clocked memories; explicit True/False always wins."""
import pytest

from spire import Component, IORecord, Input, Output, UInt, Bool
from spire.expr import Register, reset_shared_cache
from spire.primitives import FIFOPrimitive, RomPrimitive
from spire.simulator import Simulator


class _Adder(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(8)), b=Input(UInt(8)), y=Output(UInt(9)))
        self.elaborate()

    def elaborate(self):
        self.io.y <<= self.io.a + self.io.b


class _Acc(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(8)), y=Output(UInt(16)))
        self.elaborate()

    def elaborate(self):
        acc = Register(UInt(16), init=0, name="acc")
        acc <<= acc + self.io.a
        self.io.y <<= acc


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


class _AsyncRomTop(Component):
    def __init__(self):
        self.io = IORecord(addr=Input(UInt(2)), dout=Output(UInt(8)))
        self.elaborate()

    def elaborate(self):
        rom = RomPrimitive(UInt(8), depth=4, init=[10, 20, 30, 40], registered_read=False)
        rom.io.read_addr <<= self.io.addr
        self.io.dout <<= rom.io.read_data


def _port_names(m):
    return [p.name for p in m._ports]


def test_combinational_design_gets_no_clock_or_reset():
    reset_shared_cache()
    m = _Adder().to_netlist("add")
    assert (m.with_clock, m.with_reset) == (False, False)
    assert _port_names(m) == ["a", "b", "y"]
    v = _Adder().to_verilog("add")
    assert " clk" not in v and " rst" not in v and "always @" not in v


def test_registered_design_gets_clock_and_reset_ahead_of_data_ports():
    reset_shared_cache()
    m = _Acc().to_netlist("acc")
    assert (m.with_clock, m.with_reset) == (True, True)
    assert _port_names(m) == ["clk", "rst", "a", "y"]
    assert m.clk is m._signals[0] and m.rst is m._signals[1]


def test_auto_verilog_identical_to_explicit_flags():
    reset_shared_cache()
    auto = _Acc().to_verilog("acc")
    reset_shared_cache()
    explicit = _Acc().to_verilog("acc", with_clock=True, with_reset=True)
    assert auto == explicit
    assert "posedge clk or posedge rst" in auto


def test_explicit_false_on_registered_design_still_raises():
    reset_shared_cache()
    with pytest.raises(ValueError, match="with_clock=True"):
        _Acc().to_verilog("acc", with_clock=False)


def test_explicit_clock_only_derives_reset_from_registers():
    reset_shared_cache()
    assert _Acc().to_netlist("acc", with_clock=True).with_reset is True
    assert _Acc().to_netlist("acc", with_clock=True, with_reset=False).with_reset is False
    comb = _Adder().to_netlist("add", with_clock=True)
    assert (comb.with_clock, comb.with_reset) == (True, False)
    assert _port_names(comb) == ["clk", "a", "b", "y"]


def test_explicit_reset_implies_clock():
    reset_shared_cache()
    m = _Adder().to_netlist("add", with_reset=True)
    assert (m.with_clock, m.with_reset) == (True, True)


def test_write_port_memory_is_clocked_async_rom_is_not():
    reset_shared_cache()
    fifo = _FifoTop().to_netlist("fifo_top")
    assert (fifo.with_clock, fifo.with_reset) == (True, True)
    assert "posedge clk" in _FifoTop().to_verilog("fifo_top")
    reset_shared_cache()
    rom = _AsyncRomTop().to_netlist("rom_top")
    assert (rom.with_clock, rom.with_reset) == (False, False)


def test_to_aag_and_analyze_and_simulator_accept_registered_component_flagless():
    reset_shared_cache()
    assert _Acc().to_aag("acc")[0].startswith("aag ")
    _Acc().analyze("acc")
    sim = Simulator(_Acc())
    assert (sim.m.with_clock, sim.m.with_reset) == (True, True)
