"""A custom-Verilog component whose *simulation model* contains other custom-Verilog components
(memory primitives, blackboxes). Those children belong to the parent's elaborate() cone: the parent's
custom block replaces them, so neither their blocks nor the logic feeding them may leak into the Verilog."""
import shutil
import subprocess

import pytest

from spire import Component, CustomVerilogComponent, IORecord, Input, Output
from spire.expr import Register, UInt, mux, reset_shared_cache
from spire.primitives import RomPrimitive
from spire.simulator import Simulator


class _Leaf(CustomVerilogComponent):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(4)), y=Output(UInt(4)))
        self.elaborate()

    def elaborate(self):
        pass

    def custom_verilog(self):
        return f"  assign {self.io.y.name} = {self.io.a.name};  // LEAF_BLOCK"


class _RomModel(CustomVerilogComponent):
    """Sim model: an address counter (only reaches the output through the ROM) reading a ROM.
    Synthesis: a hand-written block."""

    def __init__(self, with_blackbox=False):
        self.with_blackbox = with_blackbox
        self.io = IORecord(step=Input(UInt(1)), q=Output(UInt(8)))
        self.elaborate()

    def elaborate(self):
        rom = RomPrimitive(UInt(8), 16, init=[(7 * i + 3) & 0xFF for i in range(16)], name="model_rom")
        addr = Register(UInt(4), init=0, name="model_addr")
        addr <<= (addr + self.io.step)[0:4]
        rom.io.read_addr <<= addr
        if self.with_blackbox:
            bb = _Leaf()
            bb.io.a <<= (addr ^ 5)[0:4]            # feeds only a blackbox input
        d = rom.io.read_data
        model_sel = (addr + 3)[0:4] > 6                     # compound operands: width isolation must not wire them
        self.io.q <<= mux(model_sel, (d + addr)[0:8], (d ^ 0x55) - addr * 2)[0:8]

    def custom_verilog(self):
        return f"  assign {self.io.q.name} = 8'hA5;  // PARENT_BLOCK"


class _Top(Component):
    def __init__(self, with_blackbox=False):
        self.with_blackbox = with_blackbox
        self.io = IORecord(step=Input(UInt(1)), q=Output(UInt(8)))
        self.elaborate()

    def elaborate(self):
        m = _RomModel(self.with_blackbox)
        m.io.step <<= self.io.step
        self.io.q <<= m.io.q


@pytest.mark.parametrize("with_blackbox", [False, True])
def test_nested_custom_children_are_absorbed(with_blackbox, tmp_path):
    reset_shared_cache()
    text = _Top(with_blackbox).to_verilog(name="top", with_clock=True, with_reset=True)
    assert "PARENT_BLOCK" in text, text
    assert "model_rom" not in text, text          # the ROM's own block is suppressed
    assert "model_addr" not in text, text         # logic feeding only the ROM / blackbox does not leak
    assert "LEAF_BLOCK" not in text, text
    assert "model_sel" not in text, text
    if shutil.which("iverilog"):
        src = tmp_path / "top.v"
        src.write_text(text)
        r = subprocess.run(["iverilog", "-g2012", "-o", str(tmp_path / "a.out"), str(src)], text=True, capture_output=True)
        assert r.returncode == 0, r.stderr + text


def test_nested_custom_children_still_simulate():
    reset_shared_cache()
    sim = Simulator(_Top().to_netlist("top", with_clock=True, with_reset=True))
    sim.reset()
    sim.deassert_reset()
    seen = []
    for _ in range(4):
        sim.set("step", 1).eval()
        seen.append(sim.get("q"))
        sim.step()
    def model(i):
        dd = (7 * i + 3) & 0xFF
        return ((dd + i) & 0xFF) if ((i + 3) & 15) > 6 else ((dd ^ 0x55) - ((i * 2) & 0xFF)) & 0xFF
    assert seen == [model(i) for i in range(4)]


def test_absorbed_child_emits_standalone():
    # a primitive emitted on its own (as the top) keeps its block even if marked absorbed elsewhere
    reset_shared_cache()
    standalone = RomPrimitive(UInt(8), 4, init=[1, 2, 3, 4], name="solo_rom")
    standalone._absorbed_by = object()
    text = standalone.to_verilog(name="solo")
    assert "solo_rom" in text, text
