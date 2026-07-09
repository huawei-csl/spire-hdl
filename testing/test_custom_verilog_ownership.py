"""Ownership boundaries: custom-Verilog tagging must not swallow parent signals; standalone emission of an
embedded child must not pull the parent's cone in."""
from spire import Component, CustomVerilogComponent, IORecord, Input, Output
from spire.expr import UInt, Wire, reset_shared_cache
from spire.simulator import Simulator


class _Scaler(CustomVerilogComponent):
    """Custom child whose input is wired from the enclosing context at construction (through the IO boundary)."""

    def __init__(self, src):
        self.io = IORecord(inp=Input(UInt(8)), out=Output(UInt(8)))
        self.io.inp <<= src
        self.elaborate()

    def elaborate(self):
        self.io.out <<= (self.io.inp + 1)[0:8]

    def custom_verilog(self):
        return f"  assign {self.io.out.name} = {self.io.inp.name} + 8'd1;"


class _Parent(Component):
    def __init__(self):
        self.io = IORecord(x=Input(UInt(8)), y=Output(UInt(8)), z=Output(UInt(8)))
        self.elaborate()

    def elaborate(self):
        w = Wire(UInt(8), name="w")
        w <<= (self.io.x * 2)[0:8]  # parent wire used by the parent AND passed into the custom child
        self.io.y <<= w
        self.sub = _Scaler(w)
        self.io.z <<= self.sub.io.out


def test_parent_wire_survives_custom_child_tagging():
    reset_shared_cache()
    m = _Parent().to_netlist()
    text = m.to_verilog()
    assert "wire [7:0] w;" in text, text          # declaration intact
    assert "assign w = " in text, text            # parent driver intact
    assert "assign out = inp + 8'd1;" in text, text  # custom block emitted against its own input leaf

    sim = Simulator(m)
    sim.set("x", 5)
    sim.eval()
    assert sim.get("y") == 10
    assert sim.get("z") == 11


class _AdderChild(CustomVerilogComponent):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(8)), s=Output(UInt(8)))
        self.elaborate()

    def elaborate(self):
        self.io.s <<= (self.io.a + 1)[0:8]

    def custom_verilog(self):
        return f"  assign {self.io.s.name} = {self.io.a.name} + 8'd1;"


def test_standalone_child_after_embedding_excludes_parent_cone():
    reset_shared_cache()

    class _Embedder(Component):
        def __init__(self):
            self.io = IORecord(x=Input(UInt(8)), z=Output(UInt(8)))
            self.elaborate()

        def elaborate(self):
            pw = Wire(UInt(8), name="parent_wire")
            pw <<= (self.io.x * 3)[0:8]
            self.sub = _AdderChild()
            self.sub.io.a <<= pw
            self.io.z <<= self.sub.io.s

    parent = _Embedder()
    child_text = parent.sub.to_netlist().to_verilog()
    assert "parent_wire" not in child_text, child_text  # no parent cone in the standalone child
    assert "* " not in child_text, child_text           # the parent's multiply must not leak in


class _PlainInc(Component):
    def __init__(self):
        self.io = IORecord(i=Input(UInt(8)), o=Output(UInt(8)))
        self.elaborate()

    def elaborate(self):
        self.io.o <<= (self.io.i + 1)[0:8]


class _CustomWithModelSub(CustomVerilogComponent):
    """Custom component whose simulation model is built FROM a plain sub-component."""

    def __init__(self):
        self.io = IORecord(a=Input(UInt(8)), y=Output(UInt(8)))
        self.elaborate()

    def elaborate(self):
        self.inc = _PlainInc()
        self.inc.io.i <<= self.io.a
        self.io.y <<= self.inc.io.o

    def custom_verilog(self):
        return "  assign y = a + 8'd1;"


def test_plain_subcomponent_inside_model_is_absorbed():
    # A plain component constructed inside a custom component's elaborate() is part of the
    # simulation model: none of its logic (nor the IO glue) may emit next to the custom block.
    reset_shared_cache()
    comp = _CustomWithModelSub()
    text = comp.to_netlist().to_verilog()
    assert "assign y = a + 8'd1;" in text, text
    assert "assign o =" not in text and "assign i =" not in text, text

    sim = Simulator(comp.to_netlist("model_sub_sim"))
    sim.set("a", 41)
    sim.eval()
    assert sim.get("y") == 42
