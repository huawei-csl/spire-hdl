"""Port-name uniqueness and deterministic emission (names are a pure function of the graph)."""
import pytest

from spire import Component, IORecord, Input, Output
from spire.expr import UInt
from spire.ir import Netlist


def test_duplicate_port_name_rejected():
    m = Netlist("dup", with_clock=False, with_reset=False)
    m.input(UInt(4), "a")
    with pytest.raises(ValueError, match="Duplicate port name"):
        m.input(UInt(4), "a")
    with pytest.raises(ValueError, match="Duplicate port name"):
        m.output(UInt(4), "a")


def test_port_name_collision_with_implicit_clock_rejected():
    m = Netlist("dup_clk")  # implicit clk/rst
    with pytest.raises(ValueError, match="Duplicate port name"):
        m.input(UInt(1), "clk")


class _Demo(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(4)), b=Input(UInt(4)), z=Output(UInt(9)))
        self.elaborate()

    def elaborate(self):
        t = (self.io.a - self.io.b) & 0x7
        self.io.z <<= (t + self.io.a) * 2


def test_same_process_reemission_is_byte_identical():
    # No manual cache reset: two fresh builds in one process must emit identical text, including the
    # default module name (the class name) and all auto-generated wire names.
    t1 = _Demo().to_verilog()
    t2 = _Demo().to_verilog()
    assert "module _Demo (" in t1
    assert t1 == t2


@pytest.mark.xfail(strict=True, reason="known: parent emission renames shared child ports in place "
                                       "(pure-emission plan item; fix = per-netlist naming overlay)")
def test_child_ports_survive_parent_emission():
    """Emission must be side-effect-free on construction state: emitting a parent whose ports
    collide with an embedded child's must not rename the child's ports for later emissions."""
    class Child(Component):
        def __init__(self):
            self.io = IORecord(a=Input(UInt(4)), y=Output(UInt(4)))
            self.elaborate()

        def elaborate(self):
            self.io.y <<= self.io.a + 1

    class Parent(Component):
        def __init__(self):
            self.io = IORecord(a=Input(UInt(4)), y=Output(UInt(4)))  # collides with the child's
            self.elaborate()

        def elaborate(self):
            self.child = Child()
            self.child.io.a <<= self.io.a
            self.io.y <<= self.child.io.y

    reset_shared_cache()
    p = Parent()
    before = p.child.to_verilog("child_v")
    p.to_verilog("parent_v")
    after = p.child.to_verilog("child_v")
    assert before == after, "child emission changed because the parent was emitted"
