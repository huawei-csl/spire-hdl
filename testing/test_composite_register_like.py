"""``CompositeRegister.like(value)`` and register-named ``.value`` views.

``like`` builds a register from a value you already hold (no constructor arguments to restate);
``.value`` names its leaves after the register so the Verilog stays readable. Together they make
a one-stage pipeline by hand the same thing ``pipeline(value)`` builds.
"""
import pytest

from spire import Component, IORecord, Input, Output, Wire, pipeline
from spire.composite.array import Array
from spire.composite.fixed_point import FixedPoint, FixedPointType
from spire.composite.record import CompositeRecord
from spire.composite.register import CompositeRegister
from spire.expr import Bool, SInt, UInt, cat, reset_shared_cache
from spire.ir import Netlist
from spire.simulator import Simulator

Q8_8 = FixedPointType(width_total=16, width_frac=8, signed=True)


class _Packet(CompositeRecord):
    def __init__(self, width: int = 8):
        super().__init__(valid=Wire(Bool()), data=Wire(UInt(width)))


def test_like_from_a_record_value():
    reset_shared_cache()
    a = _Packet(12)
    r = CompositeRegister.like(a, name="pk")

    assert r.width == a.width == 13
    assert r.bits.kind == "reg" and r.bits.name == "pk"
    v = r.value
    assert isinstance(v, _Packet)
    assert [leaf.name for leaf in v.to_list()] == ["pk_valid", "pk_data"]
    assert v.data.typ.width == 12
    assert a.valid._driver is None  # template left untouched


def test_like_from_an_array_value():
    reset_shared_cache()
    a = Array([Wire(UInt(4)), Wire(SInt(6))])
    v = CompositeRegister.like(a, name="vec").value
    assert isinstance(v, Array) and len(v) == 2
    assert [leaf.name for leaf in v.to_list()] == ["vec_0", "vec_1"]
    assert v[1].typ.signed


def test_like_from_a_fixed_point_view():
    reset_shared_cache()
    a = FixedPoint(Q8_8, bits=Wire(SInt(16)) + 1)  # a view, not an owning wire
    r = CompositeRegister.like(a, name="acc", init=0)
    assert r.bits.typ.signed and r.bits._init.value == 0
    v = r.value
    assert isinstance(v, FixedPoint) and v.ftype is a.ftype
    assert v.bits.name == "acc_q"  # single leaf: <name>_q


def test_constructor_path_names_its_view_the_same_way():
    reset_shared_cache()
    assert CompositeRegister(FixedPoint, Q8_8, name="acc").value.bits.name == "acc_q"
    v = CompositeRegister(_Packet, 8, name="pk").value
    assert [leaf.name for leaf in v.to_list()] == ["pk_valid", "pk_data"]


def test_like_rejects_composite_init_and_registers():
    reset_shared_cache()
    a = _Packet()
    with pytest.raises(ValueError, match="constant packed value"):
        CompositeRegister.like(a, init=a)
    with pytest.raises(TypeError, match="is storage, not a shape"):
        CompositeRegister.like(CompositeRegister.like(a))


def test_one_stage_by_hand_simulates():
    reset_shared_cache()

    class Dut(Component):
        def __init__(self):
            self.io = IORecord(v=Input(Bool()), d=Input(UInt(8)), y=Output(UInt(9)))
            self.elaborate()

        def elaborate(self):
            a = _Packet(8)
            a.valid <<= self.io.v
            a.data <<= self.io.d
            pipr = CompositeRegister.like(a, name="a_reg")  # the by-hand pipeline stage
            pipr <<= a
            b = pipr.value  # bind once; every read builds a new view
            self.io.y <<= cat(b.data, b.valid)

    m = Dut().to_netlist("byhand", with_clock=True)
    text = m.to_verilog()
    assert "reg [8:0] a_reg;" in text
    assert "wire  a_reg_valid;" in text and "wire [7:0] a_reg_data;" in text

    sim = Simulator(m)
    sim.set("v", 1)
    sim.set("d", 0x3C)
    sim.eval()
    assert sim.peek("y") == 0
    sim.step()
    sim.eval()
    assert sim.peek("y") == (1 << 8) | 0x3C


def test_pipeline_one_cycle_matches_the_by_hand_stage():
    reset_shared_cache()
    a = _Packet(8)
    by_hand = CompositeRegister.like(a, name="a_d1")
    by_hand <<= a
    lhs = by_hand.value
    rhs = pipeline(a, name="a")

    assert type(lhs) is type(rhs)
    assert [l.name for l in lhs.to_list()] == [l.name for l in rhs.to_list()] == ["a_d1_valid", "a_d1_data"]
    assert [l.typ.width for l in lhs.to_list()] == [l.typ.width for l in rhs.to_list()]
