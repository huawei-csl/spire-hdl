"""``HDLComposite.get_wire_clone()``: one generic shape clone for every composite.

The clone copies *structure* (fresh wires per leaf, recursively) and shares *description*
(widths, formats, configs). ``wire_like(template)`` is a shim onto it; subclasses override only
where construction genuinely differs — a type-driven factory (`FixedPoint` / `FloatingPoint`), a
class-fixed shape (`TemplateRecord`), or a value that must not be cloned (`CompositeRegister`).
"""
from dataclasses import dataclass

import pytest

from spire.arithmetic.int_arithmetic_config import AdderConfig, MultiplierConfig
from spire.composite.array import Array
from spire.composite.base import HDLComposite
from spire.composite.fixed_point import FixedPoint, FixedPointType
from spire.composite.floating_point import FloatingPoint, FloatingPointType
from spire.composite.record import CompositeRecord, TemplateRecord
from spire.composite.register import CompositeRegister
from spire.expr import Bool, Const, Input, Output, SInt, UInt, Wire, reset_shared_cache
from spire.hdl_traits import BitSerializable

Q8_8 = FixedPointType(width_total=16, width_frac=8, signed=True)
FT16 = FloatingPointType(exponent_width=5, fraction_width=10)


class _Packet(CompositeRecord):
    """Shape comes from a constructor argument the record does not keep verbatim."""

    def __init__(self, lanes: int = 2, width: int = 8):
        self.lanes = lanes  # plain attribute: description, not structure
        super().__init__(flag=Wire(Bool()),
                         data=Array([Wire(UInt(width)) for _ in range(lanes)]))


@dataclass
class _DataclassRec(CompositeRecord):
    a: Wire
    b: Wire
    c: Wire


# ---------------------------------------------------------------------------
# Generic clone
# ---------------------------------------------------------------------------

def test_record_clone_is_fresh_structure_and_shared_description():
    reset_shared_cache()
    src = _Packet(lanes=3, width=12)
    src.flag <<= 1
    clone = src.get_wire_clone()

    assert isinstance(clone, _Packet)
    assert clone.lanes == 3                      # description carried over
    assert clone.width == src.width == 1 + 3 * 12
    assert clone.flag is not src.flag            # structure rebuilt
    assert clone.data is not src.data
    assert all(a is not b for a, b in zip(clone.to_list(), src.to_list()))
    assert clone.flag.kind == "wire" and clone.flag._driver is None  # a shape, not the wiring


def test_clone_recurses_through_nested_composites():
    reset_shared_cache()
    src = CompositeRecord(
        vec=Array([Array([Wire(UInt(4)), Wire(UInt(4))]), Array([Wire(UInt(4)), Wire(UInt(4))])]),
        fp=FixedPoint(Q8_8),
        flag=Wire(Bool()),
    )
    clone = src.get_wire_clone()

    assert isinstance(clone.vec[0], Array) and isinstance(clone.fp, FixedPoint)
    assert clone.vec[0][0] is not src.vec[0][0]
    assert clone.fp.ftype is src.fp.ftype  # the format is a description: shared, not copied
    assert [leaf.typ.width for leaf in clone.to_list()] == [4, 4, 4, 4, 16, 1]


def test_array_clone_needs_no_override():
    reset_shared_cache()
    src = Array([Wire(UInt(8)), FixedPoint(Q8_8), Array([Wire(SInt(4))])])
    clone = src.get_wire_clone()

    assert "wire_like" not in vars(Array) and "get_wire_clone" not in vars(Array)
    assert isinstance(clone, Array) and len(clone) == 3
    assert clone[0] is not src[0]
    assert isinstance(clone[1], FixedPoint) and isinstance(clone[2], Array)
    assert [leaf.typ.width for leaf in clone.to_list()] == [8, 16, 4]
    assert clone.to_list()[2].typ.signed


def test_dataclass_record_clone_keeps_field_order():
    reset_shared_cache()
    src = _DataclassRec(a=Wire(UInt(1)), b=Wire(UInt(2)), c=Wire(UInt(3)))
    clone = src.get_wire_clone()

    assert [leaf.typ.width for leaf in clone.to_list()] == [1, 2, 3]  # bit order is field order
    assert [l.name for l in clone.to_list()] == [l.name for l in src.to_list()]  # names carried


def test_clone_of_a_view_owns_its_bits():
    reset_shared_cache()
    view = FixedPoint(Q8_8, bits=Const(0x0180, Q8_8.to_hdl_type()))
    clone = view.get_wire_clone()

    assert isinstance(clone.bits, Wire) and clone.bits.kind == "wire"
    assert clone.bits._driver is None
    assert clone.ftype is view.ftype


def test_port_leaves_clone_as_wires():
    reset_shared_cache()
    src = CompositeRecord(i=Input(UInt(8)), o=Output(UInt(8)))
    clone = src.get_wire_clone()
    assert [leaf.kind for leaf in clone.to_list()] == ["wire", "wire"]


# ---------------------------------------------------------------------------
# wire_like shim + the overrides that remain
# ---------------------------------------------------------------------------

def test_wire_like_shim_delegates_to_the_clone():
    reset_shared_cache()
    src = _Packet()
    clone = _Packet.wire_like(src)
    assert isinstance(clone, _Packet) and clone.width == src.width
    assert all(a is not b for a, b in zip(clone.to_list(), src.to_list()))


@pytest.mark.parametrize("cls, typ, attr", [(FixedPoint, Q8_8, "ftype"),
                                            (FloatingPoint, FT16, "ftype")])
def test_type_driven_wire_like_still_builds_without_a_template(cls, typ, attr):
    reset_shared_cache()
    built = cls.wire_like(typ)  # a format fully describes the shape — no value needed
    assert isinstance(built, cls) and getattr(built, attr) is typ
    assert built.width == typ.width_total

    from_template = cls.wire_like(built)
    assert isinstance(from_template, cls) and from_template.bits is not built.bits


def test_floating_point_clone_carries_arithmetic_configs():
    reset_shared_cache()
    add_cfg, mul_cfg = AdderConfig(use_operator=True), MultiplierConfig()
    src = FloatingPoint(FT16, name="fp", adder_cfg=add_cfg, mult_cfg=mul_cfg)

    for clone in (src.get_wire_clone(), FloatingPoint.wire_like(src)):
        assert clone.adder_cfg is add_cfg and clone.mult_cfg is mul_cfg


def test_named_wire_like_keeps_the_explicit_name():
    reset_shared_cache()
    src = FixedPoint(Q8_8, name="src")
    assert FixedPoint.wire_like(src, name="chosen").bits.name == "chosen"


def test_wire_like_rejects_a_foreign_argument():
    reset_shared_cache()
    with pytest.raises(TypeError, match="expects FixedPoint or FixedPointType"):
        FixedPoint.wire_like(Wire(UInt(16)))


def test_template_record_clone_comes_from_the_class():
    reset_shared_cache()

    class _Bus(TemplateRecord):
        data = Wire(UInt(8))
        valid = Output(Bool())

    clone = _Bus.wire_like(_Bus())
    assert isinstance(clone, _Bus) and clone.width == 9
    assert clone.valid.kind == "output"  # class-fixed shape keeps declared directions


def test_composite_register_refuses_to_be_cloned():
    reset_shared_cache()
    creg = CompositeRegister(FixedPoint, Q8_8, name="acc")

    with pytest.raises(TypeError, match="is storage, not a shape"):
        creg.get_wire_clone()
    with pytest.raises(TypeError, match="is storage, not a shape"):
        CompositeRegister.wire_like(creg)

    clone = creg.value.get_wire_clone()  # the documented way round
    assert isinstance(clone, FixedPoint) and clone.bits is not creg.bits


def test_subclass_override_wins():
    reset_shared_cache()

    class _Odd(HDLComposite):
        def __init__(self, width: int = 5):
            self.sig = Wire(UInt(width))
            self.cloned_by_hand = False

        def to_list_first_level(self):
            return [self.sig]

        def get_wire_clone(self):
            made = _Odd(self.sig.typ.width)
            made.cloned_by_hand = True
            return made

    src = _Odd(7)
    for clone in (src.get_wire_clone(), _Odd.wire_like(src)):
        assert clone.cloned_by_hand and clone.width == 7  # the shim routes through the override


def test_clone_is_assignable_and_packs_like_the_template():
    reset_shared_cache()
    src = _Packet(lanes=2, width=8)
    clone = src.get_wire_clone()
    clone <<= src.to_bits()  # packed assignment across the fresh leaves

    assert isinstance(clone, BitSerializable)
    assert all(leaf._driver is not None for leaf in clone.to_list())
