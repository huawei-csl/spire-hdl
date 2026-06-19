from __future__ import annotations

from typing import List
from spire.composite.register import CompositeRegister
from spire.composite.base import HDLComposite
from spire.expr import (
    UInt,
    Wire,
    Expr,
    Signal,
    as_expr,
    reset_shared_cache,
)
from spire.composite.fixed_point import FixedPoint, FixedPointType
from spire.component import Netlist
from spire.simulator import Simulator  # your HDLComposite fixed-point type


# -------------------------------------------------------------------
# Simple HDLComposite implementation for testing
# -------------------------------------------------------------------
class DummyAgg(HDLComposite):
    """
    Tiny composite for tests: wraps a single Expr (usually a Wire).
    Supports:
      - wire_like(width)
    so it works with CompositeRegister.
    """

    def __init__(self, width: int, bits: Expr | None = None, name: str = "dummy"):
        self._width = width
        self._typ = UInt(width)
        if bits is None:
            # Owning case: create a fresh wire
            self.sig = Wire(self._typ, name=name)
        else:
            # View case: reinterpret existing bits
            self.sig = as_expr(bits)

    # ---- HDLComposite API ----

    @classmethod
    def wire_like(cls, template: "DummyAgg") -> "DummyAgg":
        """
        Create a new DummyAgg with same width as template, new backing wire.
        """
        return cls(width=template.width, name=template.sig.name + "_w")

    def to_list_first_level(self) -> List[Expr]:
        return [self.sig]

    def __repr__(self) -> str:
        return f"DummyAgg(width={self.width}, sig={self.sig})"


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


def test_composite_register_with_dummyagg_basic():
    reset_shared_cache()

    # Register that stores a DummyAgg(width=5)
    reg = CompositeRegister(DummyAgg, 5, name="dummy_reg")

    # Under-the-hood: one reg Signal with width 5
    bits = reg.bits
    assert isinstance(bits, Signal)
    assert bits.kind == "reg"
    assert bits.typ.width == 5

    # HDLComposite.width property
    assert reg.width == 5

    # Structured view
    val = reg.value
    assert isinstance(val, DummyAgg)
    assert val.width == 5
    assert val.to_bits().typ.width == 5
    # For DummyAgg.from_bits, we expect it's a view on the reg's bits
    assert val.to_bits()._driver._driver.a is bits


def test_composite_register_dummyagg_assign_from_int():
    reset_shared_cache()

    reg = CompositeRegister(DummyAgg, 5, name="dummy_reg")

    # No next-state yet
    assert reg.bits._driver is None

    # Packed assignment via HDLComposite:
    reg <<= 3  # int -> Const -> resized to width 5

    assert reg.bits._driver is not None
    assert isinstance(reg.bits._driver, Expr)
    assert reg.bits._driver.typ.width == 5


def test_composite_register_dummyagg_assign_from_agg():
    reset_shared_cache()

    # src composite: DummyAgg with its own wire
    src = DummyAgg(width=5, name="src_dummy")
    src.sig <<= 7  # combinational driver, just to make it non-constant

    # reg that holds a DummyAgg(5)
    reg = CompositeRegister(DummyAgg, 5, name="dummy_reg")

    # Assign packed from composite
    reg <<= src

    assert reg.bits._driver is not None
    assert reg.bits._driver.typ.width == 5
    # Next-state should be driven by src.to_bits()
    assert reg.bits._driver._driver.a is src.to_bits()


def test_composite_register_dummyagg_init_from_int():
    reset_shared_cache()

    reg = CompositeRegister(DummyAgg, 5, name="dummy_reg_init", init=6)

    bits = reg.bits
    assert isinstance(bits, Signal)
    assert bits.kind == "reg"

    # _init should be set
    assert bits._init is not None
    assert bits._init.typ.width == 5
    # Usually bits._init will be a Const; if so, we can check the value
    if bits._init.__class__.__name__ == "Const":
        assert bits._init.value == 6


def test_composite_register_dummyagg_init_from_agg():
    reset_shared_cache()

    src = DummyAgg(width=5, name="src_init")
    src.sig <<= 9

    reg = CompositeRegister(DummyAgg, 5, name="dummy_reg_init", init=src)

    bits = reg.bits
    assert bits._init is not None
    assert bits._init.typ.width == 5
    # Init expression should match src bits
    assert bits._init is src.to_bits()


def test_composite_register_with_fixedpoint_basic():
    reset_shared_cache()
    
    ftype = FixedPointType(width_total=16, width_frac=8, signed=True)

    # Register that holds a FixedPoint(16, frac=8, signed)
    acc = CompositeRegister(
        FixedPoint,
        ftype,
        name="acc_reg",
    )

    # Raw bits
    bits = acc.bits
    assert isinstance(bits, Signal)
    assert bits.kind == "reg"
    assert bits.typ.width == 16
    assert acc.width == 16

    # Structured view
    val = acc.value
    assert isinstance(val, FixedPoint)
    assert val.ftype.width_total == 16
    assert val.ftype.width_frac == 8
    assert val.ftype.signed is True
    # The view's bits should be the reg itself (from_bits view semantics)
    assert val.bits._driver._driver.a is bits


def test_composite_register_fixedpoint_assign():
    reset_shared_cache()

    ftype = FixedPointType(width_total=16, width_frac=8, signed=True)

    # Register that holds a FixedPoint(16, frac=8, signed)
    acc = CompositeRegister(
        FixedPoint,
        ftype,
        name="acc_reg",
    )

    # Assign from integer
    acc <<= 42
    assert acc.bits._driver is not None
    assert acc.bits._driver.typ.width == 16

    # Assign from another FixedPoint
    src = FixedPoint(ftype, name="src_fp")
    acc <<= src
    assert acc.bits._driver is not None
    assert acc.bits._driver.typ.width == 16
    assert acc.bits._driver._driver.a is src.bits


def test_composite_register_fixedpoint_init():
    reset_shared_cache()

    ftype = FixedPointType(width_total=16, width_frac=8, signed=True)
    
    # Init from int
    acc1 = CompositeRegister(
        FixedPoint,
        ftype,
        name="acc_reg_init1",
        init=5,
    )
    assert acc1.bits._init is not None
    assert acc1.bits._init.typ.width == 16

    # Init from FixedPoint composite
    src = FixedPoint(ftype, name="src_fp")
    acc2 = CompositeRegister(
        FixedPoint,
        ftype,
        name="acc_reg_init2",
        init=src,
    )
    assert acc2.bits._init is not None
    assert acc2.bits._init.typ.width == 16
    assert acc2.bits._init is src.bits


def sim_test_composite_register():
    # ---------------------------------------------
    # Build module
    # ---------------------------------------------
    m = Netlist("AggRegDemo", with_clock=True, with_reset=False)

    x = m.input(UInt(8), "x")  # 8-bit unsigned input

    ftype = FixedPointType(width_total=16, width_frac=8, signed=True)

    # Composite register: 16-bit FixedPoint with 8 fractional bits (Q8.8)
    acc = CompositeRegister(
        FixedPoint,
        ftype,
        name="acc_reg",
        init=0,  # start at 0
    )

    # Make sure the underlying reg is visible to the module/simulator
    m._signals.append(acc.bits)

    # Structured view of the accumulator
    acc_val = acc.value  # FixedPoint view on acc.bits

    # Behavior: acc_next = acc + (x << 8)
    incr = as_expr(x) << 8  # Q8.8 version of x
    acc <<= acc_val.bits + incr  # packed assignment → acc.bits._driver

    # Output: raw accumulator bits
    y = m.output(UInt(16), "y")
    y <<= acc.bits

    # ---------------------------------------------
    # Simulate
    # ---------------------------------------------
    sim = Simulator(m)

    # Initial state
    sim.eval()
    assert sim.get("y") == 0
    assert sim.get("acc_reg") == 0
    # Next-state with x=0 should also be 0
    assert sim.peek_next("acc_reg") == 0

    # Now apply x = 1 for 3 cycles; each cycle adds 1<<8 = 256
    expected = 0
    for cycle in range(1, 4):
        sim.set("x", 1)

        # Before step: current value and predicted next
        cur = sim.get("acc_reg")
        nxt = sim.peek_next("acc_reg")
        assert cur == expected
        assert nxt == expected + (1 << 8)

        # One clock step
        sim.step(1)
        expected += 1 << 8

        # After step: acc_reg and y must match expected
        assert sim.get("acc_reg") == expected
        assert sim.get("y") == expected

        # Also the FixedPoint view bits must match
        assert sim.peek(acc_val.bits) == expected


if __name__ == "__main__":
    test_composite_register_with_dummyagg_basic()
    test_composite_register_dummyagg_assign_from_int()
    test_composite_register_dummyagg_assign_from_agg()
    test_composite_register_dummyagg_init_from_int()
    test_composite_register_dummyagg_init_from_agg()
    test_composite_register_with_fixedpoint_basic()
    test_composite_register_fixedpoint_assign()
    test_composite_register_fixedpoint_init()
    sim_test_composite_register()
