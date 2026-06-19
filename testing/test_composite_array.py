from spire.composite.base import HDLComposite
from spire.expr import (
    UInt,
    Wire,
    Expr,
    Signal,
    as_expr,
    reset_shared_cache,
)
from spire.composite.array import Array
from spire.expr import UInt
from spire.component import Netlist
from spire.simulator import Simulator
from testing.test_composite_register_and_array import DummyAgg


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


def test_array_scalar_wires_width_and_bits():
    reset_shared_cache()

    a0 = Wire(UInt(8))
    a1 = Wire(UInt(8))
    a2 = Wire(UInt(8))

    arr = Array([a0, a1, a2])  # only Expr elements (Signals)

    assert len(arr) == 3
    assert isinstance(arr._elems[0], Signal)
    assert arr.width == 24

    bits = arr.to_bits()
    assert isinstance(bits, Expr)
    assert bits.typ.width == 24


def test_array_nd_indexing_and_slicing():
    reset_shared_cache()

    r0c0 = Wire(UInt(4))
    r0c1 = Wire(UInt(4))
    r1c0 = Wire(UInt(4))
    r1c1 = Wire(UInt(4))

    row0 = Array([r0c0, r0c1])
    row1 = Array([r1c0, r1c1])

    mat = Array([row0, row1])  # 2x2

    # 1D access
    assert isinstance(mat[0], Array)
    assert mat[0] is row0

    # 2D access
    assert mat[0, 0] is r0c0
    assert mat[1, 1] is r1c1

    # Slicing on first axis
    sub = mat[0:2]
    assert isinstance(sub, Array)
    assert len(sub) == 2
    assert sub[0] is row0
    assert sub[1] is row1


def test_array_packed_assign_from_array():
    """
    dst <<= src  (packed HDLComposite assignment)
    """
    reset_shared_cache()

    s0 = Wire(UInt(4))
    s1 = Wire(UInt(4))
    src = Array([s0, s1])

    d0 = Wire(UInt(4))
    d1 = Wire(UInt(4))
    dst = Array([d0, d1])

    # Initially no drivers
    assert d0._driver is None
    assert d1._driver is None

    dst <<= src  # uses Array.to_bits + Array._assign_from_bits

    # Each dst element should now have a driver (some slice of src.to_bits())
    for elem in dst:
        assert isinstance(elem, Signal)
        assert elem._driver is not None
        assert elem._driver.typ.width == elem.typ.width


def test_array_packed_assign_from_constant():
    """
    dst <<= 0  (packed HDLComposite assignment from constant)
    """
    reset_shared_cache()

    d0 = Wire(UInt(4))
    d1 = Wire(UInt(4))
    d2 = Wire(UInt(4))

    arr = Array([d0, d1, d2])  # total width 12

    arr <<= 0  # constant -> as_expr -> Resize(..., width=12)

    for elem in arr:
        assert isinstance(elem, Signal)
        assert elem._driver is not None
        # Each elem is 4 bits wide
        assert elem._driver.typ.width == 4


def test_array_wire_like_clone_shape():
    """
    wire_like(template) creates a new Array with same shape, new wires / composites.
    """
    reset_shared_cache()

    d0 = Wire(UInt(8))
    d1 = Wire(UInt(8))
    d2_const = as_expr(5)  # Expr, not a Signal

    template = Array([d0, d1, d2_const])
    clone = Array.wire_like(template)

    assert len(clone) == len(template)

    # First two should be wires (cloned)
    assert isinstance(clone._elems[0], Signal)
    assert isinstance(clone._elems[1], Signal)

    # Third element: was Const in template, now a Wire with same type
    assert isinstance(clone._elems[2], Signal)
    assert clone._elems[2].typ.width == d2_const.typ.width


def test_array_elementwise_assign():
    """
    dst <<= src  (element-wise Array assignment)
    """
    reset_shared_cache()

    s0 = Wire(UInt(4))
    s1 = Wire(UInt(4))
    src = Array([s0, s1])

    d0 = Wire(UInt(4))
    d1 = Wire(UInt(4))
    dst = Array([d0, d1])

    # Drive source wires
    s0 <<= 1
    s1 <<= 2

    # Element-wise assignment
    dst @= src

    assert d0._driver is s0
    assert d1._driver is s1


def test_array_with_dummy_composite_and_wires():
    """
    Mixed Array: [Wire, DummyAgg, Wire] + packed assign + wire_like.
    """
    reset_shared_cache()

    w0 = Wire(UInt(8))
    agg = DummyAgg(width=5, name="agg")
    w1 = Wire(UInt(4))

    arr = Array([w0, agg, w1])

    # Width = 8 + 5 + 4 = 17
    assert arr.width == 17
    bits = arr.to_bits()
    assert isinstance(bits, Expr)
    assert bits.typ.width == 17

    # Clone shape with wire_like
    clone = Array.wire_like(arr)
    assert len(clone) == len(arr)
    assert isinstance(clone._elems[0], Signal)
    assert isinstance(clone._elems[1], DummyAgg)
    assert isinstance(clone._elems[2], Signal)

    # Packed assign from arr to clone
    clone <<= arr

    # w0 / w1 clones should now have drivers
    c_w0 = clone._elems[0]
    c_agg = clone._elems[1]
    c_w1 = clone._elems[2]

    assert isinstance(c_w0, Signal)
    assert c_w0._driver is not None

    # DummyAgg inside clone should have its internal wire driven
    assert isinstance(c_agg, DummyAgg)
    assert isinstance(c_agg.sig, Signal)
    assert c_agg.sig._driver is not None

    assert isinstance(c_w1, Signal)
    assert c_w1._driver is not None


def sim_test_composite_array():
    # ---------------------------------------------
    # Build module
    # ---------------------------------------------
    m = Netlist("AggArrayDemo", with_clock=False, with_reset=False)

    a0 = m.input(UInt(4), "a0")
    a1 = m.input(UInt(4), "a1")
    a2 = m.input(UInt(4), "a2")

    arr = Array([a0, a1, a2])  # Array of 3×4-bit = 12 bits total

    # Packed output (named signal)
    packed_expr = arr.to_bits()  # Expr, 12 bits
    packed = m.output(UInt(arr.width), "packed")
    packed <<= packed_expr

    # ---------------------------------------------
    # Simulate
    # ---------------------------------------------
    sim = Simulator(m)

    # Example values:
    #   a0 = 0x1, a1 = 0x2, a2 = 0x3
    # Packing order (LSB → MSB) is: a0, then a1, then a2
    #
    # So packed = a0 + (a1 << 4) + (a2 << 8)
    #           = 0x1 + (0x2 << 4) + (0x3 << 8)
    #           = 0x1 + 0x20 + 0x300 = 0x321 = 801
    sim.set("a0", 0x1).set("a1", 0x2).set("a2", 0x3).eval()

    expected = 0x321

    # Check via output
    packed_from_output = sim.get("packed")
    assert packed_from_output == expected

    # Check via direct Expr peek
    packed_from_expr = sim.peek(packed_expr)
    assert packed_from_expr == expected

    # Also verify individual inputs are what we set
    assert sim.get("a0") == 0x1
    assert sim.get("a1") == 0x2
    assert sim.get("a2") == 0x3

    # Watch the packed Expr and confirm watch value
    sim.watch(packed_expr, alias="ARR").eval()
    assert sim.get_watch("ARR") == expected


if __name__ == "__main__":
    test_array_scalar_wires_width_and_bits()
    test_array_nd_indexing_and_slicing()
    test_array_packed_assign_from_array()
    test_array_packed_assign_from_constant()
    test_array_wire_like_clone_shape()
    test_array_elementwise_assign()
    test_array_with_dummy_composite_and_wires()
    sim_test_composite_array()
