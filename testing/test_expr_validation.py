"""Construction-time validation of constants and shift amounts, and boundary-literal emission."""
import pytest

from differential.harness import diff_sim_vs_verilog, exhaustive, fresh_netlist
from spire.expr import Bool, Const, HDLType, SInt, UInt


@pytest.mark.parametrize("value,typ", [
    (16, UInt(4)), (-1, UInt(4)), (8, SInt(4)), (-9, SInt(4)), (2, Bool()), (-1, Bool()),
])
def test_const_out_of_range_raises(value, typ):
    with pytest.raises(ValueError, match="not representable"):
        Const(value, typ)


def test_const_range_bounds_accepted():
    Const(15, UInt(4))
    Const(0, UInt(4))
    Const(7, SInt(4))
    Const(-8, SInt(4))
    Const(1, Bool())
    Const(0, HDLType(0, signed=False))  # zero-width Concat placeholder


def test_negative_shift_amount_raises():
    m = fresh_netlist("shift_neg")
    a = m.input(UInt(4), "a")
    with pytest.raises(ValueError, match="Shift amount"):
        a << -1
    with pytest.raises(ValueError, match="Shift amount"):
        a >> -1


def test_boundary_literal_emission():
    assert Const(-8, SInt(4)).to_verilog() == "$signed(4'd8)"
    assert Const(-1, SInt(1)).to_verilog() == "$signed(1'd1)"
    assert Const(-7, SInt(4)).to_verilog() == "-4'sd7"


@pytest.mark.parametrize("cval,cw", [(-8, 4), (-1, 1), (-7, 4)])
def test_boundary_const_conformance(cval, cw):
    # A most-negative (or ordinary negative) literal as an arithmetic operand must survive context extension.
    m = fresh_netlist(f"bconst_{cw}_{abs(cval)}")
    c = m.input(SInt(8), "c")
    y = m.output(SInt(9), "y")
    y <<= Const(cval, SInt(cw)) + c
    bad, text = diff_sim_vs_verilog(m, ["c"], exhaustive(8))
    assert not bad, f"{len(bad)} mismatches; first {bad[0]}\n{text}"
