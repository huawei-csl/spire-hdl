"""Register init values are constants: dynamic expressions are rejected, literals emit and reset correctly."""
import pytest

from spire.expr import Const, SInt, UInt, cat
from spire.ir import Netlist
from spire.simulator import Simulator


def test_dynamic_init_rejected():
    m = Netlist("init_dyn")
    a = m.input(UInt(4), "a")
    with pytest.raises(ValueError, match="init must be a constant"):
        m.reg(UInt(4), "r", init=a + 1)


def test_compound_const_init_rejected():
    # Even a constant-valued *expression* is rejected: init must be a direct literal.
    m = Netlist("init_cat")
    with pytest.raises(ValueError, match="init must be a constant"):
        m.reg(UInt(4), "r", init=cat(Const(3, UInt(2)), Const(1, UInt(2))))


def test_out_of_range_init_rejected():
    m = Netlist("init_range")
    with pytest.raises(ValueError, match="not representable"):
        m.reg(UInt(4), "r", init=255)


def test_const_init_emits_and_resets():
    m = Netlist("init_ok")
    a = m.input(UInt(4), "a")
    r = m.reg(UInt(4), "r", init=7)
    r <<= a
    y = m.output(UInt(4), "y")
    y <<= r

    text = m.to_verilog()
    assert "r <= 4'd7;" in text  # the reset arm is a plain literal

    sim = Simulator(m)
    sim.set("a", 6)
    sim.reset(True)
    sim.deassert_reset()
    sim.eval()
    assert sim.get("y") == 7
    sim.step()
    assert sim.get("y") == 6


def test_negative_boundary_init():
    m = Netlist("init_neg")
    r = m.reg(SInt(4), "r", init=-8)
    r <<= r
    y = m.output(SInt(4), "y")
    y <<= r

    text = m.to_verilog()
    assert "r <= $signed(4'd8);" in text  # most-negative literal form

    sim = Simulator(m)
    sim.reset(True)
    sim.deassert_reset()
    sim.eval()
    assert sim.get("y") == -8
