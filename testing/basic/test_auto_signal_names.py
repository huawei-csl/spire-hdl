from sprouthdl.sprouthdl import Register, UInt, Wire
from sprouthdl.sprouthdl_module import Module


def test_unnamed_wire_and_register_infer_python_variable_names():
    m = Module("AutoNames")
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(9), "y")

    sum_wire = Wire(UInt(9))
    sum_wire <<= a + b

    acc_reg = Register(UInt(9))
    acc_reg <<= sum_wire

    y <<= acc_reg

    verilog = m.to_verilog()

    assert "wire [8:0] sum_wire;" in verilog
    assert "reg [8:0] acc_reg;" in verilog
    assert "assign sum_wire =" in verilog
    assert "wire_" not in verilog
    assert "reg_" not in verilog


def test_bound_expression_name_is_used_for_generated_shared_signal():
    m = Module("ExprNames")
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(9), "y")

    sum_expr = a + b
    y <<= sum_expr

    verilog = m.to_verilog()

    assert "wire [8:0] sum_expr;" in verilog
    assert "assign sum_expr = (a + b);" in verilog
