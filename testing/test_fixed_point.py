from __future__ import annotations
from spirehdl.composite.register import CompositeRegister
from spirehdl.composite.fixed_point import ARITHQuant, FixedPoint, FixedPointType
from spirehdl.spirehdl import HDLType, UInt, as_expr, fit_width
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator


from spirehdl.spirehdl import (
    Expr,
    Const,
    reset_shared_cache,
)
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator


def test_sim_composite_register():
    # ---------------------------------------------
    # Build module
    # ---------------------------------------------
    m = Module("AggRegDemo", with_clock=True, with_reset=False)

    # Integer input (8-bit unsigned)
    x = m.input(UInt(8), "x")

    # Fixed-point type for accumulator: Q8.8 unsigned
    acc_ftype = FixedPointType(width_total=16, width_frac=8, signed=False)

    # Composite register that stores a FixedPoint(acc_ftype)
    acc = CompositeRegister(
        FixedPoint,
        acc_ftype,  # passed into FixedPoint.wire_like(...)
        name="acc_reg",
        init=0,  # interpreted as fixed-point zero
    )

    # Ensure the underlying reg is visible to the module/simulator
    # (until CompositeRegister integrates with Module more tightly)
    m._signals.append(acc.bits)

    # Structured view of accumulator register
    acc_val: FixedPoint = acc.value  # FixedPoint(acc_ftype) view on acc.bits

    # Represent x as Q8.8 by shifting left by frac bits
    x_q = FixedPoint(
        acc_ftype,
        name="x_q",
        bits=as_expr(x) << acc_ftype.width_frac,
    )

    # Next-state: acc_next = acc_val + x_q, quantized back to acc_ftype
    next_acc: FixedPoint = acc_val.add(
        x_q,
        out_type=acc_ftype,
    )

    # Packed assignment into composite register
    acc <<= next_acc

    # Output: raw accumulator bits
    y = m.output(UInt(acc_ftype.width_total), "y")
    y <<= acc_val.bits  # or acc.bits

    # ---------------------------------------------
    # Simulate
    # ---------------------------------------------
    sim = Simulator(m)

    # Initial state
    sim.eval()
    assert sim.get("y") == 0
    assert sim.get("acc_reg") == 0
    assert sim.peek_next("acc_reg") == 0  # x=0 → next = 0

    # Now apply x = 1 for 3 cycles; each cycle adds 1<<8 = 256 to accumulator.
    expected = 0

    for cycle in range(1, 4):
        sim.set("x", 1)

        # Before step: current value in both reg and FixedPoint view
        cur_reg = sim.get("acc_reg")
        cur_fp_bits = sim.peek(acc_val.bits)
        assert cur_reg == expected
        assert cur_fp_bits == expected

        # Predicted next-state
        nxt = sim.peek_next("acc_reg")
        assert nxt == expected + (1 << acc_ftype.width_frac)  # +256 each time

        # One clock step
        sim.step(1)
        expected += 1 << acc_ftype.width_frac

        # After step: acc_reg, y, and FixedPoint bits must all match
        assert sim.get("acc_reg") == expected
        assert sim.get("y") == expected
        assert sim.peek(acc_val.bits) == expected
        
        print(f"After cycle {cycle}: acc_reg = {sim.get('acc_reg')}")  # Debug print


# Small helper: evaluate an Expr to an integer using the simulator
def _eval_expr(e: Expr) -> int:
    m = Module("FPTest", with_clock=False, with_reset=False)
    sim = Simulator(m)
    return sim.peek(e)


# -------------------------------------------------------------
# Basic add: full precision, same format, unsigned
# -------------------------------------------------------------
def test_fixedpoint_add_full_precision_unsigned():
    reset_shared_cache()

    ft = FixedPointType(width_total=8, width_frac=3, signed=False)

    # Underlying integers (raw bits)
    a_raw = 5   # represents 5 / 2^3 = 0.625
    b_raw = 7   # represents 7 / 2^3 = 0.875

    a = FixedPoint(ft, bits=Const(a_raw, ft.to_hdl_type()))
    b = FixedPoint(ft, bits=Const(b_raw, ft.to_hdl_type()))

    c = a + b   # full precision (out_type=None, WrpTrc)

    # Type checks
    assert c.ftype.signed == ft.signed
    assert c.ftype.width_frac == ft.width_frac
    # add_result_type: result width = max(width) + 1
    assert c.ftype.width_total == ft.width_total + 1

    # Numeric check: 5 + 7 = 12
    val = _eval_expr(c.bits)
    assert val == a_raw + b_raw


# -------------------------------------------------------------
# Add with quantization back to original type (truncate)
# -------------------------------------------------------------
def test_fixedpoint_add_quantized_back_to_same_type():
    reset_shared_cache()

    ft = FixedPointType(width_total=8, width_frac=3, signed=False)

    a_raw = 10   # 10 / 8 = 1.25
    b_raw = 6    # 6 / 8  = 0.75

    a = FixedPoint(ft, bits=Const(a_raw, ft.to_hdl_type()))
    b = FixedPoint(ft, bits=Const(b_raw, ft.to_hdl_type()))

    # Full-precision raw sum would be 16; we quantize back to ft with WrpTrc
    c = a.add(b, out_type=ft, q=ARITHQuant.WrpTrc)

    assert c.ftype == ft
    val = _eval_expr(c.bits)
    # 10 + 6 = 16 fits into 8 bits without overflow, so still 16
    assert val == 16


# -------------------------------------------------------------
# Quantization: truncate vs round when reducing fractional bits
# -------------------------------------------------------------
def test_fixedpoint_quantize_truncate_vs_round():
    reset_shared_cache()

    # Full format: unsigned, 8 total bits, 4 fractional bits (Q4.4)
    ft_full = FixedPointType(width_total=8, width_frac=4, signed=False)
    # Target format: same total width but only 2 fractional bits (Q6.2)
    ft_out = FixedPointType(width_total=8, width_frac=2, signed=False)

    # Value: 1.125 → raw = 1.125 * 2^4 = 18
    a_raw = 18
    a = FixedPoint(ft_full, bits=Const(a_raw, ft_full.to_hdl_type()))
    zero = FixedPoint(ft_full, bits=Const(0, ft_full.to_hdl_type()))

    # "Resize" via add with zero, truncation
    a_trc = a.add(zero, out_type=ft_out, q=ARITHQuant.WrpTrc)
    a_rnd = a.add(zero, out_type=ft_out, q=ARITHQuant.WrpRnd)

    v_trc = _eval_expr(a_trc.bits)
    v_rnd = _eval_expr(a_rnd.bits)

    # Algorithm:
    #   frac_diff = 4 - 2 = 2
    #   truncate: v_trc = (18 >> 2) = 4 → 4 / 4 = 1.0
    #   round:    v_rnd = ((18 + 2) >> 2) = 5 → 5 / 4 = 1.25
    assert v_trc == 4
    assert v_rnd == 5


# -------------------------------------------------------------
# Multiply: full precision, then quantized back to smaller type
# -------------------------------------------------------------
def test_fixedpoint_mul_full_precision_and_quantized():
    reset_shared_cache()

    # Q4.2 unsigned (6 bits total, 2 fractional)
    ft = FixedPointType(width_total=6, width_frac=2, signed=False)

    # Let a = 1.5 → 1.5 * 2^2 = 6
    #     b = 1.0 → 1.0 * 2^2 = 4
    a_raw = 6
    b_raw = 4

    a = FixedPoint(ft, bits=Const(a_raw, ft.to_hdl_type()))
    b = FixedPoint(ft, bits=Const(b_raw, ft.to_hdl_type()))

    # Full-precision multiply
    c_full = a * b

    # Type checks
    # raw width = 6 + 6 = 12, frac = 2 + 2 = 4
    assert c_full.ftype.width_total == 12
    assert c_full.ftype.width_frac == 4
    assert c_full.ftype.signed == ft.signed

    v_full = _eval_expr(c_full.bits)
    # 6 * 4 = 24
    assert v_full == 24

    # This corresponds to 24 / 2^4 = 1.5 (1.5 * 1.0 = 1.5), as expected.

    # Now quantize back to the original format ft (Q4.2) with truncation:
    c_q = a.mul(b, out_type=ft, q=ARITHQuant.WrpTrc)

    assert c_q.ftype == ft
    v_q = _eval_expr(c_q.bits)

    # Quantization:
    # - full frac = 4, out frac = 2 → frac_diff = 2
    # - WrpTrc: v_q = 24 >> 2 = 6
    assert v_q == 6  # 6 / 2^2 = 1.5 again


# -------------------------------------------------------------
# Signed add: full precision, same frac
# -------------------------------------------------------------
def test_fixedpoint_add_full_precision_signed():
    reset_shared_cache()

    ft = FixedPointType(width_total=8, width_frac=3, signed=True)

    # Represent -1.25 and +0.5 in Q5.3
    # -1.25 → two's complement of 1.25 * 2^3 = 10 -> 8-bit TC: 256-10 = 246
    a_raw = -10  # Const with signed HDLType will encode this correctly
    b_raw = 4    # 4 / 8 = 0.5

    a = FixedPoint(ft, bits=Const(a_raw, ft.to_hdl_type()))
    b = FixedPoint(ft, bits=Const(b_raw, ft.to_hdl_type()))

    c = a + b  # full precision

    assert c.ftype.width_frac == ft.width_frac
    assert c.ftype.signed is True

    v = _eval_expr(c.bits)
    # In integer domain: (-10) + 4 = -6
    assert v == _eval_expr(fit_width(Const(-10, c.typ) + Const(4, c.typ), HDLType(c.typ.width, signed=True)))  # correct TC encoding


# -------------------------------------------------------------
# Error cases: sign mismatch between operands and out_type
# -------------------------------------------------------------
def test_fixedpoint_sign_mismatch_operands():
    reset_shared_cache()

    ft_s = FixedPointType(width_total=8, width_frac=3, signed=True)
    ft_u = FixedPointType(width_total=8, width_frac=3, signed=False)

    a = FixedPoint(ft_s, bits=Const(1, ft_s.to_hdl_type()))
    b = FixedPoint(ft_u, bits=Const(1, ft_u.to_hdl_type()))

    # Operand sign mismatch should raise
    try:
        _ = a + b
        assert False, "Expected ValueError due to operand sign mismatch"
    except ValueError:
        pass


def test_fixedpoint_sign_mismatch_out_type():
    reset_shared_cache()

    ft_s = FixedPointType(width_total=8, width_frac=3, signed=True)
    ft_out_wrong = FixedPointType(width_total=8, width_frac=3, signed=False)

    a = FixedPoint(ft_s, bits=Const(1, ft_s.to_hdl_type()))
    b = FixedPoint(ft_s, bits=Const(2, ft_s.to_hdl_type()))

    # Explicit out_type with different signedness should raise
    try:
        _ = a.add(b, out_type=ft_out_wrong, q=ARITHQuant.WrpTrc)
        assert False, "Expected ValueError due to out_type sign mismatch"
    except ValueError:
        pass


# -------------------------------------------------------------
# "Resize" behaviour via add with zero, different total width
# -------------------------------------------------------------
def test_fixedpoint_resize_truncate_to_smaller_total_width():
    reset_shared_cache()

    # Full format: Q4.4 signed, 9 bits total
    ft_full = FixedPointType(width_total=9, width_frac=4, signed=True)
    # Out format: Q4.4 signed, but only 7 bits total (2 MSBs will be dropped)
    ft_out = FixedPointType(width_total=7, width_frac=4, signed=True)

    # Choose a small value that does not overflow even in the small format
    raw = 5  # 5 / 16 = 0.3125

    a = FixedPoint(ft_full, bits=Const(raw, ft_full.to_hdl_type()))
    zero = FixedPoint(ft_full, bits=Const(0, ft_full.to_hdl_type()))

    a_resized = a.add(zero, out_type=ft_out, q=ARITHQuant.WrpTrc)

    assert a_resized.ftype == ft_out
    v = _eval_expr(a_resized.bits)
    # Since 5 fits in 7 bits, value should be unchanged
    assert v == raw


if __name__ == "__main__":
    test_sim_composite_register()
    test_fixedpoint_add_full_precision_unsigned()
    test_fixedpoint_add_quantized_back_to_same_type()
    test_fixedpoint_quantize_truncate_vs_round()
    test_fixedpoint_mul_full_precision_and_quantized()
    test_fixedpoint_add_full_precision_signed()
    test_fixedpoint_sign_mismatch_operands()
    test_fixedpoint_sign_mismatch_out_type()
    test_fixedpoint_resize_truncate_to_smaller_total_width()
