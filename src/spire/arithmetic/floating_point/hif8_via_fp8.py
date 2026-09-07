"""HiFloat8 multiplier built from a conventional FP8 core.

This wrapper reuses the parameterisable floating-point multiplier from
``float_mult.py``.  Each HiFloat8 operand is translated to an FP8 (E5M2)
value using compact arithmetic, the operands are multiplied by the FP8 core,
and the product is decoded back to HiFloat8 with the same bucketed rounding
logic used by the dedicated HiFloat8 multiplier.  The translation logic avoids
large lookup tables – everything is implemented with comparisons, shifts and
small adders so the area footprint stays close to that of the FP8 core itself.
"""

from __future__ import annotations


from spire.expr import (
    Bool,
    Const,
    Expr,
    Signal,
    SInt,
    UInt,
    cat,
    fit_width,
    mux,
)
from spire.component import Component, Netlist
from spire.io_record import IORecord, Input, Output

from spire.arithmetic.floating_point.float_mult import build_fp_mul
from spire.arithmetic.floating_point.hif8 import _abs_sint, _const_sint, _const_uint, _decode_operand_expr, _round_bucket


_FP_EW = 5
_FP_FW = 2
_FP_WIDTH = 1 + _FP_EW + _FP_FW
_FP_BIAS = (1 << (_FP_EW - 1)) - 1


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _uint(width: int, value: int) -> Expr:
    return Const(value, UInt(width))


def _sint(width: int, value: int) -> Expr:
    return Const(value, SInt(width))


def _bool(value: int) -> Expr:
    return Const(value, Bool())


def _or_bits(expr: Expr, start: int, stop: int) -> Expr:
    """Return OR over bits [start, stop)."""

    acc = _bool(0)
    width = expr.typ.width
    for idx in range(start, stop):
        if idx >= width:
            break
        acc = acc | expr[idx]
    return acc


def _round_frac_to_fp(sig: Expr) -> tuple[Expr, Expr]:
    """Round HiF8 mantissa to the FP8 (E5M2) mantissa."""

    mant_low = sig[0:3]  # fractional bucket (0..7)
    q = mant_low >> 1
    rem = mant_low[0]
    odd = q[0]

    q_ext = fit_width(q, UInt(4))
    inc = fit_width(rem & odd, UInt(4))
    rounded = q_ext + inc
    overflow = rounded >= _uint(4, 4)
    frac_bits = mux(overflow, _uint(2, 0), rounded[0:2])
    return frac_bits, overflow


def _round_div_pow2(sig: Expr, shift: int) -> tuple[Expr, Expr]:
    """Round sig / 2**shift to nearest-even, returning (frac, overflow)."""

    sig_u = fit_width(sig, UInt(sig.typ.width))
    quotient = sig_u >> shift
    guard = sig_u[shift - 1] if shift - 1 < sig_u.typ.width else _bool(0)
    sticky = _or_bits(sig_u, 0, max(0, shift - 1))
    lsb = quotient[0]
    inc = guard & (sticky | lsb)
    rounded = fit_width(quotient, UInt(4)) + fit_width(inc, UInt(4))
    overflow = rounded >= _uint(4, 4)
    frac_bits = rounded[0:2]
    return frac_bits, overflow


def _hif8_to_fp8_expr(bits: Expr) -> Expr:
    decoded = _decode_operand_expr(bits)

    sign = fit_width(decoded["sign"], UInt(1))
    zero = decoded["is_zero"]
    is_nan = decoded["is_nan"]
    is_inf = decoded["is_inf"]

    pos_inf = _uint(_FP_WIDTH, 0x7C)
    neg_inf = _uint(_FP_WIDTH, 0xFC)
    nan = _uint(_FP_WIDTH, 0x7E)
    zero_bits = _uint(_FP_WIDTH, 0x00)

    sig = fit_width(decoded["sig"], UInt(4))
    exp = decoded["exp"]

    exp_bias = fit_width(exp, SInt(9)) + _sint(9, _FP_BIAS)
    exp_bias_le0 = exp_bias <= _sint(9, 0)

    frac_norm, overflow_norm = _round_frac_to_fp(sig)
    exp_bias_norm = exp_bias + fit_width(overflow_norm, SInt(9))
    exp_bits_norm = fit_width(exp_bias_norm, UInt(9))[0:5]
    frac_bits_norm = frac_norm
    exp_overflow = exp_bias_norm >= _sint(9, 31)

    # Subnormal handling (HiF8 denorm bucket -> FP8 subnormal)
    exp_plus_13 = fit_width(exp, SInt(9)) + _sint(9, 13)
    shift_amt = fit_width((_sint(9, 0) - exp_plus_13), UInt(5))
    shift_clamped = mux(
        shift_amt < _uint(5, 2),
        _uint(5, 2),
        mux(shift_amt > _uint(5, 10), _uint(5, 10), shift_amt),
    )

    frac_sub = _uint(2, 0)
    overflow_sub = _bool(0)
    for shift in range(2, 11):
        frac_candidate, overflow_candidate = _round_div_pow2(sig, shift)
        cond = shift_clamped == _uint(5, shift)
        frac_sub = mux(cond, frac_candidate, frac_sub)
        overflow_sub = mux(cond, overflow_candidate, overflow_sub)

    exp_bits_sub = mux(overflow_sub, _uint(5, 1), _uint(5, 0))
    frac_bits_sub = mux(overflow_sub, _uint(2, 0), frac_sub)

    finite = mux(
        exp_bias_le0,
        cat(frac_bits_sub, exp_bits_sub, sign),
        cat(frac_bits_norm, exp_bits_norm, sign),
    )

    finite = mux(zero, zero_bits, finite)
    finite = mux(exp_overflow, mux(sign, neg_inf, pos_inf), finite)

    with_nan = mux(is_inf, mux(sign, neg_inf, pos_inf), finite)
    result = mux(is_nan, nan, with_nan)
    return result


def _fp8_to_hif8_expr(bits: Expr) -> Expr:
    sign = bits[7]
    exp_field = bits[2:7]
    frac = bits[0:2]

    exp_all_ones = exp_field == _uint(5, 0b11111)
    exp_zero = exp_field == _uint(5, 0)
    frac_zero = frac == _uint(2, 0)

    is_nan = exp_all_ones & (~frac_zero)
    is_inf = exp_all_ones & frac_zero
    is_zero = exp_zero & frac_zero

    frac_shift = fit_width(frac, UInt(2)) << 1
    sig_norm = cat(frac_shift, _uint(1, 1))
    exp_norm = fit_width(exp_field, SInt(8)) - _const_sint(15)

    mant_base = frac << 1
    shift_is_one = mant_base >= _uint(3, 4)
    sig_shift1 = fit_width(mant_base << 1, UInt(4))
    sig_shift2 = fit_width(mant_base << 2, UInt(5))
    sig_shift2_u4 = fit_width(sig_shift2, UInt(4))
    sig_sub = mux(shift_is_one, sig_shift1, sig_shift2_u4)
    exp_sub = mux(shift_is_one, _const_sint(-15), _const_sint(-16))

    sig_sel = mux(exp_zero, sig_sub, sig_norm)
    exp_sel = mux(exp_zero, exp_sub, exp_norm)

    mant_full = fit_width(sig_sel, UInt(4)) << 3
    exp_after_shift = fit_width(exp_sel, SInt(8))

    mant_d4, exp_d4 = _round_bucket(mant_full, exp_after_shift, keep_frac_bits=1)
    mant_d3, exp_d3 = _round_bucket(mant_full, exp_after_shift, keep_frac_bits=2)
    mant_d2, exp_d2 = _round_bucket(mant_full, exp_after_shift, keep_frac_bits=3)

    abs_exp_d4, exp_d4_neg = _abs_sint(exp_d4)
    abs_exp_d3, exp_d3_neg = _abs_sint(exp_d3)
    abs_exp_d2, exp_d2_neg = _abs_sint(exp_d2)

    valid_d4_raw = (abs_exp_d4 >= _const_uint(8, 8)) & (abs_exp_d4 <= _const_uint(8, 15))
    valid_d3_raw = (abs_exp_d3 >= _const_uint(8, 4)) & (abs_exp_d3 <= _const_uint(8, 7))
    valid_d2_raw = (abs_exp_d2 >= _const_uint(8, 2)) & (abs_exp_d2 <= _const_uint(8, 3))
    valid_d1_raw = abs_exp_d2 == _const_uint(8, 1)
    valid_d0_raw = exp_d2 == _const_sint(0)

    valid_d0 = valid_d0_raw
    valid_d1 = (~valid_d0) & valid_d1_raw
    valid_d2 = (~valid_d0) & (~valid_d1) & valid_d2_raw
    valid_d3 = (~valid_d0) & (~valid_d1) & (~valid_d2) & valid_d3_raw
    valid_d4 = (~valid_d0) & (~valid_d1) & (~valid_d2) & (~valid_d3) & valid_d4_raw

    mant_frac_d4 = mant_d4[0:1]
    mant_frac_d3 = mant_d3[0:2]
    mant_frac_d2 = mant_d2[0:3]

    mag_tail_d4 = abs_exp_d4 - _const_uint(8, 8)
    mag_tail_d3 = abs_exp_d3 - _const_uint(8, 4)
    mag_tail_d2 = abs_exp_d2 - _const_uint(8, 2)

    dot_d4 = _const_uint(2, 0b11)
    dot_d3 = _const_uint(2, 0b10)
    dot_d2 = _const_uint(2, 0b01)
    dot_d1 = _const_uint(3, 0b001)
    dot_d0 = _const_uint(4, 0b0001)
    dot_dml = _const_uint(4, 0b0000)

    payload_d4 = cat(mant_frac_d4, mag_tail_d4[0:3], exp_d4_neg, dot_d4)
    payload_d3 = cat(mant_frac_d3, mag_tail_d3[0:2], exp_d3_neg, dot_d3)
    payload_d2 = cat(mant_frac_d2, mag_tail_d2[0:1], exp_d2_neg, dot_d2)
    payload_d1 = cat(mant_frac_d2, exp_d2_neg, dot_d1)
    payload_d0 = cat(mant_frac_d2, dot_d0)

    threshold_dml = _const_uint(7, 91)
    round_up_dml = (mant_full >= threshold_dml) | (exp_after_shift <= _const_sint(-23))
    exp_dml = exp_after_shift + mux(round_up_dml, _const_sint(1), _const_sint(0))
    valid_dml = (exp_dml <= _const_sint(-16)) & (exp_dml >= _const_sint(-22))
    mant_dml = exp_dml + _const_sint(23)
    mant_dml_bits = fit_width(mant_dml, UInt(8))[0:3]

    normal_payload = _uint(8, 0)
    normal_payload = mux(valid_d0, payload_d0, normal_payload)
    normal_payload = mux(valid_d1, payload_d1, normal_payload)
    normal_payload = mux(valid_d2, payload_d2, normal_payload)
    normal_payload = mux(valid_d3, payload_d3, normal_payload)
    normal_payload = mux(valid_d4, payload_d4, normal_payload)

    normal_valid = valid_d4 | valid_d3 | valid_d2 | valid_d1 | valid_d0

    dml_promote_to_d4 = (
        (~normal_valid)
        & (~valid_dml)
        & (exp_dml == _const_sint(-15))
    )
    payload_d4_promote = cat(
        _uint(1, 0),
        _uint(3, 0b111),
        _uint(1, 1),
        dot_d4,
    )
    normal_payload = mux(dml_promote_to_d4, payload_d4_promote, normal_payload)
    normal_valid = normal_valid | dml_promote_to_d4

    payload_dml = cat(mant_dml_bits, dot_dml)
    packed_inf = mux(sign, _uint(8, 0xEF), _uint(8, 0x6F))
    packed_zero = _uint(8, 0x00)
    packed_nan = _uint(8, 0x80)

    overflow = exp_d4 > _const_sint(15)
    underflow_neg_nan = (
        (~is_nan)
        & (~is_inf)
        & (sign == _uint(1, 1))
        & (exp_after_shift <= _const_sint(-23))
        & (~valid_dml)
        & (~is_zero)
    )
    inf_flag = (~is_nan) & (is_inf | overflow)

    dml_zero = mant_dml_bits == _const_uint(3, 0)
    use_dml = (~normal_valid) & valid_dml & (~dml_zero)

    zero_flag = (~is_nan) & (~inf_flag) & (~underflow_neg_nan) & (
        is_zero
        | (~normal_valid & ~valid_dml)
        | (valid_dml & dml_zero)
    )

    payload_sel = _uint(7, 0)
    sign_sel = _uint(1, 0)
    payload_sel = mux(use_dml, payload_dml, payload_sel)
    sign_sel = mux(use_dml, sign, sign_sel)
    payload_sel = mux(normal_valid, normal_payload, payload_sel)
    sign_sel = mux(normal_valid, sign, sign_sel)

    payload_ext = fit_width(payload_sel, UInt(8))
    sign_ext = fit_width(sign_sel, UInt(8)) << 7
    non_special = payload_ext | sign_ext

    result = mux(
        is_nan | underflow_neg_nan,
        packed_nan,
        mux(
            inf_flag,
            packed_inf,
            mux(
                zero_flag,
                packed_zero,
                non_special,
            ),
        ),
    )

    return result


# ---------------------------------------------------------------------------
# Component definition
# ---------------------------------------------------------------------------


class HiF8MulViaFP8(Component):
    """Component that multiplies two HiFloat8 numbers using an FP8 core."""

    def __init__(self):
        self.io = IORecord(
            a=Input(UInt(8)),
            b=Input(UInt(8)),
            y=Output(UInt(8)),
        )
        self.elaborate()

    def elaborate(self):
        enc_a = _hif8_to_fp8_expr(self.io.a)
        enc_b = _hif8_to_fp8_expr(self.io.b)

        core = Component.from_netlist(build_fp_mul("HiF8Fp8Core", EW=_FP_EW, FW=_FP_FW))
        core.io.a <<= enc_a
        core.io.b <<= enc_b

        dec_y = _fp8_to_hif8_expr(core.io.y)
        self.io.y <<= dec_y


def build_hif8_mul_via_fp8_module(name: str = "HiF8MulViaFP8") -> Netlist:
    """Return a Netlist wrapper around :class:`HiF8MulViaFP8`."""

    comp = HiF8MulViaFP8()
    return comp.to_netlist(name)


def build_hif8_mul_via_fp8_component() -> HiF8MulViaFP8:
    """Construct the HiFloat8×HiFloat8 component instance."""

    return HiF8MulViaFP8()


# -----------------------------------------------------------------------------
# Standalone smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # get module and print Verilog
    mod = build_hif8_mul_via_fp8_module()

    # get transistor count
    from spire.helpers import get_yosys_transistor_count

    transistor_count = get_yosys_transistor_count(mod)
    print(f"Estimated transistor count: {transistor_count}")
