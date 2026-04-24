# =========================
# HiFloat8 reference utilities
# =========================
# This module provides a pure-Python reference model for the Ascend HiFloat8
# (HiF8) tapered floating-point format as defined in
#   "Ascend HiFloat8 Format for Deep Learning", arXiv:2409.16626.
#
# The focus is on accurate encode/decode helpers and a behavioural multiplier
# that observes "rounding-half-away-from-zero" (TA) as described in the paper.
# These utilities are useful for golden-model simulation and test-vector
# generation.  Hardware generation is intentionally left separate because
# synthesising the full tapered arithmetic datapath is a substantial effort
# beyond the scope of this helper.

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

try:
    from spirehdl.helpers import get_yosys_transistor_count
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def get_yosys_transistor_count(*_args, **_kwargs):  # type: ignore
        raise RuntimeError(
            "get_yosys_transistor_count requires the optional 'aigverse' dependency"
        )
from spirehdl.spirehdl import *
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator

# -----------------------------------------------------------------------------
# Shared constants / helpers for hardware path
# -----------------------------------------------------------------------------

EXP_INT_WIDTH = 8  # signed exponent math range [-128, 127]


def _const_sint(val: int) -> Expr:
    return Const(val, SInt(EXP_INT_WIDTH))


def _const_uint(width: int, val: int) -> Expr:
    return Const(val, UInt(width))


def _abs_sint(expr: Expr) -> Tuple[Expr, Expr]:
    """Return (abs(expr) as UInt(EXP_INT_WIDTH), is_negative Bool)."""
    is_neg = expr < _const_sint(0)
    neg_val = _const_sint(0) - expr
    abs_val = mux(
        is_neg,
        fit_width(neg_val, UInt(EXP_INT_WIDTH)),
        fit_width(expr, UInt(EXP_INT_WIDTH)),
    )
    return abs_val, is_neg


def _or_reduce(expr: Expr, width: int) -> Expr:
    if width <= 0:
        return _const_uint(1, 0)
    acc = expr[0]
    for i in range(1, width):
        acc = acc | expr[i]
    return acc


def _round_bucket(mant_full: Expr, exp_in: Expr, keep_frac_bits: int) -> Tuple[Expr, Expr]:
    shift = 6 - keep_frac_bits
    width_full = mant_full.typ.width
    kept = mant_full if shift <= 0 else mant_full >> shift
    kept_width = width_full if shift <= 0 else width_full - shift

    if shift <= 0:
        round_up_bit = _const_uint(1, 0)
    else:
        guard_bit = mant_full[shift - 1]
        round_up_bit = guard_bit

    sum_width = kept_width + 1
    kept_ext = fit_width(kept, UInt(sum_width))
    round_ext = fit_width(round_up_bit, UInt(sum_width))
    mant_sum = kept_ext + round_ext
    overflow = mant_sum[sum_width - 1]
    mant_post = mux(overflow, mant_sum >> 1, mant_sum)
    if keep_frac_bits == 1:
        max_finite = _const_uint(sum_width, 3)
        abs_exp_tmp, exp_neg_tmp = _abs_sint(exp_in + mux(overflow, _const_sint(1), _const_sint(0)))
        hits_inf = (
            (mant_post == max_finite)
            & (~exp_neg_tmp)
            & (abs_exp_tmp == _const_uint(EXP_INT_WIDTH, 15))
        )
        mant_post = mux(hits_inf, kept_ext, mant_post)
        overflow = mux(hits_inf, _const_uint(1, 0), overflow)
    exp_out = exp_in + mux(overflow, _const_sint(1), _const_sint(0))
    mant_out = mant_post[0:kept_width]
    return mant_out, exp_out


# -----------------------------------------------------------------------------
# Encoding helpers
# -----------------------------------------------------------------------------


def _is_nan(enc: int) -> bool:
    return enc == 0x80


def _is_inf(enc: int) -> bool:
    return enc in (0x6F, 0xEF)


def _is_zero(enc: int) -> bool:
    return enc == 0x00


def hif8_to_float(enc: int) -> float:
    """Decode an 8-bit HiFloat8 payload into a Python float."""
    if _is_nan(enc):
        return math.nan
    if _is_zero(enc):
        return 0.0
    if _is_inf(enc):
        return -math.inf if enc & 0x80 else math.inf

    sign = -1.0 if (enc & 0x80) else 1.0
    payload = enc & 0x7F

    # Dot-field prefix decoding (variable length)
    bits = payload
    if bits >> 5 == 0b11:  # D = 4  (2-bit dot)
        exp_field = (bits >> 1) & 0xF
        mant_field = bits & 0x1
        exp_sign = (exp_field >> 3) & 0x1
        mag = 8 + (exp_field & 0x7)
        exponent = -mag if exp_sign else mag
        significand = 1.0 + mant_field / 2.0
    elif bits >> 5 == 0b10:  # D = 3
        exp_field = (bits >> 2) & 0x7
        mant_field = bits & 0x3
        exp_sign = (exp_field >> 2) & 0x1
        mag = 4 + (exp_field & 0x3)
        exponent = -mag if exp_sign else mag
        significand = 1.0 + mant_field / 4.0
    elif bits >> 5 == 0b01:  # D = 2
        exp_field = (bits >> 3) & 0x3
        mant_field = bits & 0x7
        exp_sign = (exp_field >> 1) & 0x1
        mag = 2 + (exp_field & 0x1)
        exponent = -mag if exp_sign else mag
        significand = 1.0 + mant_field / 8.0
    elif bits >> 4 == 0b001:  # D = 1
        exp_sign = (bits >> 3) & 0x1
        exponent = -1 if exp_sign else 1
        mant_field = bits & 0x7
        significand = 1.0 + mant_field / 8.0
    elif bits >> 3 == 0b0001:  # D = 0
        exponent = 0
        mant_field = bits & 0x7
        significand = 1.0 + mant_field / 8.0
    else:  # DML denormals
        mant_field = bits & 0x7
        if mant_field == 0:
            return 0.0
        exponent = mant_field - 23  # M encodes exponent offset
        significand = 1.0

    return sign * math.ldexp(significand, exponent)


@dataclass(frozen=True)
class HiF8Value:
    encoding: int
    value: float


@lru_cache(maxsize=None)
def _positive_hif8_catalogue() -> List[HiF8Value]:
    """Enumerate all non-negative, non-NaN HiF8 encodings with their values."""
    catalogue: List[HiF8Value] = []
    for enc in range(0x00, 0x80):
        if _is_nan(enc):
            continue
        val = hif8_to_float(enc)
        if math.isnan(val):
            continue
        catalogue.append(HiF8Value(enc, val))
    # Sort by numeric value to ease binary search / tie-breaking.
    catalogue.sort(key=lambda v: v.value)
    return catalogue


def _tie_away_key(target: float, candidate: HiF8Value) -> Tuple[float, float]:
    diff = abs(candidate.value - target)
    # Tie-away-from-zero: prefer larger magnitude on equal difference.
    return (diff, -candidate.value)


def float_to_hif8(value: float) -> int:
    """Encode a Python float into HiFloat8 using TA rounding."""
    if math.isnan(value):
        return 0x80  # canonical NaN
    if math.isinf(value):
        return 0xEF if value < 0 else 0x6F

    sign = 1 if math.copysign(1.0, value) < 0 else 0
    abs_val = abs(value)

    if abs_val == 0.0:
        return 0x00

    catalogue = _positive_hif8_catalogue()
    best = min(catalogue, key=lambda cand: _tie_away_key(abs_val, cand))
    return best.encoding | (sign << 7)


def multiply_hif8(a: int, b: int) -> int:
    """HiF8 multiply using the reference encode/decode helpers."""
    fa = hif8_to_float(a)
    fb = hif8_to_float(b)

    if math.isnan(fa) or math.isnan(fb):
        return 0x80
    if (math.isinf(fa) and fb == 0.0) or (math.isinf(fb) and fa == 0.0):
        return 0x80

    product = fa * fb
    return float_to_hif8(product)


# -----------------------------------------------------------------------------
# Hardware module builder
# -----------------------------------------------------------------------------


def build_hif8_mul_logic(name: str = "HiF8Mul", *, debug: bool = False) -> Module:
    m = Module(name, with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(8), "y")

    da = _decode_operand_expr(a)
    db = _decode_operand_expr(b)

    sign_out = da["sign"] ^ db["sign"]

    is_nan_in = da["is_nan"] | db["is_nan"] | (
        (da["is_inf"] & db["is_zero"]) | (db["is_inf"] & da["is_zero"])
    )
    inf_in = da["is_inf"] | db["is_inf"]
    zero_in = da["is_zero"] | db["is_zero"]

    sig_prod = da["sig"] * db["sig"]
    needs_shift = sig_prod[7]
    sig_after_shift = mux(needs_shift, sig_prod >> 1, sig_prod)
    exp_after_shift = da["exp"] + db["exp"] + mux(needs_shift, _const_sint(1), _const_sint(0))
    mant_full = sig_after_shift[0:7]

    mant_d4, exp_d4 = _round_bucket(mant_full, exp_after_shift, keep_frac_bits=1)
    mant_d3, exp_d3 = _round_bucket(mant_full, exp_after_shift, keep_frac_bits=2)
    mant_d2, exp_d2 = _round_bucket(mant_full, exp_after_shift, keep_frac_bits=3)

    abs_exp_d4, exp_d4_neg = _abs_sint(exp_d4)
    abs_exp_d3, exp_d3_neg = _abs_sint(exp_d3)
    abs_exp_d2, exp_d2_neg = _abs_sint(exp_d2)

    # valid_d4_raw = (abs_exp_d4 >= _const_uint(EXP_INT_WIDTH, 8)) & (
    #     abs_exp_d4 <= _const_uint(EXP_INT_WIDTH, 15)
    # )
    # valid_d3_raw = (abs_exp_d3 >= _const_uint(EXP_INT_WIDTH, 4)) & (
    #     abs_exp_d3 <= _const_uint(EXP_INT_WIDTH, 7)
    # )
    # valid_d2_raw = (abs_exp_d2 >= _const_uint(EXP_INT_WIDTH, 2)) & (
    #     abs_exp_d2 <= _const_uint(EXP_INT_WIDTH, 3)
    # )
    # valid_d1_raw = abs_exp_d2 == _const_uint(EXP_INT_WIDTH, 1)
    # valid_d0_raw = exp_d2 == _const_sint(0)

    # valid_d0 = valid_d0_raw
    # valid_d1 = (~valid_d0) & valid_d1_raw
    # valid_d2 = (~valid_d0) & (~valid_d1) & valid_d2_raw
    # valid_d3 = (~valid_d0) & (~valid_d1) & (~valid_d2) & valid_d3_raw
    # valid_d4 = (~valid_d0) & (~valid_d1) & (~valid_d2) & (~valid_d3) & valid_d4_raw
    
    valid_d4 = (abs_exp_d4 >= _const_uint(EXP_INT_WIDTH, 8)) & (
        abs_exp_d4 <= _const_uint(EXP_INT_WIDTH, 15)
    )
    valid_d3 = (abs_exp_d3 >= _const_uint(EXP_INT_WIDTH, 4)) & (
        abs_exp_d3 <= _const_uint(EXP_INT_WIDTH, 7)
    )
    valid_d2 = (abs_exp_d2 >= _const_uint(EXP_INT_WIDTH, 2)) & (
        abs_exp_d2 <= _const_uint(EXP_INT_WIDTH, 3)
    )
    valid_d1 = abs_exp_d2 == _const_uint(EXP_INT_WIDTH, 1)
    valid_d0 = exp_d2 == _const_sint(0)

    mant_frac_d4 = mant_d4[0:1]
    mant_frac_d3 = mant_d3[0:2]
    mant_frac_d2 = mant_d2[0:3]

    mag_tail_d4 = abs_exp_d4 - _const_uint(EXP_INT_WIDTH, 8)
    mag_tail_d3 = abs_exp_d3 - _const_uint(EXP_INT_WIDTH, 4)
    mag_tail_d2 = abs_exp_d2 - _const_uint(EXP_INT_WIDTH, 2)

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
    mant_dml_bits = fit_width(mant_dml, UInt(EXP_INT_WIDTH))[0:3]

    normal_payload = _const_uint(8, 0)
    normal_payload = mux(valid_d0, payload_d0, normal_payload)
    normal_payload = mux(valid_d1, payload_d1, normal_payload)
    normal_payload = mux(valid_d2, payload_d2, normal_payload)
    normal_payload = mux(valid_d3, payload_d3, normal_payload)
    normal_payload = mux(valid_d4, payload_d4, normal_payload)

    normal_valid = valid_d4 | valid_d3 | valid_d2 | valid_d1 | valid_d0

    # added dml rounding promotion
    dml_promote_to_d4 = (
        (~normal_valid)
        & (~valid_dml)
        & (exp_dml == _const_sint(-15))
    )
    payload_d4_promote = cat(
        _const_uint(1, 0),
        _const_uint(3, 0b111),
        _const_uint(1, 1),
        dot_d4,
    )
    normal_payload = mux(dml_promote_to_d4, payload_d4_promote, normal_payload)
    normal_valid = normal_valid | dml_promote_to_d4
    # end dml rounding promotion

    payload_dml = cat(mant_dml_bits, dot_dml)
    packed_inf = mux(sign_out, _const_uint(8, 0xEF), _const_uint(8, 0x6F))
    packed_zero = _const_uint(8, 0x00)
    packed_nan = _const_uint(8, 0x80)

    overflow = exp_d4 > _const_sint(15)
    # Results smaller than the DML window (|exp| < 23 after normalization)
    # are treated as zero (not NaN), matching the reference path.
    underflow_neg_nan = (
        (~is_nan_in)
        & (~inf_in)
        & (sign_out == _const_uint(1, 1))
        & (exp_after_shift <= _const_sint(-23))
        & (~valid_dml)
        & (~zero_in)
    )
    is_inf = (~is_nan_in) & (inf_in | overflow)

    dml_zero = mant_dml_bits == _const_uint(3, 0)
    use_dml = (~normal_valid) & valid_dml & (~dml_zero)

    is_zero = (~is_nan_in) & (~is_inf) & (
        zero_in
        | (~normal_valid & ~valid_dml)
        | (valid_dml & dml_zero)
    )

    payload_sel = _const_uint(7, 0)
    sign_sel = _const_uint(1, 0)
    payload_sel = mux(use_dml, payload_dml, payload_sel)
    sign_sel = mux(use_dml, sign_out, sign_sel)
    payload_sel = mux(normal_valid, normal_payload, payload_sel)
    sign_sel = mux(normal_valid, sign_out, sign_sel)

    payload_ext = fit_width(payload_sel, UInt(8))
    sign_ext = fit_width(sign_sel, UInt(8)) << 7
    non_special = payload_ext | sign_ext

    result = mux(
        is_nan_in | underflow_neg_nan,
        packed_nan,
        mux(
            is_inf,
            packed_inf,
            mux(
                is_zero,
                packed_zero,
                non_special,
            ),
        ),
    )

    y <<= result
    if debug:
        dbg_sign = m.output(UInt(1), "dbg_sign")
        dbg_sign <<= sign_sel
        dbg_use_dml = m.output(UInt(1), "dbg_use_dml")
        dbg_use_dml <<= use_dml
        dbg_norm = m.output(UInt(1), "dbg_norm_valid")
        dbg_norm <<= normal_valid
        dbg_signout = m.output(UInt(1), "dbg_sign_out")
        dbg_signout <<= sign_out
        dbg_zero = m.output(UInt(1), "dbg_is_zero")
        dbg_zero <<= is_zero
        dbg_nan = m.output(UInt(1), "dbg_is_nan")
        dbg_nan <<= (is_nan_in | underflow_neg_nan)
        dbg_inf = m.output(UInt(1), "dbg_is_inf")
        dbg_inf <<= is_inf
        dbg_payload = m.output(UInt(7), "dbg_payload")
        dbg_payload <<= payload_sel
        dbg_non = m.output(UInt(8), "dbg_non_special")
        dbg_non <<= non_special
        dbg_valid_d4 = m.output(UInt(1), "dbg_valid_d4")
        dbg_valid_d4 <<= valid_d4
        dbg_valid_d3 = m.output(UInt(1), "dbg_valid_d3")
        dbg_valid_d3 <<= valid_d3
        dbg_valid_d2 = m.output(UInt(1), "dbg_valid_d2")
        dbg_valid_d2 <<= valid_d2
        dbg_valid_d1 = m.output(UInt(1), "dbg_valid_d1")
        dbg_valid_d1 <<= valid_d1
        dbg_valid_d0 = m.output(UInt(1), "dbg_valid_d0")
        dbg_valid_d0 <<= valid_d0
        dbg_exp_d2 = m.output(SInt(EXP_INT_WIDTH), "dbg_exp_d2")
        dbg_exp_d2 <<= exp_d2
        dbg_exp_d3 = m.output(SInt(EXP_INT_WIDTH), "dbg_exp_d3")
        dbg_exp_d3 <<= exp_d3
        dbg_exp_d4 = m.output(SInt(EXP_INT_WIDTH), "dbg_exp_d4")
        dbg_exp_d4 <<= exp_d4
    return m

def _decode_operand_expr(x: Expr) -> Dict[str, Expr]:
    sign = x[7]

    dot_top2 = x[5:7]
    bit4 = x[4]
    bit3 = x[3]

    is_d4 = dot_top2 == _const_uint(2, 0b11)
    is_d3 = dot_top2 == _const_uint(2, 0b10)
    is_d2 = dot_top2 == _const_uint(2, 0b01)

    prefix00 = dot_top2 == _const_uint(2, 0b00)
    is_d1 = prefix00 & (bit4 == _const_uint(1, 1))
    is_d0 = prefix00 & (bit4 == _const_uint(1, 0)) & (bit3 == _const_uint(1, 1))
    is_dml = prefix00 & (bit4 == _const_uint(1, 0)) & (bit3 == _const_uint(1, 0))

    mant3 = x[0:3]
    mant2 = x[0:2]
    mant1 = x[0:1]

    mant_frac_d4 = mant1 << 2
    mant_frac_d3 = mant2 << 1

    sig_low = cat(mant3, _const_uint(1, 1))
    sig_d3 = cat(mant_frac_d3, _const_uint(1, 1))
    sig_d4 = cat(mant_frac_d4, _const_uint(1, 1))
    sig_dml = cat(_const_uint(3, 0), _const_uint(1, 1))

    sig = sig_low
    sig = mux(is_d3, sig_d3, sig)
    sig = mux(is_d4, sig_d4, sig)
    sig = mux(is_dml, sig_dml, sig)

    exp = _const_sint(0)

    exp_bits_d4 = x[1:5]
    exp_sign_d4 = exp_bits_d4[3]
    mag_tail_d4 = exp_bits_d4[0:3]
    mag_with_hidden_d4 = cat(mag_tail_d4, _const_uint(1, 1))
    mag_d4 = fit_type(fit_width(mag_with_hidden_d4, UInt(EXP_INT_WIDTH)), SInt(EXP_INT_WIDTH))
    exp_d4 = mux(exp_sign_d4, _const_sint(0) - mag_d4, mag_d4)

    exp_bits_d3 = x[2:5]
    exp_sign_d3 = exp_bits_d3[2]
    mag_tail_d3 = exp_bits_d3[0:2]
    mag_with_hidden_d3 = cat(mag_tail_d3, _const_uint(1, 1))
    mag_d3 = fit_type(fit_width(mag_with_hidden_d3, UInt(EXP_INT_WIDTH)), SInt(EXP_INT_WIDTH))
    exp_d3 = mux(exp_sign_d3, _const_sint(0) - mag_d3, mag_d3)

    exp_bits_d2 = x[3:5]
    exp_sign_d2 = exp_bits_d2[1]
    mag_tail_d2 = exp_bits_d2[0:1]
    mag_with_hidden_d2 = cat(mag_tail_d2, _const_uint(1, 1))
    mag_d2 = fit_type(fit_width(mag_with_hidden_d2, UInt(EXP_INT_WIDTH)), SInt(EXP_INT_WIDTH))
    exp_d2 = mux(exp_sign_d2, _const_sint(0) - mag_d2, mag_d2)

    exp_sign_d1 = x[3]
    exp_d1 = mux(exp_sign_d1, _const_sint(-1), _const_sint(1))

    mant_m = mant3
    exp_dml = fit_type(fit_width(mant_m, UInt(EXP_INT_WIDTH)), SInt(EXP_INT_WIDTH))
    exp_dml = exp_dml - _const_sint(23)

    exp = mux(is_d4, exp_d4, exp)
    exp = mux(is_d3, exp_d3, exp)
    exp = mux(is_d2, exp_d2, exp)
    exp = mux(is_d1, exp_d1, exp)
    exp = mux(is_d0, _const_sint(0), exp)
    exp = mux(is_dml, exp_dml, exp)

    is_zero = x == _const_uint(8, 0x00)
    is_nan = x == _const_uint(8, 0x80)

    exp_bits_for_inf = x[1:5]
    mant_bit_inf = x[0]
    is_inf = (
        is_d4
        & (exp_bits_for_inf[3] == _const_uint(1, 0))
        & (exp_bits_for_inf[0:3] == _const_uint(3, 0b111))
        & (mant_bit_inf == _const_uint(1, 1))
    )

    return {
        "sign": sign,
        "sig": sig,
        "exp": exp,
        "is_zero": is_zero,
        "is_nan": is_nan,
        "is_inf": is_inf,
        "is_dml": is_dml,
    }


def build_hif8_mul_lut(name: str = "HiF8Mul_LUT") -> Module:
    m = Module(name, with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(8), "y")

    table = [[multiply_hif8(av, bv) for bv in range(256)] for av in range(256)]

    a_low = a[0:4]
    a_high = a[4:8]
    b_low = b[0:4]
    b_high = b[4:8]

    result = _const_uint(8, 0)
    for ah in range(16):
        row_low = _const_uint(8, 0)
        for al in range(16):
            aval = (ah << 4) | al
            row = _const_uint(8, 0)
            for bh in range(16):
                group = _const_uint(8, 0)
                for bl in range(16):
                    bval = (bh << 4) | bl
                    prod = table[aval][bval]
                    group = mux(
                        b_low == _const_uint(4, bl),
                        _const_uint(8, prod),
                        group,
                    )
                row = mux(b_high == _const_uint(4, bh), group, row)
            row_low = mux(a_low == _const_uint(4, al), row, row_low)
        result = mux(a_high == _const_uint(4, ah), row_low, result)

    y <<= result
    return m


def build_hif8_mul_module(name: str = "HiF8Mul") -> Module:
    return build_hif8_mul_logic(name)

# -----------------------------------------------------------------------------
# Vector helpers for regression tests / demos
# -----------------------------------------------------------------------------

def catalogue_summary() -> Dict[str, int]:
    """Return basic counts for documentation / sanity checks."""
    cats = _positive_hif8_catalogue()
    normals = [v for v in cats if v.encoding not in (0x00,) and not (v.encoding >> 5 == 0)]
    denormals = [v for v in cats if (v.encoding >> 3) == 0 and v.encoding != 0x00]
    return {
        "total": len(cats),
        "normals": len(normals),
        "denormals": len(denormals),
    }


def build_basic_vectors() -> List[Tuple[str, int, int, int]]:
    """Simple regression vectors anchored on specification examples."""
    return [
        ("unity", 0x33, 0x33, multiply_hif8(0x33, 0x33)),
        ("2 * 0.5", 0x6C, 0x17, multiply_hif8(0x6C, 0x17)),
        ("zero * max", 0x00, 0x6E, multiply_hif8(0x00, 0x6E)),
        ("min denorm * 2", 0x01, 0x6C, multiply_hif8(0x01, 0x6C)),
        ("inf * finite", 0x6F, 0x33, multiply_hif8(0x6F, 0x33)),
        ("finite * inf", 0x33, 0x6F, multiply_hif8(0x33, 0x6F)),
    ]


# -----------------------------------------------------------------------------
# Standalone smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("HiF8 catalogue summary:", catalogue_summary())
    for name, a, b, exp in build_basic_vectors():
        fa = hif8_to_float(a)
        fb = hif8_to_float(b)
        fc = hif8_to_float(exp)
        prod = hif8_to_float(multiply_hif8(a, b))
        print(
            f"{name:12s}: a=0x{a:02X} ({fa:+.6e})  "
            f"b=0x{b:02X} ({fb:+.6e})  ->  "
            f"y=0x{exp:02X} ({fc:+.6e})  [recalc {prod:+.6e}]"
        )

    print("\nVerifying logic multiplier against reference model...")
    dut_logic = build_hif8_mul_logic("HiF8Mul_Logic_Ref")
    sim_logic = Simulator(dut_logic)
    mismatches_logic = 0
    mismatch_largest = 0
    largest_delta_tuple = None
    tests_done = 0
    tests_mismatch=0
    
    for aval in range(256):
        for bval in range(256):
            sim_logic.set("a", aval).set("b", bval).eval()
            got = sim_logic.get("y")
            exp = multiply_hif8(aval, bval)
            tests_done += 1
            if got != exp:
                tests_mismatch += 1
                if mismatches_logic < 50:
                    aval_float = hif8_to_float(aval)
                    bval_float = hif8_to_float(bval)
                    got_float = hif8_to_float(got)
                    exp_float = hif8_to_float(exp)
                    print(
                        f"Logic mismatch a=0x{aval:02X} ({aval_float:+.6e}) b=0x{bval:02X} ({bval_float:+.6e}) "
                        f"got=0x{got:02X} ({got_float:+.6e}) "
                        f"exp=0x{exp:02X} ({exp_float:+.6e})"
                    )                    
                mismatches_logic += 1
                if abs(got_float - exp_float) > mismatch_largest:
                    mismatch_largest = abs(got_float - exp_float)
                    largest_delta_tuple = (aval_float, bval_float, got_float, exp_float)
    if mismatches_logic == 0:
        print("Logic implementation matches all 65,536 combinations.")
    else:
        print(f"Logic mismatches: {mismatches_logic}, largest delta: {mismatch_largest}")
        print(f"Largest delta details (float): aval={largest_delta_tuple[0]:+.6e}, bval={largest_delta_tuple[1]:+.6e}, got={largest_delta_tuple[2]:+.6e}, exp={largest_delta_tuple[3]:+.6e}")
        print(f"Total tests done: {tests_done}, total mismatches: {tests_mismatch}")
    transistor_count = get_yosys_transistor_count(dut_logic)
    print(f"\nEstimated transistor count for logic multiplier: {transistor_count:,}")
    

    print("\nVerifying LUT multiplier against reference model...")
    dut_lut = build_hif8_mul_lut("HiF8Mul_LUT_Ref")
    sim_lut = Simulator(dut_lut)
    mismatches_lut = 0
    for aval in range(256):
        for bval in range(256):
            sim_lut.set("a", aval).set("b", bval).eval()
            got = sim_lut.get("y")
            exp = multiply_hif8(aval, bval)
            if got != exp:
                if mismatches_lut < 10:
                    print(
                        f"LUT mismatch a=0x{aval:02X} b=0x{bval:02X} "
                        f"got=0x{got:02X} exp=0x{exp:02X}"
                    )
                mismatches_lut += 1
    if mismatches_lut == 0:
        print("LUT implementation matches all 65,536 combinations.")
    else:
        print(f"LUT mismatches: {mismatches_lut}")

    transistor_count = get_yosys_transistor_count(dut_logic)
    print(f"\nEstimated transistor count for logic multiplier: {transistor_count:,}")
