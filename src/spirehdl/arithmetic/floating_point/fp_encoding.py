"""Floating-point bit encoding/decoding utilities for arbitrary (EW, FW) formats.

Pure-Python, no hardware dependencies; safe to import from both src/ and testing/.
"""
from __future__ import annotations

from math import copysign, floor, frexp, isinf, isnan, ldexp


def fp_bias(ew: int) -> int:
    return (1 << (ew - 1)) - 1


def fp_pack(sign: int, exp: int, frac: int, ew: int, fw: int) -> int:
    return ((sign & 1) << (ew + fw)) | ((exp & ((1 << ew) - 1)) << fw) | (frac & ((1 << fw) - 1))


def fp_unpack(bits: int, ew: int, fw: int) -> tuple[int, int, int]:
    sign = (bits >> (ew + fw)) & 1
    exp = (bits >> fw) & ((1 << ew) - 1)
    frac = bits & ((1 << fw) - 1)
    return sign, exp, frac


def _round_to_even(x: float) -> int:
    lo = floor(x)
    frac = x - lo
    if frac < 0.5:
        return int(lo)
    if frac > 0.5:
        return int(lo + 1)
    return int(lo if lo % 2 == 0 else lo + 1)


def fp_decode(bits: int, ew: int, fw: int) -> float:
    """IEEE-like decode (bias=2^(EW-1)-1), with subnormals and Inf/NaN."""
    s, e, f = fp_unpack(bits, ew, fw)
    bias = fp_bias(ew)
    if e == 0:
        if f == 0:
            return -0.0 if s else 0.0
        return copysign((f / (1 << fw)) * 2.0 ** (1 - bias), -1.0 if s else 1.0)
    if e == (1 << ew) - 1:
        if f == 0:
            return float("-inf") if s else float("inf")
        return float("nan")
    return copysign((1.0 + f / (1 << fw)) * 2.0 ** (e - bias), -1.0 if s else 1.0)


def fp_encode(x: float, ew: int, fw: int, *, subnormals: bool = True) -> int:
    """Encode Python float into IEEE-like (EW, FW) format with round-to-nearest-even."""
    if isnan(x):
        return fp_pack(0, (1 << ew) - 1, 1 << (fw - 1 if fw > 0 else 0), ew, fw)
    if isinf(x):
        return fp_pack(1 if x < 0 else 0, (1 << ew) - 1, 0, ew, fw)
    if x == 0.0:
        return fp_pack(1 if copysign(1.0, x) < 0 else 0, 0, 0, ew, fw)

    sgn = 1 if x < 0 else 0
    ax = abs(x)
    bias = fp_bias(ew)
    m, e = frexp(ax)
    m *= 2.0
    e -= 1

    E = e + bias
    if E >= (1 << ew) - 1:
        return fp_pack(sgn, (1 << ew) - 1, 0, ew, fw)

    if E <= 0:
        f_unrnd = ax * (2.0 ** (fw + bias - 1))
        f = max(0, _round_to_even(f_unrnd))
        if f >= (1 << fw):
            return fp_pack(sgn, 1, 0, ew, fw)  # rounds up to min_normal
        if not subnormals:
            return fp_pack(sgn, 0, 0, ew, fw)  # FTZ: truly subnormal → zero
        return fp_pack(sgn, 0, f, ew, fw)

    frac_unrnd = (m - 1.0) * (1 << fw)
    f = _round_to_even(frac_unrnd)
    if f == (1 << fw):
        f = 0
        E += 1
        if E >= (1 << ew) - 1:
            return fp_pack(sgn, (1 << ew) - 1, 0, ew, fw)
    return fp_pack(sgn, int(E), int(f), ew, fw)


# ---------------------------------------------------------------------------
# Convenience constructors for canonical bit patterns
# ---------------------------------------------------------------------------

def bits_zero(ew: int, fw: int, sign: int = 0) -> int:
    return fp_pack(sign, 0, 0, ew, fw)


def bits_inf(ew: int, fw: int, sign: int = 0) -> int:
    return fp_pack(sign, (1 << ew) - 1, 0, ew, fw)


def bits_qnan(ew: int, fw: int) -> int:
    return fp_pack(0, (1 << ew) - 1, 1 << (fw - 1 if fw > 0 else 0), ew, fw)


def bits_min_normal(ew: int, fw: int) -> int:
    return fp_pack(0, 1, 0, ew, fw)


def bits_max_finite(ew: int, fw: int, sign: int = 0) -> int:
    return fp_pack(sign, (1 << ew) - 2, (1 << fw) - 1, ew, fw)


def bits_min_sub(ew: int, fw: int) -> int:
    return fp_pack(0, 0, 1, ew, fw)


def bits_max_sub(ew: int, fw: int) -> int:
    return fp_pack(0, 0, (1 << fw) - 1, ew, fw)


def bits_pow2(k: int, ew: int, fw: int) -> int:
    """Exact power of two 2^k, when representable as a normal."""
    E = k + fp_bias(ew)
    if 1 <= E <= (1 << ew) - 2:
        return fp_pack(0, E, 0, ew, fw)
    if E >= (1 << ew) - 1:
        return bits_inf(ew, fw, 0)
    f = int(round(2.0 ** (k - (1 - fp_bias(ew))) * (1 << fw)))
    if f <= 0:
        return bits_zero(ew, fw, 0)
    if f >= (1 << fw):
        return bits_min_normal(ew, fw)
    return fp_pack(0, 0, f, ew, fw)


def fp_limits(ew: int, fw: int) -> dict:
    """Return a dict with max_finite, min_finite, overflow thresholds, min_normal_pos, min_sub_pos."""
    bias = fp_bias(ew)
    Emax_enc = (1 << ew) - 2
    e_unbiased = Emax_enc - bias
    max_finite = ldexp(2.0 - 2.0 ** (-fw), e_unbiased)
    pos_overflow_threshold = ldexp(2.0 - 2.0 ** (-(fw + 1)), e_unbiased)
    return dict(
        max_finite=max_finite,
        min_finite=-max_finite,
        pos_overflow_threshold=pos_overflow_threshold,
        neg_overflow_threshold=-pos_overflow_threshold,
        min_normal_pos=ldexp(1.0, 1 - bias),
        min_sub_pos=0.0 if fw == 0 else ldexp(1.0, 1 - bias - fw),
    )
