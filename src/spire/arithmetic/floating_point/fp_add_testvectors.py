"""Test vector generation for the FP adder (FpAdd).

Provides random, exhaustive, and hand-crafted vector generators following
the same tuple-based format as the integer arithmetic test vectors:
    [(name, {"a": bits, "b": bits}, {"y": bits}), ...]

Also provides targeted edge-case builders (normals, specials, subnormals)
that return 4-tuples ``(name, a_bits, b_bits, expected_bits)`` for use
with ``run_vectors_aby``-style test harnesses.
"""
from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np

from spire.arithmetic.floating_point.fp_encoding import (
    bits_inf,
    bits_max_finite,
    bits_max_sub,
    bits_min_normal,
    bits_min_sub,
    bits_qnan,
    bits_zero,
    fp_decode,
    fp_encode,
)

TestVectors = List[Tuple[str, Dict[str, int], Dict[str, int]]]


class FpAddTestVectors:
    """Random test vector generator for the FP adder."""

    def __init__(
        self,
        EW: int,
        FW: int,
        num_vectors: int = 10_000,
        subnormals: bool = True,
        seed: int = 42,
    ) -> None:
        self.EW = EW
        self.FW = FW
        self.W = 1 + EW + FW
        self.num_vectors = num_vectors
        self.subnormals = subnormals
        self.seed = seed

    def generate(self) -> TestVectors:
        rng = np.random.default_rng(self.seed)
        vectors: TestVectors = []
        for i in range(self.num_vectors):
            a = int(rng.integers(0, 1 << self.W))
            b = int(rng.integers(0, 1 << self.W))
            y = fp_encode(
                fp_decode(a, self.EW, self.FW) + fp_decode(b, self.EW, self.FW),
                self.EW, self.FW, subnormals=self.subnormals,
            )
            vectors.append((f"v{i}", {"a": a, "b": b}, {"y": y}))
        return vectors


class FpAddTestVectorsExhaustive:
    """Exhaustive test vector generator for the FP adder (small formats)."""

    def __init__(
        self,
        EW: int,
        FW: int,
        subnormals: bool = True,
    ) -> None:
        self.EW = EW
        self.FW = FW
        self.W = 1 + EW + FW
        self.subnormals = subnormals

    def generate(self) -> TestVectors:
        vectors: TestVectors = []
        for a in range(1 << self.W):
            for b in range(1 << self.W):
                y = fp_encode(
                    fp_decode(a, self.EW, self.FW) + fp_decode(b, self.EW, self.FW),
                    self.EW, self.FW, subnormals=self.subnormals,
                )
                vectors.append((f"v{len(vectors)}", {"a": a, "b": b}, {"y": y}))
        return vectors


# ---------------------------------------------------------------------------
# Targeted edge-case vectors (4-tuple format: name, a, b, expected)
# ---------------------------------------------------------------------------


def build_add_vectors(EW: int, FW: int):
    """Basic sanity vectors (normals/specials) for addition; works for any (EW, FW)."""
    one = fp_encode(1.0, EW, FW)
    two = fp_encode(2.0, EW, FW)
    thr = fp_encode(3.0, EW, FW)
    half = fp_encode(0.5, EW, FW)
    onept5 = fp_encode(1.5, EW, FW)
    onept25 = fp_encode(1.25, EW, FW)
    onept75 = fp_encode(1.75, EW, FW)
    two_pt_five = fp_encode(2.5, EW, FW)
    three_qtr = fp_encode(0.75, EW, FW)
    neg_onept25 = fp_encode(-1.25, EW, FW)
    neg_half = fp_encode(-0.5, EW, FW)
    neg_two = fp_encode(-2.0, EW, FW)
    min_norm = bits_min_normal(EW, FW)
    max_sub = bits_max_sub(EW, FW)
    neg_min_norm = min_norm | (1 << (EW + FW))
    neg_max_sub = max_sub | (1 << (EW + FW))
    pos0 = bits_zero(EW, FW, 0)
    neg0 = bits_zero(EW, FW, 1)
    pinf = bits_inf(EW, FW, 0)
    ninf = bits_inf(EW, FW, 1)
    qnan = bits_qnan(EW, FW)

    return [
        ("1+1=2", one, one, two),
        ("2+1=3", two, one, thr),
        ("1+0.5=1.5", one, half, onept5),
        ("0.75+0.75=1.5", three_qtr, three_qtr, onept5),
        ("1.25+0.5=1.75", onept25, half, onept75),
        ("1.5+(-1.25)=0.25", onept5, neg_onept25, fp_encode(0.25, EW, FW)),
        ("1.75+(-0.5)=1.25", onept75, neg_half, onept25),
        ("1.5+1.5=3.0", onept5, onept5, thr),
        ("2.5+1.5=4.0", two_pt_five, onept5, fp_encode(4.0, EW, FW)),
        ("minNorm+(-maxSub)=minSub", min_norm, neg_max_sub, bits_min_sub(EW, FW)),
        ("minNorm+(-minNorm)=+0", min_norm, neg_min_norm, pos0),
        ("-2+2=0", neg_two, two, pos0),
        ("(-0)+0=0", neg0, pos0, pos0),
        ("inf+(-inf)=nan", pinf, ninf, qnan),
        ("inf+1=inf", pinf, one, pinf),
    ]


def build_add_subnormal_vectors(EW: int, FW: int):
    """Subnormal-focused addition tests (assumes DUT supports subnormals)."""
    min_norm = bits_min_normal(EW, FW)
    min_sub = bits_min_sub(EW, FW)
    max_sub = bits_max_sub(EW, FW)
    neg_max_sub = max_sub | (1 << (EW + FW))
    neg_min_norm = min_norm | (1 << (EW + FW))
    pos0 = bits_zero(EW, FW, 0)

    return [
        ("minSub+minSub=2*minSub", min_sub, min_sub,
         fp_encode(fp_decode(min_sub, EW, FW) * 2, EW, FW)),
        ("maxSub+minSub=minNorm", max_sub, min_sub, min_norm),
        ("minNorm+(-maxSub)=minSub", min_norm, neg_max_sub, min_sub),
        ("minNorm+(-minNorm)=+0", min_norm, neg_min_norm, pos0),
        ("maxSub+maxSub", max_sub, max_sub,
         fp_encode(fp_decode(max_sub, EW, FW) * 2, EW, FW)),
    ]


def build_add_rounding_vectors(EW: int, FW: int):
    """Rounding edge cases for addition: alignment shift and tie-to-even.

    Exercises the guard/round/sticky logic by adding values whose exponent
    difference causes the smaller operand to be right-shifted so the discarded
    bits land exactly on the rounding boundary.
    """
    bias = (1 << (EW - 1)) - 1
    vecs = []

    # 1. Large + small where the small operand is just above/at/below the
    #    tie boundary after alignment.  Use 1.0 + 1.0*2^-(FW+1) which is
    #    exactly at the tie (half-ULP of 1.0).
    one = fp_encode(1.0, EW, FW)

    # 1 + ulp/2 -> tie-to-even -> rounds to 1.0 (LSB of 1.0 is 0)
    half_ulp = 2.0 ** -(FW + 1)
    val = 1.0 + half_ulp
    vecs.append(("1+half_ulp (tie->even)", one, fp_encode(half_ulp, EW, FW),
                 fp_encode(val, EW, FW)))

    # 1 + ulp -> rounds up to 1+ulp
    ulp_1 = 2.0 ** -FW
    vecs.append(("1+ulp", one, fp_encode(ulp_1, EW, FW),
                 fp_encode(1.0 + ulp_1, EW, FW)))

    # 2. Near-cancellation: subtracting nearly equal values.
    #    1.0 + (-(1-ulp)) = ulp  (massive cancellation, tests leading-zero detection)
    one_minus_ulp = 1.0 - ulp_1
    neg_one_minus_ulp = fp_encode(-one_minus_ulp, EW, FW)
    vecs.append(("1+(-1+ulp)=ulp (cancellation)", one, neg_one_minus_ulp,
                 fp_encode(1.0 + (-one_minus_ulp), EW, FW)))

    # 3. Addition that causes exponent increment (carry-out).
    #    max_mantissa + max_mantissa at same exponent -> carries into next exponent.
    max_frac = (1 << FW) - 1
    e_mid = bias  # exponent = 0, value = 1.xxx...
    a_bits = (e_mid << FW) | max_frac  # 1.111...1 * 2^0
    a_val = fp_decode(a_bits, EW, FW)
    vecs.append(("max_frac+max_frac (carry-out)", a_bits, a_bits,
                 fp_encode(a_val + a_val, EW, FW)))

    # 4. Subtraction producing subnormal from normal.
    #    smallest normal - one ulp of normal -> largest subnormal.
    min_norm = bits_min_normal(EW, FW)
    min_sub = bits_min_sub(EW, FW)
    neg_min_sub = min_sub | (1 << (EW + FW))
    vecs.append(("minNorm-minSub (normal->sub boundary)", min_norm, neg_min_sub,
                 fp_encode(fp_decode(min_norm, EW, FW) - fp_decode(min_sub, EW, FW), EW, FW)))

    # 5. Large exponent difference: the smaller operand is completely shifted out.
    #    1.0 + tiny (tiny < half-ULP of 1.0) -> 1.0  (sticky bit only, no rounding)
    tiny = 2.0 ** -(FW + 3)
    vecs.append(("1+tiny (shifted out, sticky only)", one, fp_encode(tiny, EW, FW),
                 fp_encode(1.0 + tiny, EW, FW)))

    # 6. Negative result rounding: (-1) + half_ulp -> tie-to-even
    neg_one = fp_encode(-1.0, EW, FW)
    vecs.append(("(-1)+half_ulp (tie->even neg)", neg_one, fp_encode(half_ulp, EW, FW),
                 fp_encode(-1.0 + half_ulp, EW, FW)))

    # 7. Overflow via addition: max_finite + max_finite -> inf
    maxf = bits_max_finite(EW, FW)
    pinf = bits_inf(EW, FW, 0)
    vecs.append(("max+max=inf (overflow)", maxf, maxf, pinf))

    return vecs


def build_targeted_add_vectors(EW: int, FW: int, subnormals: bool = True) -> TestVectors:
    """Collect all applicable targeted add vectors in the standard TestVectors format."""
    four_tuples = build_add_vectors(EW, FW)
    four_tuples += build_add_rounding_vectors(EW, FW)
    if subnormals:
        four_tuples += build_add_subnormal_vectors(EW, FW)
    return [(name, {"a": a, "b": b}, {"y": y}) for name, a, b, y in four_tuples]
