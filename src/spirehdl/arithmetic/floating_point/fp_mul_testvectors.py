"""Test vector generation for the FP multiplier (FpMulSN).

Provides random, exhaustive, and hand-crafted vector generators following
the same tuple-based format as the integer arithmetic test vectors:
    [(name, {"a": bits, "b": bits}, {"y": bits}), ...]

Also provides targeted edge-case builders (normals, specials, subnormals,
tie-to-even) that return 4-tuples ``(name, a_bits, b_bits, expected_bits)``
for use with ``run_vectors_aby``-style test harnesses.
"""
from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np

from spirehdl.arithmetic.floating_point.fp_encoding import (
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


def _is_subnormal(bits: int, EW: int, FW: int) -> bool:
    e = (bits >> FW) & ((1 << EW) - 1)
    f = bits & ((1 << FW) - 1)
    return e == 0 and f != 0


def _should_skip(
    a: int, b: int, product_val: float, y_enc: int,
    EW: int, FW: int, subnormals: bool, always_subnormal_rounding: bool,
) -> bool:
    """Return True if this vector should be skipped due to FTZ limitations."""
    if not subnormals:
        if _is_subnormal(a, EW, FW) or _is_subnormal(b, EW, FW):
            return True
        if not always_subnormal_rounding:
            min_normal_val = 2.0 ** (2 - (1 << (EW - 1)))
            if 0 < abs(product_val) < min_normal_val and y_enc != 0:
                return True
    return False


class FpMulTestVectors:
    """Random test vector generator for the FP multiplier."""

    def __init__(
        self,
        EW: int,
        FW: int,
        num_vectors: int = 10_000,
        subnormals: bool = True,
        always_subnormal_rounding: bool = False,
        seed: int = 42,
    ) -> None:
        self.EW = EW
        self.FW = FW
        self.W = 1 + EW + FW
        self.num_vectors = num_vectors
        self.subnormals = subnormals
        self.always_subnormal_rounding = always_subnormal_rounding
        self.seed = seed

    def generate(self) -> TestVectors:
        rng = np.random.default_rng(self.seed)
        vectors: TestVectors = []
        while len(vectors) < self.num_vectors:
            a = int(rng.integers(0, 1 << self.W))
            b = int(rng.integers(0, 1 << self.W))
            product_val = fp_decode(a, self.EW, self.FW) * fp_decode(b, self.EW, self.FW)
            y = fp_encode(product_val, self.EW, self.FW, subnormals=self.subnormals)
            if _should_skip(a, b, product_val, y,
                            self.EW, self.FW, self.subnormals, self.always_subnormal_rounding):
                continue
            vectors.append((f"v{len(vectors)}", {"a": a, "b": b}, {"y": y}))
        return vectors


class FpMulTestVectorsExhaustive:
    """Exhaustive test vector generator for the FP multiplier (small formats)."""

    def __init__(
        self,
        EW: int,
        FW: int,
        subnormals: bool = True,
        always_subnormal_rounding: bool = False,
    ) -> None:
        self.EW = EW
        self.FW = FW
        self.W = 1 + EW + FW
        self.subnormals = subnormals
        self.always_subnormal_rounding = always_subnormal_rounding

    def generate(self) -> TestVectors:
        vectors: TestVectors = []
        for a in range(1 << self.W):
            for b in range(1 << self.W):
                product_val = fp_decode(a, self.EW, self.FW) * fp_decode(b, self.EW, self.FW)
                y = fp_encode(product_val, self.EW, self.FW, subnormals=self.subnormals)
                if _should_skip(a, b, product_val, y,
                                self.EW, self.FW, self.subnormals, self.always_subnormal_rounding):
                    continue
                vectors.append((f"v{len(vectors)}", {"a": a, "b": b}, {"y": y}))
        return vectors


# ---------------------------------------------------------------------------
# Targeted edge-case vectors (4-tuple format: name, a, b, expected)
# ---------------------------------------------------------------------------


def build_fp_vectors(EW: int, FW: int, subnormals: bool = False):
    """Basic sanity vectors (normals/specials); works for any (EW, FW).

    The ``subnormals`` flag controls the expected result for the underflow
    case: with subnormals disabled (FTZ) the result is zero; with subnormals
    enabled the result is the correct subnormal encoding.
    """
    one = fp_encode(1.0, EW, FW)
    two = fp_encode(2.0, EW, FW)
    thr = fp_encode(3.0, EW, FW)
    four = fp_encode(4.0, EW, FW)
    half = fp_encode(0.5, EW, FW)
    onept5 = fp_encode(1.5, EW, FW)
    neg2 = fp_encode(-2.0, EW, FW)
    pos0 = bits_zero(EW, FW, 0)
    neg0 = bits_zero(EW, FW, 1)
    pinf = bits_inf(EW, FW, 0)
    ninf = bits_inf(EW, FW, 1)
    qnan = bits_qnan(EW, FW)
    maxf = bits_max_finite(EW, FW)
    minN = bits_min_normal(EW, FW)
    underflow_expected = fp_encode(
        fp_decode(minN, EW, FW) * 0.5, EW, FW, subnormals=subnormals,
    )

    return [
        ("1*2 = 2", one, two, fp_encode(2.0, EW, FW)),
        ("(-2)*2 = -4", neg2, two, fp_encode(-4.0, EW, FW)),
        ("1.5*1.5 = 2.25", onept5, onept5, fp_encode(2.25, EW, FW)),
        ("3*0.5 = 1.5", thr, half, onept5),
        ("0 * 1 = 0", pos0, one, pos0),
        ("(-0) * 2 = (-0)", neg0, two, neg0),
        ("Inf * 3 = Inf", pinf, thr, pinf),
        ("(-Inf) * (-2) = +Inf", ninf, neg2, pinf),
        ("Inf * 0 = NaN", pinf, pos0, qnan),
        ("NaN * 2 = NaN", qnan, two, qnan),
        ("Overflow: max*2 = Inf", maxf, two, pinf),
        ("Underflow: min*0.5", minN, half, underflow_expected),
        ("(-1)*1 = -1", fp_encode(-1.0, EW, FW), one, fp_encode(-1.0, EW, FW)),
        ("4 * 0.5 = 2", four, half, two),
    ]


def build_fp_subnormal_vectors(EW: int, FW: int):
    """Subnormal-focused tests (assumes DUT supports subnormals)."""
    half = fp_encode(0.5, EW, FW)
    qtr = fp_encode(0.25, EW, FW)
    eigth = fp_encode(0.125, EW, FW)
    six = fp_encode(0.0625, EW, FW)
    minN = bits_min_normal(EW, FW)
    minS = bits_min_sub(EW, FW)
    maxS = bits_max_sub(EW, FW)
    one = fp_encode(1.0, EW, FW)
    two = fp_encode(2.0, EW, FW)
    thr = fp_encode(3.0, EW, FW)

    return [
        ("minNorm * 0.5  -> sub", minN, half, fp_encode(fp_decode(minN, EW, FW) * 0.5, EW, FW)),
        ("minNorm * 0.25 -> sub", minN, qtr, fp_encode(fp_decode(minN, EW, FW) * 0.25, EW, FW)),
        ("minNorm * 0.125-> sub", minN, eigth, fp_encode(fp_decode(minN, EW, FW) * 0.125, EW, FW)),
        ("minNorm * 0.0625-> sub", minN, six, fp_encode(fp_decode(minN, EW, FW) * 0.0625, EW, FW)),
        ("minSub * 2 -> next sub", minS, two, fp_encode(fp_decode(minS, EW, FW) * 2.0, EW, FW)),
        ("minSub * 3 -> 3*minSub", minS, thr, fp_encode(fp_decode(minS, EW, FW) * 3.0, EW, FW)),
        ("maxSub * 1.0 -> maxSub", maxS, one, maxS),
        ("minSub * 1.0 -> minSub", minS, one, minS),
        ("maxSub * 0.5  (tie->even)", maxS, half, fp_encode(fp_decode(maxS, EW, FW) * 0.5, EW, FW)),
        ("minSub * minSub -> 0", minS, minS, bits_zero(EW, FW, 0)),
    ]


def build_fp_subnormal_ext_vectors(EW: int, FW: int):
    """Tie-to-even edge cases inside the subnormal range (multiply by 0.5)."""
    half = fp_encode(0.5, EW, FW)
    maxS = bits_max_sub(EW, FW)
    odd_vals = [1, 3, 5, 7, maxS - 2, maxS]
    even_vals = [maxS - 1, 2, 4]
    vecs = []
    for n in odd_vals:
        exp = fp_encode(fp_decode(n, EW, FW) * 0.5, EW, FW)
        vecs.append((f"0x{n:04x} * 0.5 (tie->even)", n, half, exp))
    for n in even_vals:
        exp = fp_encode(fp_decode(n, EW, FW) * 0.5, EW, FW)
        vecs.append((f"0x{n:04x} * 0.5 (below tie)", n, half, exp))
    return vecs


# ---------------------------------------------------------------------------
# Convenience wrappers for common formats
# ---------------------------------------------------------------------------


def build_f16_vectors():
    return build_fp_vectors(5, 10)


def build_f16_subnormal_vectors():
    return build_fp_subnormal_vectors(5, 10)


def build_f16_subnormal_ext_vectors():
    return build_fp_subnormal_ext_vectors(5, 10)


def build_bf16_vectors():
    return build_fp_vectors(8, 7)


def build_bf16_subnormal_vectors():
    return build_fp_subnormal_vectors(8, 7)


def build_bf16_subnormal_ext_vectors():
    return build_fp_subnormal_ext_vectors(8, 7)


def build_targeted_mul_vectors(EW: int, FW: int, subnormals: bool = True) -> TestVectors:
    """Collect all applicable targeted multiply vectors in the standard TestVectors format."""
    four_tuples = build_fp_vectors(EW, FW, subnormals=subnormals)
    if subnormals:
        four_tuples += build_fp_subnormal_vectors(EW, FW)
        four_tuples += build_fp_subnormal_ext_vectors(EW, FW)
    return [(name, {"a": a, "b": b}, {"y": y}) for name, a, b, y in four_tuples]
