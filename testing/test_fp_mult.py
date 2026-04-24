"""Tests for FpMul (flush-to-zero multiplier).

Note: test_floating_point.py already covers FpMul f16 normal/subnormal vectors via
test_floating_point_mul_matches_general_vectors and test_floating_point_mul_matches_subnormal_vectors.
The tests here add BF16 coverage and serve as a standalone entry point for the multiplier.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(ROOT)

from spirehdl.arithmetic.floating_point.fp_encoding import fp_decode, fp_encode, fp_unpack
from spirehdl.arithmetic.floating_point.spire_hdl_float_mult import FpMul, build_fp_mul, run_vectors_aby
from spirehdl.spirehdl_simulator import Simulator
from spirehdl.arithmetic.floating_point.fp_mul_testvectors import (
    build_bf16_vectors,
    build_f16_vectors,
)


def test_f16_mul_vectors():
    mod = build_fp_mul("F16MulTest", EW=5, FW=10)
    passed = run_vectors_aby(mod, build_f16_vectors(), label="f16 mul",
                             decoder=lambda b: fp_decode(b, 5, 10))
    assert passed


def test_bf16_mul_vectors():
    mod = build_fp_mul("BF16MulTest", EW=8, FW=7)
    passed = run_vectors_aby(mod, build_bf16_vectors(), label="bf16 mul",
                             decoder=lambda b: fp_decode(b, 8, 7))
    assert passed


def _is_subnormal(bits: int, EW: int, FW: int) -> bool:
    e = (bits >> FW) & ((1 << EW) - 1)
    f = bits & ((1 << FW) - 1)
    return e == 0 and f != 0


def _rounds_subnormal_to_min_normal(product_val: float, exp: int, EW: int, FW: int) -> bool:
    """True when a subnormal product rounds up to min_normal.

    FpMul flushes pre-rounding, so it wrongly outputs zero in this case.
    FpMulSN(subnormals=False) handles it correctly.
    """
    min_normal = 2.0 ** (2 - (1 << (EW - 1)))  # 2^(1-bias)
    if abs(product_val) >= min_normal or product_val == 0.0:
        return False
    _, re, rf = fp_unpack(exp, EW, FW)
    return re == 1 and rf == 0  # exp rounded up to min_normal


def _run_random_mul(EW: int, FW: int, name: str, num_vectors: int = 10_000, seed: int = 42):
    """Random test for FpMul (flush-to-zero).

    Known exclusions:
    - Subnormal inputs: FpMul zeros the mantissa but not the exponent path → wrong results.
    - Subnormal products that round to min_normal: FpMul flushes pre-rounding (gives 0)
      instead of outputting min_normal. Use FpMulSN(subnormals=False) for correct FTZ.
    """
    W = 1 + EW + FW
    mul = FpMul(EW, FW)
    sim = Simulator(mul.to_module(name, with_clock=False, with_reset=False))
    rng = np.random.default_rng(seed)
    failures = []
    tested = 0
    while tested < num_vectors:
        a = int(rng.integers(0, 1 << W))
        b = int(rng.integers(0, 1 << W))
        if _is_subnormal(a, EW, FW) or _is_subnormal(b, EW, FW):
            continue  # subnormal inputs not supported by FpMul (FTZ-in is buggy)
        product_val = fp_decode(a, EW, FW) * fp_decode(b, EW, FW)
        exp = fp_encode(product_val, EW, FW, subnormals=False)
        if _rounds_subnormal_to_min_normal(product_val, exp, EW, FW):
            continue  # FpMul known limitation: pre-rounding FTZ flush
        tested += 1
        sim.set(mul.io.a, a)
        sim.set(mul.io.b, b)
        sim.eval()
        got = sim.get(mul.io.y)
        if got != exp:
            failures.append((a, b, exp, got))
    assert not failures, (
        f"{len(failures)}/{num_vectors} failures; first 5:\n"
        + "\n".join(f"  a={a:#06x} b={b:#06x} exp={e:#06x} got={g:#06x}"
                    for a, b, e, g in failures[:5])
    )


def test_f16_mul_random():
    _run_random_mul(5, 10, "F16MulRand")


def test_bf16_mul_random():
    _run_random_mul(8, 7, "BF16MulRand")


def test_e3f4_mul_random():
    _run_random_mul(3, 4, "E3F4MulRand")


if __name__ == "__main__":
    test_f16_mul_vectors()
    test_bf16_mul_vectors()
    test_f16_mul_random()
    test_bf16_mul_random()
    test_e3f4_mul_random()
