"""Tests for FpMulSN (subnormal-aware multiplier).

Covers:
  - Normal cases (same vectors as FpMul, both f16 and bf16)
  - Subnormal input/output cases
  - Tie-to-even edge cases inside the subnormal range
  - Flush-to-zero mode (subnormals=False)
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(ROOT)

import numpy as np

from spire.arithmetic.floating_point.fp_encoding import fp_decode, fp_encode
from spire.arithmetic.floating_point.fp_mul_testvectors import (
    FpMulTestVectors,
    FpMulTestVectorsExhaustive,
    _should_skip,
)
from spire.arithmetic.floating_point.spire_hdl_float_mult import run_vectors_aby
from spire.arithmetic.floating_point.spire_hdl_float_mult_sn import FpMulSN, build_fp_mul_sn
from spire.helpers import run_vectors_on_simulator
from spire.simulator import Simulator
from spire.arithmetic.floating_point.fp_mul_testvectors import (
    build_bf16_subnormal_ext_vectors,
    build_bf16_subnormal_vectors,
    build_bf16_vectors,
    build_f16_subnormal_ext_vectors,
    build_f16_subnormal_vectors,
    build_f16_vectors,
)


def _sn_normal_vectors(EW, FW):
    from spire.arithmetic.floating_point.fp_mul_testvectors import build_fp_vectors
    return build_fp_vectors(EW, FW, subnormals=True)


def test_f16_mul_sn_normal_vectors():
    mod = build_fp_mul_sn("F16MulSNTest", EW=5, FW=10, subnormals=True)
    passed = run_vectors_aby(mod, _sn_normal_vectors(5, 10), label="f16 mul_sn normal",
                             decoder=lambda b: fp_decode(b, 5, 10))
    assert passed


def test_bf16_mul_sn_normal_vectors():
    mod = build_fp_mul_sn("BF16MulSNTest", EW=8, FW=7, subnormals=True)
    passed = run_vectors_aby(mod, _sn_normal_vectors(8, 7), label="bf16 mul_sn normal",
                             decoder=lambda b: fp_decode(b, 8, 7))
    assert passed


def test_f16_mul_sn_subnormal_vectors():
    mod = build_fp_mul_sn("F16MulSNSubTest", EW=5, FW=10, subnormals=True)
    passed = run_vectors_aby(mod, build_f16_subnormal_vectors(), label="f16 mul_sn subnormal",
                             decoder=lambda b: fp_decode(b, 5, 10))
    assert passed


def test_bf16_mul_sn_subnormal_vectors():
    mod = build_fp_mul_sn("BF16MulSNSubTest", EW=8, FW=7, subnormals=True)
    passed = run_vectors_aby(mod, build_bf16_subnormal_vectors(), label="bf16 mul_sn subnormal",
                             decoder=lambda b: fp_decode(b, 8, 7))
    assert passed


def test_f16_mul_sn_subnormal_ext():
    """Tie-to-even edge cases inside the f16 subnormal range."""
    mod = build_fp_mul_sn("F16MulSNExtTest", EW=5, FW=10, subnormals=True)
    passed = run_vectors_aby(mod, build_f16_subnormal_ext_vectors(), label="f16 mul_sn subnormal ext",
                             decoder=lambda b: fp_decode(b, 5, 10))
    assert passed


def test_bf16_mul_sn_subnormal_ext():
    """Tie-to-even edge cases inside the bf16 subnormal range."""
    mod = build_fp_mul_sn("BF16MulSNExtTest", EW=8, FW=7, subnormals=True)
    passed = run_vectors_aby(mod, build_bf16_subnormal_ext_vectors(), label="bf16 mul_sn subnormal ext",
                             decoder=lambda b: fp_decode(b, 8, 7))
    assert passed


def test_f16_mul_sn_ftz():
    """With subnormals=False, subnormal results flush to zero (same as FpMul)."""
    mod = build_fp_mul_sn("F16MulSNFtzTest", EW=5, FW=10, subnormals=False)
    passed = run_vectors_aby(mod, build_f16_vectors(), label="f16 mul_sn ftz",
                             decoder=lambda b: fp_decode(b, 5, 10))
    assert passed


def test_bf16_mul_sn_ftz():
    mod = build_fp_mul_sn("BF16MulSNFtzTest", EW=8, FW=7, subnormals=False)
    passed = run_vectors_aby(mod, build_bf16_vectors(), label="bf16 mul_sn ftz",
                             decoder=lambda b: fp_decode(b, 8, 7))
    assert passed


def _run_random_mul_sn(EW: int, FW: int, name: str, *, subnormals: bool,
                       num_vectors: int = 10_000, seed: int = 42,
                       always_subnormal_rounding: bool = False):
    mul = FpMulSN(EW, FW, subnormals=subnormals, always_subnormal_rounding=always_subnormal_rounding)
    sim = Simulator(mul.to_module(name, with_clock=False, with_reset=False))
    vectors = FpMulTestVectors(
        EW=EW, FW=FW, num_vectors=num_vectors, subnormals=subnormals,
        always_subnormal_rounding=always_subnormal_rounding, seed=seed,
    ).generate()
    fails = run_vectors_on_simulator(sim, vectors, use_signed=False, raise_on_fail=True, print_on_pass=False)


def test_f16_mul_sn_random():
    _run_random_mul_sn(5, 10, "F16MulSNRand", subnormals=True)


def test_bf16_mul_sn_random():
    _run_random_mul_sn(8, 7, "BF16MulSNRand", subnormals=True)


def test_f16_mul_sn_random_ftz():
    _run_random_mul_sn(5, 10, "F16MulSNRandFtz", subnormals=False)


def test_bf16_mul_sn_random_ftz():
    _run_random_mul_sn(8, 7, "BF16MulSNRandFtz", subnormals=False)


def test_custom_34_mul_random_ftz():
    _run_random_mul_sn(3, 4, "BF16MulSNRandFtz", subnormals=False, always_subnormal_rounding=False)
    
def test_custom_34_mul_random_sn_round():
    _run_random_mul_sn(3, 4, "BF16MulSNRandFtz", subnormals=False, always_subnormal_rounding=True)
    
def test_custom_34_mul_random_sn():
    _run_random_mul_sn(3, 4, "BF16MulSNRandFtz", subnormals=True)


def _run_random_mul_sn_inline(EW: int, FW: int, name: str, subnormals: bool,
                              num_vectors: int = 10_000, seed: int = 42,
                              always_subnormal_rounding: bool = False):
    """Illustrative inline version — does the same as _run_random_mul_sn /
    run_vectors_on_simulator but with an explicit simulate-and-compare loop."""
    W = 1 + EW + FW
    mul = FpMulSN(EW, FW, subnormals=subnormals, always_subnormal_rounding=always_subnormal_rounding)
    sim = Simulator(mul.to_module(name, with_clock=False, with_reset=False))
    rng = np.random.default_rng(seed)
    failures = []
    tested = 0
    while tested < num_vectors:
        a = int(rng.integers(0, 1 << W))
        b = int(rng.integers(0, 1 << W))
        product_val = fp_decode(a, EW, FW) * fp_decode(b, EW, FW)
        exp = fp_encode(product_val, EW, FW, subnormals=subnormals)
        if _should_skip(a, b, product_val, exp, EW, FW, subnormals, always_subnormal_rounding):
            continue
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


def test_f16_mul_sn_random_inline():
    _run_random_mul_sn_inline(5, 10, "F16MulSNRandInline", subnormals=True)


def test_bf16_mul_sn_random_ftz_inline():
    _run_random_mul_sn_inline(8, 7, "BF16MulSNRandFtzInline", subnormals=False)


def _run_random_mul_sn_asr(EW: int, FW: int, name: str, num_vectors: int = 10_000, seed: int = 42):
    """Random FTZ test with always_subnormal_rounding=True."""
    _run_random_mul_sn(EW, FW, name, subnormals=False,
                       always_subnormal_rounding=True,
                       num_vectors=num_vectors, seed=seed)


def test_f16_mul_sn_random_ftz_asr():
    _run_random_mul_sn_asr(5, 10, "F16MulSNRandFtzAsr")


def test_bf16_mul_sn_random_ftz_asr():
    _run_random_mul_sn_asr(8, 7, "BF16MulSNRandFtzAsr")


def test_e3f4_mul_sn_ftz_asr_boundary():
    """Fixed vectors confirming the subnormal→min_normal boundary case.

    Without always_subnormal_rounding the hardware flushes these to zero;
    with it, they correctly round up to min_normal.
    Only includes cases where the first-round mantissa is exact (guard=0),
    avoiding the double-rounding ambiguity that affects random vectors in
    small formats.
    """
    mod = build_fp_mul_sn("E3F4MulSNFtzAsrBoundary", EW=3, FW=4,
                          subnormals=False, always_subnormal_rounding=True)
    vectors = [
        # product = 0.244140625 → 15.625 sub-ULPs → rounds to 16 = +min_normal
        ("sub→min_normal (pos)", 0x0024, 0x0019, 0x0010),
        # same magnitude, negative
        ("sub→min_normal (neg)", 0x0024, 0x0099, 0x0090),
    ]
    passed = run_vectors_aby(mod, vectors, label="e3f4 ftz asr boundary",
                             decoder=lambda b: fp_decode(b, 3, 4))
    assert passed


def _run_exhaustive_mul_sn(EW: int, FW: int, name: str, *, subnormals: bool,
                           always_subnormal_rounding: bool = False):
    """Exhaustive test for small formats (all input pairs)."""
    mul = FpMulSN(EW, FW, subnormals=subnormals, always_subnormal_rounding=always_subnormal_rounding)
    sim = Simulator(mul.to_module(name, with_clock=False, with_reset=False))
    vectors = FpMulTestVectorsExhaustive(
        EW=EW, FW=FW, subnormals=subnormals,
        always_subnormal_rounding=always_subnormal_rounding,
    ).generate()
    fails = run_vectors_on_simulator(sim, vectors, use_signed=False, raise_on_fail=True, print_on_pass=False)


def test_e1f2_mul_sn_exhaustive():
    """Exhaustive test for the degenerate EW=1 format (no normal values)."""
    _run_exhaustive_mul_sn(1, 2, "E1F2MulSN", subnormals=True)


def test_e1f2_mul_ftz_exhaustive():
    """FTZ exhaustive test for EW=1 (skips subnormal inputs)."""
    _run_exhaustive_mul_sn(1, 2, "E1F2MulFTZ", subnormals=False)


def test_e1f2_mul_ftz_asr_exhaustive():
    """FTZ+ASR exhaustive test for EW=1 (skips subnormal inputs)."""
    _run_exhaustive_mul_sn(1, 2, "E1F2MulFTZASR", subnormals=False, always_subnormal_rounding=True)


def test_e2f3_mul_sn_exhaustive():
    """Exhaustive test for EW=2, FW=3 (small format with normals)."""
    _run_exhaustive_mul_sn(2, 3, "E2F3MulSN", subnormals=True)


def test_e2f3_mul_ftz_exhaustive():
    _run_exhaustive_mul_sn(2, 3, "E2F3MulFTZ", subnormals=False)


def test_e2f3_mul_ftz_asr_exhaustive():
    _run_exhaustive_mul_sn(2, 3, "E2F3MulFTZASR", subnormals=False, always_subnormal_rounding=True)


if __name__ == "__main__":
    test_f16_mul_sn_normal_vectors()
    test_bf16_mul_sn_normal_vectors()
    test_f16_mul_sn_subnormal_vectors()
    test_bf16_mul_sn_subnormal_vectors()
    test_f16_mul_sn_subnormal_ext()
    test_bf16_mul_sn_subnormal_ext()
    test_f16_mul_sn_ftz()
    test_bf16_mul_sn_ftz()
    test_f16_mul_sn_random()
    test_bf16_mul_sn_random()
    test_f16_mul_sn_random_ftz()
    test_bf16_mul_sn_random_ftz()
