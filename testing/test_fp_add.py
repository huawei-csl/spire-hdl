"""Tests for FpAdd (floating-point adder).

Covers:
  - Hand-picked normal / subnormal / special-value vectors
  - Flush-to-zero mode (subnormals=False)
  - Random tests via FpAddTestVectors
  - Exhaustive tests for small EW/FW via FpAddTestVectorsExhaustive
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(ROOT)

from spirehdl.arithmetic.floating_point.fp_add_testvectors import (
    FpAddTestVectors,
    FpAddTestVectorsExhaustive,
)
from spirehdl.arithmetic.floating_point.fp_encoding import (
    bits_inf,
    bits_max_sub,
    bits_min_normal,
    bits_min_sub,
    bits_qnan,
    bits_zero,
    fp_decode,
    fp_encode,
)
from spirehdl.arithmetic.floating_point.spire_hdl_float_add import FpAdd, build_fp_add
from spirehdl.arithmetic.floating_point.spire_hdl_float_mult import run_vectors_aby
from spirehdl.helpers import run_vectors_on_simulator
from spirehdl.spirehdl_simulator import Simulator


# -- Hand-picked vectors ----------------------------------------------------

def build_add_vectors(EW: int, FW: int):
    one = fp_encode(1.0, EW, FW)
    two = fp_encode(2.0, EW, FW)
    thr = fp_encode(3.0, EW, FW)
    half = fp_encode(0.5, EW, FW)
    onept5 = fp_encode(1.5, EW, FW)
    onept25 = fp_encode(1.25, EW, FW)
    onept75 = fp_encode(1.75, EW, FW)
    two_pt_five = fp_encode(2.5, EW, FW)
    three_qtr = fp_encode(0.75, EW, FW)
    quarter = fp_encode(0.25, EW, FW)
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
        ("1.5+(-1.25)=0.25", onept5, neg_onept25, quarter),
        ("1.75+(-0.5)=1.25", onept75, neg_half, onept25),
        ("1.5+1.5=3.0", onept5, onept5, fp_encode(3.0, EW, FW)),
        ("2.5+1.5=4.0", two_pt_five, onept5, fp_encode(4.0, EW, FW)),
        ("minNorm+(-maxSub)=minSub", min_norm, neg_max_sub, bits_min_sub(EW, FW)),
        ("minNorm+(-minNorm)=+0", min_norm, neg_min_norm, pos0),
        ("-2+2=0", neg_two, two, pos0),
        ("(-0)+0=0", neg0, pos0, pos0),
        ("inf+(-inf)=nan", pinf, ninf, qnan),
        ("inf+1=inf", pinf, one, pinf),
    ]


def test_f16_adder_vectors():
    mod = build_fp_add("F16AddTest", EW=5, FW=10)
    vectors = build_add_vectors(5, 10)
    passed = run_vectors_aby(mod, vectors, label="f16 add")
    assert passed


def test_f16_adder_subnormal_vectors():
    """Explicit subnormal input/output cases with subnormals=True (default)."""
    EW, FW = 5, 10
    min_norm = bits_min_normal(EW, FW)
    min_sub = bits_min_sub(EW, FW)
    max_sub = bits_max_sub(EW, FW)
    neg_max_sub = max_sub | (1 << (EW + FW))
    neg_min_norm = min_norm | (1 << (EW + FW))
    pos0 = bits_zero(EW, FW, 0)

    vectors = [
        ("minSub+minSub=2*minSub", min_sub, min_sub,
         fp_encode(fp_decode(min_sub, EW, FW) * 2, EW, FW)),
        ("maxSub+minSub=minNorm", max_sub, min_sub, min_norm),
        ("minNorm+(-maxSub)=minSub", min_norm, neg_max_sub, min_sub),
        ("minNorm+(-minNorm)=+0", min_norm, neg_min_norm, pos0),
        ("maxSub+maxSub", max_sub, max_sub,
         fp_encode(fp_decode(max_sub, EW, FW) * 2, EW, FW)),
    ]

    mod = build_fp_add("F16SubnormalTest", EW=EW, FW=FW, subnormals=True)
    passed = run_vectors_aby(mod, vectors, label="f16 subnormal")
    assert passed


def test_f16_adder_flush_to_zero():
    """With subnormals=False, results in the subnormal range flush to +0."""
    EW, FW = 5, 10
    min_norm = bits_min_normal(EW, FW)
    max_sub = bits_max_sub(EW, FW)
    min_sub = bits_min_sub(EW, FW)
    neg_max_sub = max_sub | (1 << (EW + FW))
    pos0 = bits_zero(EW, FW, 0)

    vectors = [
        ("minNorm+(-maxSub) flushes to 0", min_norm, neg_max_sub, pos0),
        ("minSub+minSub flushes to 0", min_sub, min_sub, pos0),
        ("1+1=2 still works", fp_encode(1.0, EW, FW), fp_encode(1.0, EW, FW), fp_encode(2.0, EW, FW)),
    ]

    mod = build_fp_add("F16FtzTest", EW=EW, FW=FW, subnormals=False)
    passed = run_vectors_aby(mod, vectors, label="f16 flush-to-zero")
    assert passed


# -- Random tests using FpAddTestVectors ------------------------------------

def _run_random_add(EW: int, FW: int, name: str, subnormals: bool = True,
                    num_vectors: int = 10_000, seed: int = 42):
    adder = FpAdd(EW, FW, subnormals=subnormals)
    sim = Simulator(adder.to_module(name, with_clock=False, with_reset=False))
    vectors = FpAddTestVectors(
        EW=EW, FW=FW, num_vectors=num_vectors, subnormals=subnormals, seed=seed,
    ).generate()
    run_vectors_on_simulator(sim, vectors, use_signed=False, raise_on_fail=True, print_on_pass=False)


def test_f16_adder_random():
    _run_random_add(5, 10, "F16RandTest")


def test_f16_adder_random_ftz():
    _run_random_add(5, 10, "F16RandFtzTest", subnormals=False)


def test_bf16_adder_random():
    _run_random_add(8, 7, "BF16RandTest")


def test_bf16_adder_random_ftz():
    _run_random_add(8, 7, "BF16RandFtzTest", subnormals=False)


# -- Inline random test (illustrative, same as _run_random_add / run_vectors_on_simulator) --

def _run_random_add_inline(EW: int, FW: int, name: str, subnormals: bool = True,
                           num_vectors: int = 10_000, seed: int = 42):
    """Illustrative inline version — does the same as _run_random_add /
    run_vectors_on_simulator but with an explicit simulate-and-compare loop."""
    W = 1 + EW + FW
    adder = FpAdd(EW, FW, subnormals=subnormals)
    sim = Simulator(adder.to_module(name, with_clock=False, with_reset=False))
    rng = np.random.default_rng(seed)
    failures = []
    for _ in range(num_vectors):
        a = int(rng.integers(0, 1 << W))
        b = int(rng.integers(0, 1 << W))
        exp = fp_encode(fp_decode(a, EW, FW) + fp_decode(b, EW, FW), EW, FW, subnormals=subnormals)
        sim.set(adder.io.a, a)
        sim.set(adder.io.b, b)
        sim.eval()
        got = sim.get(adder.io.y)
        if got != exp:
            failures.append((a, b, exp, got))
    assert not failures, (
        f"{len(failures)}/{num_vectors} failures; first 5:\n"
        + "\n".join(f"  a={a:#06x} b={b:#06x} exp={e:#06x} got={g:#06x}"
                    for a, b, e, g in failures[:5])
    )


def test_f16_adder_random_inline():
    _run_random_add_inline(5, 10, "F16RandInline")


def test_f16_adder_random_ftz_inline():
    _run_random_add_inline(5, 10, "F16RandFtzInline", subnormals=False)


# -- Exhaustive tests using FpAddTestVectorsExhaustive ----------------------

def _run_exhaustive_add(EW: int, FW: int, name: str, subnormals: bool = True):
    """Exhaustive test for small formats (all input pairs)."""
    adder = FpAdd(EW, FW, subnormals=subnormals)
    sim = Simulator(adder.to_module(name, with_clock=False, with_reset=False))
    vectors = FpAddTestVectorsExhaustive(
        EW=EW, FW=FW, subnormals=subnormals,
    ).generate()
    run_vectors_on_simulator(sim, vectors, use_signed=False, raise_on_fail=True, print_on_pass=False)


def test_e1f2_add_sn_exhaustive():
    """Exhaustive test for degenerate EW=1 format (no normal values)."""
    _run_exhaustive_add(1, 2, "E1F2AddSN", subnormals=True)


def test_e1f2_add_ftz_exhaustive():
    _run_exhaustive_add(1, 2, "E1F2AddFTZ", subnormals=False)


def test_e1f3_add_sn_exhaustive():
    _run_exhaustive_add(1, 3, "E1F3AddSN", subnormals=True)


def test_e2f2_add_sn_exhaustive():
    _run_exhaustive_add(2, 2, "E2F2AddSN", subnormals=True)


def test_e2f2_add_ftz_exhaustive():
    _run_exhaustive_add(2, 2, "E2F2AddFTZ", subnormals=False)


def test_e2f3_add_sn_exhaustive():
    _run_exhaustive_add(2, 3, "E2F3AddSN", subnormals=True)


def test_e2f3_add_ftz_exhaustive():
    _run_exhaustive_add(2, 3, "E2F3AddFTZ", subnormals=False)


if __name__ == "__main__":
    test_f16_adder_vectors()
    test_f16_adder_subnormal_vectors()
    test_f16_adder_flush_to_zero()
    test_f16_adder_random()
    test_f16_adder_random_ftz()
    test_f16_adder_random_inline()
    test_f16_adder_random_ftz_inline()
