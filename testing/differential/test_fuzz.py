"""Seeded random-circuit fuzz: simulator vs IEEE text evaluation vs AIGER round-trip.

Fixed seeds keep runs deterministic. The unsigned generator profile avoids the operators that are known-divergent at
the current baseline so it must pass; the full profile (mixed signedness) carries strict xfail marks naming the
owning issues and flips green as the emitter/AIGER fixes land.
"""
from __future__ import annotations

import random

import pytest

from .harness import diff_sim_vs_aiger, diff_sim_vs_verilog, format_mismatches, gen_random_module, random_vectors

N_CIRCUITS = 25
N_VECTORS = 40


def _run(compare, seed_base: int, signed_ok: bool):
    failures = []
    for k in range(N_CIRCUITS):
        seed = seed_base + k
        m, inputs = gen_random_module(seed, signed_ok=signed_ok)
        vectors = random_vectors(random.Random(seed ^ 0xBEEF), m, inputs, N_VECTORS)
        bad = compare(m, inputs, vectors)
        if bad:
            failures.append((seed, bad, inputs))
    assert not failures, "\n".join(
        f"seed {seed}: {format_mismatches(bad, inputs)}" for seed, bad, inputs in failures[:3]
    ) + f"\n({len(failures)}/{N_CIRCUITS} circuits diverge)"


def _cmp_verilog(m, inputs, vectors):
    bad, _text = diff_sim_vs_verilog(m, inputs, vectors)
    return bad


def test_fuzz_sim_vs_verilog_unsigned():
    _run(_cmp_verilog, seed_base=1000, signed_ok=False)


@pytest.mark.xfail(strict=True, reason="ISSUES 0.1/0.2 + ISSUES2 §1: emitted Verilog diverges on signed/nested shapes")
def test_fuzz_sim_vs_verilog_signed():
    _run(_cmp_verilog, seed_base=2000, signed_ok=True)


def test_fuzz_sim_vs_aiger_unsigned():
    _run(diff_sim_vs_aiger, seed_base=3000, signed_ok=False)


@pytest.mark.xfail(strict=True, reason="ISSUES 0.3-0.6: AIGER signedness handling (extension, multiplier dispatch, eq)")
def test_fuzz_sim_vs_aiger_signed():
    _run(diff_sim_vs_aiger, seed_base=4000, signed_ok=True)
