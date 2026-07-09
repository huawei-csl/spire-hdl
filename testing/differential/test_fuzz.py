"""Seeded random-circuit fuzz: simulator vs IEEE text evaluation vs AIGER round-trip.

Fixed seeds keep runs deterministic. Both signedness profiles are conformance guarantees across all three backends;
known exporter defects too narrow for the generator get pinned as dedicated strict-xfail shapes below.
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


def test_fuzz_sim_vs_verilog_signed():
    _run(_cmp_verilog, seed_base=2000, signed_ok=True)


def test_fuzz_sim_vs_aiger_unsigned():
    _run(diff_sim_vs_aiger, seed_base=3000, signed_ok=False)


def test_fuzz_sim_vs_aiger_signed():
    # Conformant since the operator builders normalize mixed signedness (promoted compares, pre-materialized
    # extensions): the AIGER exporter's own mixed-sign paths are no longer exercised by operator-built IR.
    _run(diff_sim_vs_aiger, seed_base=4000, signed_ok=True)


def test_aiger_mux_selector_multibit():
    # The generator above only builds 1-bit selectors (mux(a < b, ...)), so this shape guards the
    # exporter's OR-reduced multi-bit selector.
    import itertools

    from spire.expr import UInt, mux

    from .harness import diff_sim_vs_aiger, fresh_netlist

    m = fresh_netlist("mux_sel3")
    s = m.input(UInt(3), "s")
    c = m.input(UInt(4), "c")
    d = m.input(UInt(4), "d")
    y = m.output(UInt(4), "y")
    y <<= mux(s, c, d)
    vectors = list(itertools.product(range(8), range(16), range(16)))
    bad = diff_sim_vs_aiger(m, ["s", "c", "d"], vectors)
    assert not bad, f"{len(bad)} mismatches; first: {bad[0]}"
