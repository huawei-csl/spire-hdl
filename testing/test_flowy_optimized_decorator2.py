#!/usr/bin/env python3
"""Simple test: @flowy_optimized decorator with direct execution (no Docker).

Builds a 4-bit multiplier with and without the decorator, compares AIG stats.
"""

from spire.component import Module
from spire.expr import UInt, Const, mux
from spire.aiger import AigerExporter
from spire.optimize import flowy_optimized


def aig_gate_count(m: Module) -> int:
    """Return the AND-gate count from the AAG header."""
    m.collect_signals()
    header = AigerExporter(m).get_aag()[0].split()
    return int(header[5])  # A (AND gates)


# ── Original (no optimization) ──────────────────────────────────────────────


def build_original():
    m = Module("mult_orig", with_clock=False, with_reset=False)
    a = m.input(UInt(4), "a")
    b = m.input(UInt(4), "b")
    p = m.output(UInt(8), "p")
    p <<= a * b
    return m


m_orig = build_original()
gates_orig = aig_gate_count(m_orig)
print(f"Original:  {gates_orig} AIG gates")


# ── With @flowy_optimized ────────────────────────────────────────────────────


@flowy_optimized(direct=True, iterations=1, mockturtle_chains=1, mockturtle_chain_len=2, mockturtle_chain_workers=1)
def optimized_mult(a, b):
    return a * b


def build_optimized():
    m = Module("mult_opt", with_clock=False, with_reset=False)
    a = m.input(UInt(4), "a")
    b = m.input(UInt(4), "b")
    p = m.output(UInt(8), "p")
    p <<= optimized_mult(a, b)
    return m


m_opt = build_optimized()
gates_opt = aig_gate_count(m_opt)
print(f"Optimized: {gates_opt} AIG gates")


# ── Summary ──────────────────────────────────────────────────────────────────

delta = gates_orig - gates_opt
pct = delta / gates_orig * 100 if gates_orig else 0
print(f"Reduction: {delta} gates ({pct:.1f}%)")


# ── Multi-input with 1-bit signal (regression test for from_module fix) ──────

PROD_W = 22
FW = 10
LZ_W = 5


@flowy_optimized(direct=True, iterations=1, mockturtle_chains=1, mockturtle_chain_len=3, mockturtle_chain_workers=1)
def compute_guard_sticky_normal(prod, sr, need_right):
    """Compute guard and sticky bits for normal rounding."""
    pref_bits = [prod[0]]
    for i in range(1, PROD_W):
        pref_bits.append(pref_bits[-1] | prod[i])

    guard_mux = Const(0, UInt(1))
    sticky_mux = Const(0, UInt(1))
    for k in range(FW + 1):
        guard_mux = mux(sr == Const(k + 1, UInt(LZ_W)), prod[k], guard_mux)
    for k in range(FW):
        sticky_mux = mux(sr == Const(k + 2, UInt(LZ_W)), pref_bits[k], sticky_mux)
    guard_w = mux(need_right, guard_mux, Const(0, UInt(1)))
    sticky_w = mux(need_right, sticky_mux, Const(0, UInt(1)))
    return guard_w, sticky_w


def build_guard_sticky_orig():
    m = Module("gs_orig", with_clock=False, with_reset=False)
    prod = m.input(UInt(PROD_W), "prod")
    sr = m.input(UInt(LZ_W), "sr")
    nr = m.input(UInt(1), "need_right")
    g = m.output(UInt(1), "g")
    s = m.output(UInt(1), "s")

    pref_bits = [prod[0]]
    for i in range(1, PROD_W):
        pref_bits.append(pref_bits[-1] | prod[i])
    guard_mux = Const(0, UInt(1))
    sticky_mux = Const(0, UInt(1))
    for k in range(FW + 1):
        guard_mux = mux(sr == Const(k + 1, UInt(LZ_W)), prod[k], guard_mux)
    for k in range(FW):
        sticky_mux = mux(sr == Const(k + 2, UInt(LZ_W)), pref_bits[k], sticky_mux)
    g <<= mux(nr, guard_mux, Const(0, UInt(1)))
    s <<= mux(nr, sticky_mux, Const(0, UInt(1)))
    return m


print("\n-- Guard/sticky (multi-input + 1-bit signal) --")
m_gs_orig = build_guard_sticky_orig()
gates_gs_orig = aig_gate_count(m_gs_orig)
print(f"Original:  {gates_gs_orig} AIG gates")


def build_guard_sticky_opt():
    m = Module("gs_opt", with_clock=False, with_reset=False)
    prod = m.input(UInt(PROD_W), "prod")
    sr = m.input(UInt(LZ_W), "sr")
    nr = m.input(UInt(1), "need_right")
    g = m.output(UInt(1), "g")
    s = m.output(UInt(1), "s")
    gv, sv = compute_guard_sticky_normal(prod, sr, nr)
    g <<= gv
    s <<= sv
    return m


m_gs_opt = build_guard_sticky_opt()
gates_gs_opt = aig_gate_count(m_gs_opt)
print(f"Optimized: {gates_gs_opt} AIG gates")

delta2 = gates_gs_orig - gates_gs_opt
pct2 = delta2 / gates_gs_orig * 100 if gates_gs_orig else 0
print(f"Reduction: {delta2} gates ({pct2:.1f}%)")
