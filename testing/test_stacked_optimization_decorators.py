"""Test for stacking ``@abc_optimized`` on top of ``@arithmetic_optimized``.

Verifies, on an 8x8 -> 17-bit MAC ``y = a*b + c``, that:

  1. Each decorator produces a functionally correct circuit.
  2. Stacking both decorators yields a smaller AIG than either applied alone.

Note: ``abc_optimize`` is nondeterministic within a single Python process
(yosys/pyosys state survives ``design -reset`` and shifts ABC's output between
cold and warm processes).  The test only checks *inequalities* between the
three gate counts, which hold under both cold and warm yosys states.

Run:
    pytest testing/test_stacked_optimization_decorators.py -v -s
    python testing/test_stacked_optimization_decorators.py
"""
from __future__ import annotations

import random

from spire.expr import UInt, reset_shared_cache
from spire.component import Module
from spire.aiger import AigerExporter
from spire.simulator import Simulator
from spire.optimize import (
    abc_optimized,
    arithmetic_optimized,
    clear_optimization_cache,
)


ABC_SCRIPT = "strash; balance; rewrite -l; refactor -l; balance"


@abc_optimized(abc_script=ABC_SCRIPT, cache_read="mem", cache_write="mem")
@arithmetic_optimized(objective="area")
def stacked_mac(a, b, c):
    return a * b + c


@arithmetic_optimized(objective="area")
def arith_only_mac(a, b, c):
    return a * b + c


@abc_optimized(abc_script=ABC_SCRIPT, cache_read="mem", cache_write="mem")
def abc_only_mac(a, b, c):
    return a * b + c


def _build_module(fn, name: str) -> Module:
    reset_shared_cache()
    m = Module(name, with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    c = m.input(UInt(16), "c")
    y = m.output(UInt(17), "y")
    y <<= fn(a, b, c)
    m.collect_signals()
    return m


def _gate_count(m: Module) -> int:
    return int(AigerExporter(m).get_aag()[0].split()[5])


def _check_functional(m: Module, n_vectors: int = 20) -> None:
    rng = random.Random(0)
    mask = 0x1FFFF
    sim = Simulator(m)
    for _ in range(n_vectors):
        a, b, c = rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 65535)
        sim.set("a", a)
        sim.set("b", b)
        sim.set("c", c)
        sim.eval()
        expected = (a * b + c) & mask
        got = sim.get("y")
        assert got == expected, f"sim mismatch: a={a} b={b} c={c} got={got} exp={expected}"


def test_stacked_decorators_beat_each_alone():
    clear_optimization_cache()

    m_arith = _build_module(arith_only_mac, "arith_only")
    m_abc = _build_module(abc_only_mac, "abc_only")
    m_stacked = _build_module(stacked_mac, "stacked")

    g_arith = _gate_count(m_arith)
    g_abc = _gate_count(m_abc)
    g_stacked = _gate_count(m_stacked)

    print(f"\narith   : {g_arith} gates")
    print(f"abc     : {g_abc} gates")
    print(f"stacked : {g_stacked} gates")

    _check_functional(m_arith)
    _check_functional(m_abc)
    _check_functional(m_stacked)

    assert g_stacked < g_arith, f"expected stacked ({g_stacked}) < arith_only ({g_arith})"
    assert g_stacked < g_abc, f"expected stacked ({g_stacked}) < abc_only ({g_abc})"


if __name__ == "__main__":
    test_stacked_decorators_beat_each_alone()
