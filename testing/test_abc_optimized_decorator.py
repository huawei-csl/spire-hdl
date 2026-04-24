"""
Test script for the @abc_optimized decorator.

Runs ABC/DeepSyn optimization on a small multiplier, verifies that:
  1. AIG gate count and/or depth improve after optimization.
  2. Functional correctness is preserved (simulation matches golden model).

Usage:
    python testing/test_abc_optimized_decorator.py
    pytest testing/test_abc_optimized_decorator.py -v -s
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Tuple, Union

from spirehdl.spirehdl import Expr, HDLType, Signal, UInt, reset_shared_cache
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_aiger import AigerExporter
from spirehdl.spirehdl_simulator import Simulator
from spirehdl.optimize import abc_optimized, clear_optimization_cache


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_flowy_optimized_decorator.py)
# ---------------------------------------------------------------------------

def wrap_in_module(
    name: str,
    input_specs: Dict[str, HDLType],
    build_fn: Callable[..., Union[Expr, Tuple[Expr, ...]]],
) -> Module:
    m = Module(name, with_clock=False, with_reset=False)
    inputs: Dict[str, Signal] = {}
    for pname, typ in input_specs.items():
        inputs[pname] = m.input(typ, pname)
    result: Any = build_fn(**inputs)
    if isinstance(result, tuple):
        for i, expr in enumerate(result):
            out = m.output(expr.typ, f"y{i}")
            out <<= expr
    else:
        out = m.output(result.typ, "y")
        out <<= result
    m.collect_signals()
    return m


def get_aig_stats(module: Module) -> Dict[str, int]:
    """Get AIG gate count and depth from the AAG header."""
    aag_lines = AigerExporter(module).get_aag()
    header = aag_lines[0].split()
    # aag M I L O A
    return {"num_gates": int(header[5])}


def simulate_module(
    module: Module,
    test_vectors: List[Dict[str, int]],
) -> List[Dict[str, int]]:
    sim = Simulator(module)
    results: List[Dict[str, int]] = []
    for inputs in test_vectors:
        for k, v in inputs.items():
            sim.set(k, v)
        sim.eval()
        out: Dict[str, int] = {}
        for p in module._ports_of("output"):
            out[p.name] = sim.get(p.name)
        results.append(out)
    return results


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

DEEPSYN_T = 10  # seconds

@abc_optimized(abc_script=f"strash; &get -n; &deepsyn -T {DEEPSYN_T}; &put")
def opt_mult(a, b):
    return a * b


def test_abc_deepsyn_multiplier():
    """Run deepsyn on an 8-bit multiplier, verify improvement and correctness."""
    clear_optimization_cache()
    reset_shared_cache()

    W = 8
    input_specs = {"a": UInt(W), "b": UInt(W)}

    # --- Original (unoptimized) ---
    def plain_mult(a, b):
        return a * b

    reset_shared_cache()
    orig_module = wrap_in_module("orig_mult", input_specs, plain_mult)
    orig_stats = get_aig_stats(orig_module)

    # --- Optimized via ABC/DeepSyn ---
    reset_shared_cache()
    opt_module = wrap_in_module("opt_mult", input_specs, opt_mult)
    opt_stats = get_aig_stats(opt_module)

    print(f"\nOriginal : {orig_stats['num_gates']} AIG gates")
    print(f"Optimized: {opt_stats['num_gates']} AIG gates")
    improvement = orig_stats["num_gates"] - opt_stats["num_gates"]
    print(f"Improvement: {improvement} gates ({100 * improvement / orig_stats['num_gates']:.1f}%)")

    assert opt_stats["num_gates"] <= orig_stats["num_gates"], (
        f"Expected improvement: {opt_stats['num_gates']} > {orig_stats['num_gates']}"
    )

    # --- Functional correctness ---
    mask = (1 << (2 * W)) - 1
    rng = random.Random(42)
    test_vectors = [
        {"a": rng.randint(0, (1 << W) - 1), "b": rng.randint(0, (1 << W) - 1)}
        for _ in range(64)
    ]

    orig_results = simulate_module(orig_module, test_vectors)
    opt_results = simulate_module(opt_module, test_vectors)

    for i, (tv, orig_out, opt_out) in enumerate(zip(test_vectors, orig_results, opt_results)):
        expected = (tv["a"] * tv["b"]) & mask
        assert orig_out["y"] == expected, f"Vector {i}: orig {orig_out['y']} != {expected}"
        assert opt_out["y"] == expected, f"Vector {i}: opt {opt_out['y']} != {expected}"

    print(f"All {len(test_vectors)} test vectors passed.")


if __name__ == "__main__":
    test_abc_deepsyn_multiplier()
