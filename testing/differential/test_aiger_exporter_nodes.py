"""AIGER exporter semantics on RAW Op2 nodes (charter §2 per-node evaluation).

The operator builders promote/pre-align mixed-sign operands, so operator-built IR no longer
exercises the exporter's own extension paths. These tests build raw ``Op2`` nodes directly
(as simplify rewrites or imported graphs might) and require the exporter to extend each
operand per its OWN signedness.
"""
import itertools

import pytest

from spire.expr import HDLType, Op2, SInt, UInt

from .harness import diff_sim_vs_aiger, fresh_netlist


def _raw_op_module(op, wa, sa, wb, sb, w_out, s_out, name):
    m = fresh_netlist(name)
    a = m.input(SInt(wa) if sa else UInt(wa), "a")
    b = m.input(SInt(wb) if sb else UInt(wb), "b")
    y = m.output(UInt(w_out), "y")
    y <<= Op2(a, b, op, HDLType(w_out, s_out))
    return m


CASES = [
    ("+", 4, True, 3, False, 5, True),    # 0.3: S4 + U3 — U3 must zero-extend
    ("-", 4, False, 4, True, 5, True),
    ("*", 2, True, 2, False, 4, True),    # 0.4: mixed mul must not use Baugh-Wooley
    ("*", 3, False, 3, True, 6, True),
    ("==", 4, True, 3, False, 1, False),  # 0.5: exact integer equality
    ("<", 4, True, 3, False, 1, False),   # 0.3-compare: S4 < U3
    ("<=", 3, False, 4, True, 1, False),
    (">", 4, True, 4, False, 1, False),
]


@pytest.mark.parametrize("op,wa,sa,wb,sb,w_out,s_out", CASES)
def test_raw_op2_aiger_matches_sim(op, wa, sa, wb, sb, w_out, s_out):
    m = _raw_op_module(op, wa, sa, wb, sb, w_out, s_out, f"raw_{op}_{wa}{sa}_{wb}{sb}".replace("<", "lt")
                       .replace(">", "gt").replace("=", "e").replace("+", "add").replace("-", "sub")
                       .replace("*", "mul"))
    vectors = list(itertools.product(range(1 << wa), range(1 << wb)))
    bad = diff_sim_vs_aiger(m, ["a", "b"], vectors)
    assert not bad, f"{op} ({wa},{sa})x({wb},{sb}): {len(bad)} mismatches; first: {bad[0]}"
