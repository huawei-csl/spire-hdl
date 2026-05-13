"""Nested ``optimized_encoding`` + ``optimized_fsm`` (case10).

Composes both wrappers per the proposal's recommended pattern.
Inner ``__exit__`` (``optimized_fsm``) runs first → Hopcroft shrinks the
state set; outer ``__exit__`` (``optimized_encoding``) then searches
bit-assignments over the survivors.

The default cost oracle uses in-process pyosys + aigverse, so both the
synthetic-cost and real-synth variants run unconditionally (no `yosys`
binary required).
"""
from __future__ import annotations

import pytest

from spirehdl.optimize.fsm._emit import restore_encoding
from spirehdl.spirehdl import Bool, UInt
from spirehdl.spirehdl_control_structures import case_, default, if_, else_, switch_
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_state import (
    Encoding, State, optimized_encoding, optimized_fsm, state,
)


class Case10(State, encoding=Encoding.BINARY):
    S0 = state(); S1 = state(); S2 = state()
    S3 = state(); S4 = state(); S5 = state(); S6 = state()


@pytest.fixture(autouse=True)
def _restore_case10():
    restore_encoding(Case10, {"S0": 0, "S1": 1, "S2": 2, "S3": 3,
                              "S4": 4, "S5": 5, "S6": 6})


def _build_case10_body(reg, out, x):
    out <<= 0
    with switch_(reg):
        with case_(Case10.S0):
            out <<= 1
            with if_(x): reg <<= Case10.S2
            with else_(): reg <<= Case10.S1
        with case_(Case10.S1):
            out <<= 1
            with if_(x): reg <<= Case10.S5
            with else_(): reg <<= Case10.S3
        with case_(Case10.S2):
            out <<= 0
            with if_(x): reg <<= Case10.S4
            with else_(): reg <<= Case10.S5
        with case_(Case10.S3):
            out <<= 1
            with if_(x): reg <<= Case10.S6
            with else_(): reg <<= Case10.S1
        with case_(Case10.S4):
            out <<= 0
            with if_(x): reg <<= Case10.S2
            with else_(): reg <<= Case10.S5
        with case_(Case10.S5):
            out <<= 0
            with if_(x): reg <<= Case10.S3
            with else_(): reg <<= Case10.S4
        with case_(Case10.S6):
            out <<= 0
            with if_(x): reg <<= Case10.S6
            with else_(): reg <<= Case10.S5
        with default():
            reg <<= Case10.S0


def test_nested_wrappers_minimize_then_search_synthetic():
    """With a synthetic cost_fn that prefers a specific class-representative
    layout, the nested wrappers must (a) reduce to 4 classes, then (b)
    rewrite the representatives onto the preferred values.
    """
    m = Module("case10", with_clock=True, with_reset=False)
    x = m.input(Bool(), "x")
    out = m.output(UInt(1), "out")
    reg = m.reg(Case10.typ, "state_reg", init=Case10.S0)

    # The outer encoding search runs AFTER Hopcroft has merged the state set.
    # After minimisation, only 4 distinct values remain in Case10._values, but
    # all 7 names still exist (S3 shares S0's value, S4/S6 share S2's, etc.).
    # The cost_fn observes the post-merge assignment and rewards specific
    # representative values.
    preferred_class_values = (3, 1, 2, 0)   # for class reps of {S0,S3}, {S1}, {S2,S4,S6}, {S5}

    def cost_fn(assignment):
        # Each class's representative is whatever the merge picked
        # (smallest in class, deterministic): S0 for {S0,S3}; S1 for {S1};
        # S2 for {S2,S4,S6}; S5 for {S5}. We score by how close the rep
        # values are to a target.
        reps = (assignment["S0"], assignment["S1"], assignment["S2"], assignment["S5"])
        return float(sum(abs(a - b) for a, b in zip(reps, preferred_class_values)))

    with optimized_encoding(Case10, module=m, search="exhaustive", cost_fn=cost_fn):
        with optimized_fsm(reg, module=m, minimize=True, outputs=[out]):
            _build_case10_body(reg, out, x)

    v = Case10._values
    # Hopcroft still merged the same classes (S0=S3, S2=S4=S6).
    assert v["S0"] == v["S3"]
    assert v["S2"] == v["S4"] == v["S6"]
    # The encoding search picked the preferred class-representative values.
    assert v["S0"] == 3      # rep of {S0,S3}
    assert v["S1"] == 1
    assert v["S2"] == 2      # rep of {S2,S4,S6}
    assert v["S5"] == 0


def test_nested_wrappers_case10_real_synth():
    """End-to-end with the real (pyosys-backed) cost oracle. We don't insist
    on a specific cell count — just that the nested wrappers produce a valid
    synthesisable module and don't regress on the original. Cells after the
    pipeline must be <= cells before (i.e. optimisation is non-negative)."""
    from spirehdl.optimize.fsm._cost_oracle import _measure

    # Baseline: build without wrappers, synth, record cells.
    m_base = Module("case10_base", with_clock=True, with_reset=False)
    x_b = m_base.input(Bool(), "x")
    out_b = m_base.output(UInt(1), "out")
    reg_b = m_base.reg(Case10.typ, "state_reg", init=Case10.S0)
    _build_case10_body(reg_b, out_b, x_b)
    cells_base = _measure(m_base, "cells")

    # Reset Case10's encoding before building the optimised version.
    restore_encoding(Case10, {"S0": 0, "S1": 1, "S2": 2, "S3": 3,
                              "S4": 4, "S5": 5, "S6": 6})

    # Optimised: nested wrappers around a fresh build.
    m_opt = Module("case10_opt", with_clock=True, with_reset=False)
    x_o = m_opt.input(Bool(), "x")
    out_o = m_opt.output(UInt(1), "out")
    reg_o = m_opt.reg(Case10.typ, "state_reg", init=Case10.S0)
    with optimized_encoding(Case10, module=m_opt, search="exhaustive"):
        with optimized_fsm(reg_o, module=m_opt, minimize=True, outputs=[out_o]):
            _build_case10_body(reg_o, out_o, x_o)
    cells_opt = _measure(m_opt, "cells")

    # The wrapper must not regress — and should strictly improve in practice
    # (case10 was the motivating benchmark).
    assert cells_opt != float("inf"), f"synth of optimised module failed"
    assert cells_opt <= cells_base, (
        f"optimisation regressed: baseline {cells_base} → wrapped {cells_opt}")
