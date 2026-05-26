"""Transition-table extraction (Step 6)."""
from __future__ import annotations

import pytest

from spirehdl.optimize.fsm._emit import restore_encoding
from spirehdl.optimize.fsm._table import (
    MAX_INPUT_COMBINATIONS, TooLargeForExhaustiveExtraction, extract_transition_table,
)
from spirehdl.spirehdl import Bool, UInt
from spirehdl.spirehdl_control_structures import case_, default, if_, else_, switch_
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_state import Encoding, State, state


# ---------------------------------------------------------------------------
# case10 — the 7-state FSM used as the canonical validation target
# ---------------------------------------------------------------------------

class Case10(State, encoding=Encoding.BINARY):
    S0 = state(); S1 = state(); S2 = state()
    S3 = state(); S4 = state(); S5 = state(); S6 = state()


@pytest.fixture(autouse=True)
def _restore_case10():
    """Keep Case10's encoding stable across tests in this file."""
    yield
    restore_encoding(Case10, {"S0": 0, "S1": 1, "S2": 2, "S3": 3,
                              "S4": 4, "S5": 5, "S6": 6})


def _build_case10():
    """Build the case10 FSM exactly as in the RTLRewriter benchmark.
    Returns (module, reg, out, x) for downstream tests."""
    m = Module("case10", with_clock=True, with_reset=False)
    x = m.input(Bool(), "x")
    out = m.output(UInt(1), "out")
    reg = m.reg(Case10.typ, "reg", init=Case10.S0)
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
    return m, reg, out, x


def test_extract_case10_transition_table():
    m, reg, out, x = _build_case10()
    table = extract_transition_table(reg, Case10, outputs=[out])

    # 7 states × 2 input combinations = 14 (state, input) entries.
    assert sorted(table.state_values) == [0, 1, 2, 3, 4, 5, 6]
    combos = table.all_input_combos()
    assert combos == [(0,), (1,)]

    # Hand-check a few transitions vs the canonical FSM definition.
    assert table.transitions[0][(0,)] == 1     # S0, x=0 → S1
    assert table.transitions[0][(1,)] == 2     # S0, x=1 → S2
    assert table.transitions[6][(1,)] == 6     # S6, x=1 → S6 (self-loop)
    assert table.transitions[5][(0,)] == 4     # S5, x=0 → S4

    # Moore output per state — case10's spec: S0/S1/S3 output 1, others 0.
    assert table.outputs[0][(0,)] == (1,)
    assert table.outputs[0][(1,)] == (1,)
    assert table.outputs[2][(0,)] == (0,)
    assert table.outputs[2][(1,)] == (0,)
    assert table.outputs[3][(0,)] == (1,)


def test_extract_with_no_outputs():
    m, reg, _out, _x = _build_case10()
    table = extract_transition_table(reg, Case10, outputs=[])
    # Transitions still populated; outputs is empty per state.
    assert table.transitions[0][(1,)] == 2
    assert table.outputs[0][(1,)] == ()


def test_extract_raises_when_input_domain_too_large():
    m = Module("t", with_clock=True, with_reset=False)
    big = m.input(UInt(8), "big_input")
    big2 = m.input(UInt(10), "big_input2")
    reg = m.reg(Case10.typ, "reg", init=Case10.S0)
    reg <<= Case10.S1  # trivially uses no input — but we'll pass them as roots via outputs

    out = m.output(UInt(8), "out")
    out <<= big + big2          # 18 bits of input domain = 2^18 combos > cap

    with pytest.raises(TooLargeForExhaustiveExtraction):
        extract_transition_table(reg, Case10, outputs=[out],
                                  max_input_combinations=1 << 16)


def test_extract_raises_when_reg_has_no_driver():
    m = Module("t", with_clock=True, with_reset=False)
    reg = m.reg(Case10.typ, "reg", init=Case10.S0)
    # Note: no <<= assignment to reg.
    with pytest.raises(ValueError, match="no driver"):
        extract_transition_table(reg, Case10, outputs=[])
