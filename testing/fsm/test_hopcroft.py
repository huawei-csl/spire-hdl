"""Hopcroft DFA minimisation (Step 7)."""
from __future__ import annotations

from spirehdl.fsm._hopcroft import equivalence_classes, minimize_fsm
from spirehdl.fsm._table import TransitionTable


def _make_table(
    states: list[int],
    transitions: dict[int, dict[tuple[int, ...], int]],
    outputs: dict[int, dict[tuple[int, ...], tuple[int, ...]]],
) -> TransitionTable:
    """Build a TransitionTable by hand (without going through eval)."""
    # Derive the input-combo list from the transitions dict (every state has
    # the same set of input tuples in a well-formed table).
    first = next(iter(transitions.values()))
    combos = list(first.keys())
    t = TransitionTable(
        state_values=states,
        input_signals=[],
        transitions=transitions,
        outputs=outputs,
        output_signals=[],
        input_combos=combos,
    )
    return t


def test_already_minimal_fsm_is_unchanged():
    """A 3-state machine with distinct outputs and distinct transition
    signatures cannot be reduced; the canonical map is the identity."""
    states = [0, 1, 2]
    transitions = {
        0: {(0,): 1, (1,): 2},
        1: {(0,): 2, (1,): 0},
        2: {(0,): 0, (1,): 1},
    }
    outputs = {0: {(0,): (0,), (1,): (0,)},
               1: {(0,): (1,), (1,): (1,)},
               2: {(0,): (2,), (1,): (2,)}}

    table = _make_table(states, transitions, outputs)
    canon = minimize_fsm(table)
    assert canon == {0: 0, 1: 1, 2: 2}


def test_two_equivalent_states_merge():
    """States 1 and 2 produce the same output and the same next-state
    signatures, so they should merge into a single class."""
    states = [0, 1, 2]
    transitions = {
        0: {(0,): 1, (1,): 2},
        1: {(0,): 0, (1,): 0},
        2: {(0,): 0, (1,): 0},
    }
    outputs = {0: {(0,): (1,), (1,): (1,)},
               1: {(0,): (0,), (1,): (0,)},
               2: {(0,): (0,), (1,): (0,)}}

    table = _make_table(states, transitions, outputs)
    canon = minimize_fsm(table)
    # 1 and 2 are equivalent; canonical representative is the smaller value (1).
    assert canon[1] == canon[2]
    assert canon[0] == 0
    # The smaller of {1, 2} is 1.
    assert canon[1] == 1


def test_case10_seven_to_four_classes():
    """Case10 (7-state FSM) collapses to 4 equivalence classes under Hopcroft.
    This is the canonical motivating example from the FSM-encoding-search
    proposal: {S0, S3}, {S1}, {S5}, {S2, S4, S6}.
    """
    # Hand-built transition + output tables for case10 (matching test_table.py).
    states = list(range(7))   # S0..S6 with BINARY encoding (values 0..6)
    transitions = {
        0: {(0,): 1, (1,): 2},
        1: {(0,): 3, (1,): 5},
        2: {(0,): 5, (1,): 4},
        3: {(0,): 1, (1,): 6},
        4: {(0,): 5, (1,): 2},
        5: {(0,): 4, (1,): 3},
        6: {(0,): 5, (1,): 6},
    }
    outputs_per_state = {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0}
    outputs = {
        sv: {(0,): (v,), (1,): (v,)}
        for sv, v in outputs_per_state.items()
    }

    table = _make_table(states, transitions, outputs)
    classes = equivalence_classes(table)

    # Sort each class internally and the list of classes by smallest element.
    classes_sorted = sorted([sorted(c) for c in classes], key=lambda c: c[0])
    expected = [[0, 3], [1], [2, 4, 6], [5]]
    assert classes_sorted == expected


def test_singleton_fsm_is_minimal():
    """One-state FSM: trivially minimal."""
    states = [0]
    transitions = {0: {(0,): 0}}
    outputs    = {0: {(0,): (7,)}}
    table = _make_table(states, transitions, outputs)
    assert minimize_fsm(table) == {0: 0}


def test_canonical_representative_is_smallest_value():
    """Confirm the canonical-state choice is deterministic + uses the smallest."""
    states = [0, 3, 5]   # 3 and 5 are equivalent.
    transitions = {
        0: {(0,): 3, (1,): 5},
        3: {(0,): 0, (1,): 0},
        5: {(0,): 0, (1,): 0},
    }
    outputs = {0: {(0,): (1,), (1,): (1,)},
               3: {(0,): (0,), (1,): (0,)},
               5: {(0,): (0,), (1,): (0,)}}
    table = _make_table(states, transitions, outputs)
    canon = minimize_fsm(table)
    # min({3, 5}) = 3.
    assert canon[5] == 3
    assert canon[3] == 3
    assert canon[0] == 0
