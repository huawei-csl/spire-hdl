"""Step 7: Hopcroft DFA minimisation on an extracted transition table.

Operates on the ``TransitionTable`` produced by ``_table.py``. Returns a
mapping ``{state_value -> canonical_state_value}`` where every state in a
behavioural-equivalence class maps to a single representative.

The algorithm is the standard partition-refinement:

1. Initial partition: group states by their per-input output signature
   (Moore output for each input combination).
2. Refine: split each block whenever two states have different
   next-state-class signatures across all inputs.
3. Iterate until stable.

Canonical representative is the *smallest* state value in each class so the
original ``S0`` survives wherever possible (deterministic, debuggable).

Only the transition table is consulted; the State class is not modified
here. Callers (e.g. ``optimized_fsm``) use the returned mapping to drive
``apply_encoding``, which mutates the class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spirehdl.optimize.fsm._table import TransitionTable


def minimize_fsm(table: "TransitionTable") -> dict[int, int]:
    """Return ``{state_value -> canonical_state_value}``.

    Equivalent states (those that produce the same output sequence under
    every input sequence) map to a single canonical representative.

    If no states are equivalent, the mapping is the identity.
    """
    state_values = list(table.state_values)
    if not state_values:
        return {}

    # Build the initial partition: states with the same per-input output
    # signature belong to the same initial class. For a Moore FSM all rows of
    # the output table for a given state are identical (output depends on
    # state only) so the per-input loop collapses to a single tuple; for
    # Mealy outputs it captures the full input-dependent profile.
    combos = table.all_input_combos()

    def output_sig(sv: int) -> tuple:
        return tuple(table.outputs[sv][ic] for ic in combos)

    initial: dict[tuple, list[int]] = {}
    for sv in state_values:
        initial.setdefault(output_sig(sv), []).append(sv)
    partition = [sorted(block) for block in initial.values()]

    def class_of(sv: int, p: list[list[int]]) -> int:
        for idx, block in enumerate(p):
            if sv in block:
                return idx
        raise AssertionError(f"state {sv} not in partition")

    # Refine: split each block whenever two states have different
    # next-state-class signatures.
    while True:
        new_partition: list[list[int]] = []
        for block in partition:
            sig_to_states: dict[tuple, list[int]] = {}
            for sv in block:
                sig = tuple(class_of(table.transitions[sv][ic], partition) for ic in combos)
                sig_to_states.setdefault(sig, []).append(sv)
            new_partition.extend(sorted(sub) for sub in sig_to_states.values())
        new_partition = sorted(new_partition, key=lambda b: b[0])
        partition_norm = sorted([sorted(b) for b in partition], key=lambda b: b[0])
        if new_partition == partition_norm:
            break
        partition = new_partition

    # Build the canonical map: each state maps to its block's smallest member.
    out: dict[int, int] = {}
    for block in partition:
        canon = min(block)
        for sv in block:
            out[sv] = canon
    return out


def equivalence_classes(table: "TransitionTable") -> list[list[int]]:
    """Convenience: return the equivalence-class partition as a list of lists.

    Useful for displaying / debugging without consulting the canonical map.
    """
    canon = minimize_fsm(table)
    classes: dict[int, list[int]] = {}
    for sv, c in canon.items():
        classes.setdefault(c, []).append(sv)
    return [sorted(v) for v in classes.values()]
