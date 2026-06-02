"""Two-stage encoding search: cheap adjacency screen → verify top-K with cost_fn.

Brute-force / swap encoding search pays a full synthesis per candidate, which is
slow (ticket: 20 160 encodings ≈ hours) and, for the noisy `adp_proxy`, swap can
get stuck in a local minimum. This module screens the whole space with a
*synthesis-free* analytic cost — the classic MUSTANG/JEDI weighted-Hamming state
assignment objective — and then runs the real ``cost_fn`` only on the top-K
screened candidates.

The adjacency weight between two states is how much they "want" to share logic:
``w(i,j) = (#outputs they agree on) + (#input combinations where they go to the
same next state)``. The screen cost of an encoding is
``Σ_{i<j} w(i,j) · HammingDistance(code_i, code_j)`` — low cost clusters
logic-sharing states into Hamming-adjacent codes, which minimises the
next-state/output logic. The screen is O(states²) arithmetic per candidate (no
yosys/aigverse), so the full space scores in well under a second; only the K
verifications touch the real flow.

Empirically (ticket, nangate45 ADP): screen all 20 160 in ~0.07 s, verify
top-64 in ~40 s → finds a *better* encoding than the 2 h exhaustive proxy search
(it optimises the real objective on the screened set rather than the proxy).
"""
from __future__ import annotations

from itertools import permutations
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from spirehdl.optimize.fsm._table import (
    TooLargeForExhaustiveExtraction,
    extract_transition_table,
)

if TYPE_CHECKING:
    from spirehdl.spirehdl import Signal
    from spirehdl.spirehdl_state import State


def _adjacency_weights(table, state_cls: "type[State]") -> Dict[tuple, int]:
    """Weight per (name_i, name_j) pair: output-agreement + next-state-agreement.

    Encoding-independent structure recovered from the transition table (which is
    enumerated under the *current* declared encoding; we map codes back to names
    via ``state_cls._values``).
    """
    names = state_cls.names
    val = state_cls._values
    combos = table.all_input_combos()
    rep = combos[0] if combos else None
    W: Dict[tuple, int] = {}
    for i, a in enumerate(names):
        va = val[a]
        for b in names[i + 1:]:
            vb = val[b]
            # output agreement: count matching output bits (Moore: input-
            # independent, so one representative input combo suffices).
            if rep is not None:
                oa = table.outputs[va][rep]
                ob = table.outputs[vb][rep]
                out_agree = sum(1 for x, y in zip(oa, ob) if x == y)
            else:
                out_agree = 0
            # next-state agreement: # input combos where both go to the same state
            ns_agree = sum(1 for ic in combos
                           if table.transitions[va][ic] == table.transitions[vb][ic])
            W[(a, b)] = out_agree + ns_agree
    return W


def _hamming(x: int, y: int) -> int:
    return bin(x ^ y).count("1")


def _screen_cost(assignment: Dict[str, int], weights: Dict[tuple, int]) -> int:
    return sum(w * _hamming(assignment[a], assignment[b])
               for (a, b), w in weights.items())


def adjacency_search(
    state_cls: "type[State]",
    cost_fn: Callable[[Dict[str, int]], float],
    module,
    *,
    width: Optional[int] = None,
    top_k: int = 64,
    outputs=None,
    state_reg: "Optional[Signal]" = None,
) -> Optional[Dict[str, int]]:
    """Return the best name→code assignment found by screen-then-verify.

    Stage 1: score every injective code assignment by the adjacency screen
    (cheap). Stage 2: evaluate ``cost_fn`` (the real objective) on the ``top_k``
    lowest-screen-cost assignments and return the argmin.

    Returns ``None`` if the transition table can't be extracted (caller should
    fall back to another strategy).
    """
    from spirehdl.optimize.fsm._minimize_emit import find_state_register, find_fsm_outputs

    if state_reg is None:
        try:
            state_reg = find_state_register(module, state_cls)
        except Exception:
            return None
    if outputs is None:
        try:
            outputs = find_fsm_outputs(module, state_reg, state_cls)
        except Exception:
            outputs = []
    if state_reg._driver is None:
        return None
    try:
        table = extract_transition_table(state_reg, state_cls, outputs=outputs)
    except (TooLargeForExhaustiveExtraction, ValueError):
        return None

    weights = _adjacency_weights(table, state_cls)
    names = state_cls.names
    n = len(names)
    if width is None:
        width = state_cls._width
    n_codes = 1 << width
    if n_codes < n:
        return None

    # Stage 1 — screen the whole space (cheap). For small state counts the code
    # universe is tiny; permutations(n_codes, n) enumerates all injective maps.
    scored: List[tuple] = []
    for codes in permutations(range(n_codes), n):
        assignment = dict(zip(names, codes))
        scored.append((_screen_cost(assignment, weights), assignment))
    scored.sort(key=lambda t: t[0])

    # Stage 2 — verify the real objective on the top-K screened candidates.
    best: Optional[tuple] = None
    for _, assignment in scored[:top_k]:
        c = cost_fn(assignment)
        if c != float("inf") and (best is None or c < best[0]):
            best = (c, assignment)
    return best[1] if best is not None else None
