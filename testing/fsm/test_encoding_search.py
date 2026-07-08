"""Encoding-search strategies (Step 8) with a synthetic cost_fn."""
from __future__ import annotations

import pytest

from spire.optimize.fsm._emit import restore_encoding
from spire.optimize.fsm._encoding_search import search_encoding
from spire.state import Encoding, State, state


class S4(State, encoding=Encoding.BINARY):
    A = state(); B = state(); C = state(); D = state()


@pytest.fixture(autouse=True)
def _restore_s4():
    yield
    restore_encoding(S4, {"A": 0, "B": 1, "C": 2, "D": 3})


def _exact_match_cost(target: dict[str, int]):
    """A synthetic cost: 0 when assignment matches `target`, else hamming-ish."""
    def cost_fn(a: dict[str, int]) -> float:
        return float(sum(1 for k in target if a[k] != target[k]))
    return cost_fn


def test_exhaustive_finds_unique_minimum():
    target = {"A": 3, "B": 0, "C": 1, "D": 2}
    chosen = search_encoding(S4, _exact_match_cost(target), strategy="exhaustive")
    assert chosen == target


def test_exhaustive_finds_among_ties():
    """If two assignments share the minimum cost, exhaustive picks one of them
    deterministically (first-encountered)."""
    # Cost is 0 if A>B, 1 otherwise. Many optimal assignments.
    def cost_fn(a: dict[str, int]) -> float:
        return 0.0 if a["A"] > a["B"] else 1.0
    chosen = search_encoding(S4, cost_fn, strategy="exhaustive")
    assert chosen["A"] > chosen["B"]


def test_predefined_picks_better_of_binary_gray():
    """Predefined tries BINARY and GRAY; whichever scores lower wins."""
    # GRAY for 4 states = [0, 1, 3, 2]. BINARY = [0, 1, 2, 3].
    # Define a cost that prefers GRAY's pattern (C=3, D=2).
    def cost_fn(a: dict[str, int]) -> float:
        return 0.0 if (a["C"] == 3 and a["D"] == 2) else 1.0
    chosen = search_encoding(S4, cost_fn, strategy="predefined")
    assert chosen == {"A": 0, "B": 1, "C": 3, "D": 2}


def test_swap_converges_on_known_minimum():
    target = {"A": 2, "B": 3, "C": 1, "D": 0}
    import random
    chosen = search_encoding(
        S4, _exact_match_cost(target),
        strategy="swap", swap_iters=50, swap_restarts=8,
        rng=random.Random(42),
    )
    # Swap is a local-search heuristic but for n=4 with multiple restarts and
    # a clean cost landscape it should reliably find the global minimum.
    assert chosen == target


def test_auto_strategy_picks_exhaustive_for_small_n():
    """N=4 → 24 candidates, well under the 5040 default → auto picks exhaustive."""
    target = {"A": 1, "B": 0, "C": 3, "D": 2}
    chosen = search_encoding(S4, _exact_match_cost(target), strategy="auto")
    assert chosen == target


def test_anneal_is_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        search_encoding(S4, lambda a: 0.0, strategy="anneal")


def test_width_change_not_supported():
    with pytest.raises(NotImplementedError):
        search_encoding(S4, lambda a: 0.0, strategy="exhaustive", width=4)


def test_width_too_narrow_for_n():
    """N=4 states need at least 2 bits."""
    class S5(State, encoding=Encoding.BINARY):
        V = state(); W = state(); X = state(); Y = state(); Z = state()
    # S5 needs 3 bits. Passing width=2 should fail.
    with pytest.raises(ValueError, match="cannot encode"):
        # apply_encoding would later refuse the width change too, but the
        # search must catch this up front before iterating.
        search_encoding(S5, lambda a: 0.0, strategy="exhaustive", width=2)


def test_auto_ladder_gates_on_real_permutation_space():
    """ISSUES2 7.5: the exhaustive space is P(2^width, n), not n!. Gating on n! sent 5-state
    ONEHOT classes (P(32,5) = 24,165,120 full syntheses) into an effective hang — at the
    pre-fix baseline this test does not fail, it times out."""
    from spire.optimize.fsm._encoding_search import _perm_count

    assert _perm_count(32, 5, 5040) > 5040          # the real ONEHOT-class space
    assert _perm_count(8, 8, 50000) == 40320        # width-matched: equals n! (8!)

    class S5(State, encoding=Encoding.ONEHOT):
        A = state(); B = state(); C = state(); D = state(); E = state()

    calls = []

    def cost(assignment):
        calls.append(1)
        return float(len(calls))

    got = search_encoding(S5, cost, strategy="auto")
    assert got is not None
    assert len(calls) < 5000, f"auto picked exhaustive on a ONEHOT class ({len(calls)} calls)"
