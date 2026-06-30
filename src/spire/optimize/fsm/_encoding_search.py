"""Bit-assignment search.

Given a State subclass and a ``cost_fn`` that scores an assignment by applying it to the design and reading back a
synthesis metric, find the cheapest bit-assignment under one of four strategies:

- ``predefined`` — try BINARY / GRAY / ONEHOT-like encodings at the current width (no widening).
- ``exhaustive`` — enumerate every permutation of codepoints over the state set. Only safe for small N (n! ≤ 5040
  by default).
- ``swap`` — pair-swap accept-on-improvement with random restarts.
- ``anneal`` — placeholder for future work (currently raises).
- ``auto`` — pick predefined for n ≤ 2, exhaustive for n! ≤ ``exhaustive_budget``, swap otherwise.

``cost_fn(assignment)`` must take an assignment dict and return a float; ``float('inf')`` is treated as "rejected"
(synthesis failed).
"""
from __future__ import annotations

import random
from itertools import permutations
from math import factorial
from typing import Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from spire.state import State


Strategy = Literal["predefined", "exhaustive", "swap", "anneal", "auto"]
CostFn = Callable[[dict[str, int]], float]


# Defaults / caps -------------------------------------------------------------

DEFAULT_EXHAUSTIVE_BUDGET = 5040          # 7! — case10's natural ceiling
DEFAULT_SWAP_ITERS = 200
DEFAULT_SWAP_RESTARTS = 4


# Predefined encodings -------------------------------------------------------

def _binary_codes(n: int) -> list[int]:
    return list(range(n))


def _gray_codes(n: int) -> list[int]:
    return [i ^ (i >> 1) for i in range(n)]


def _onehot_codes(n: int) -> list[int]:
    return [1 << i for i in range(n)]


# Strategies -----------------------------------------------------------------

def _try_predefined(state_cls: "type[State]", cost_fn: CostFn) -> tuple[dict[str, int], float]:
    """Score BINARY and GRAY at the current width. ONEHOT needs a wider
    register, which apply_encoding doesn't yet support, so we skip it.
    """
    names = list(state_cls.names)
    n = len(names)
    best: tuple[dict[str, int], float] | None = None
    for label, codes in (("binary", _binary_codes(n)), ("gray", _gray_codes(n))):
        assignment = dict(zip(names, codes))
        cost = cost_fn(assignment)
        if best is None or cost < best[1]:
            best = (assignment, cost)
    assert best is not None
    return best


def _exhaustive(
    state_cls: "type[State]",
    cost_fn: CostFn,
    *,
    width: int,
) -> tuple[dict[str, int], float]:
    names = list(state_cls.names)
    n = len(names)
    code_universe = range(1 << width)
    best: tuple[dict[str, int], float] | None = None
    for codes in permutations(code_universe, n):
        assignment = dict(zip(names, codes))
        cost = cost_fn(assignment)
        if best is None or cost < best[1]:
            best = (assignment, cost)
    assert best is not None
    return best


def _swap(
    state_cls: "type[State]",
    cost_fn: CostFn,
    *,
    width: int,
    max_iters: int = DEFAULT_SWAP_ITERS,
    restarts: int = DEFAULT_SWAP_RESTARTS,
    rng: random.Random | None = None,
) -> tuple[dict[str, int], float]:
    rng = rng or random.Random(0)
    names = list(state_cls.names)
    n = len(names)
    code_universe = list(range(1 << width))
    best_overall: tuple[dict[str, int], float] | None = None
    for _ in range(restarts):
        codes = rng.sample(code_universe, n)
        assignment = dict(zip(names, codes))
        cost = cost_fn(assignment)
        for _ in range(max_iters):
            i, j = rng.sample(range(n), 2)
            ni, nj = names[i], names[j]
            assignment[ni], assignment[nj] = assignment[nj], assignment[ni]
            new_cost = cost_fn(assignment)
            if new_cost < cost:
                cost = new_cost
            else:
                assignment[ni], assignment[nj] = assignment[nj], assignment[ni]
        if best_overall is None or cost < best_overall[1]:
            best_overall = (dict(assignment), cost)
    assert best_overall is not None
    return best_overall


# Dispatcher -----------------------------------------------------------------

def search_encoding(
    state_cls: "type[State]",
    cost_fn: CostFn,
    *,
    strategy: Strategy = "auto",
    width: int | None = None,
    exhaustive_budget: int = DEFAULT_EXHAUSTIVE_BUDGET,
    swap_iters: int = DEFAULT_SWAP_ITERS,
    swap_restarts: int = DEFAULT_SWAP_RESTARTS,
    rng: random.Random | None = None,
) -> dict[str, int]:
    """Return the best-scoring assignment found.

    ``width`` defaults to ``state_cls._width`` (no widening). Changing the width requires the apply_encoding
    width-change path, which is future work — pass ``width=None`` for now.

    Strategy ladder for ``"auto"``:
      n ≤ 2                           → predefined (try BINARY only really)
      n! ≤ exhaustive_budget          → exhaustive
      else                            → swap
    """
    if width is None:
        width = state_cls._width

    n = len(state_cls.names)
    if 1 << width < n:
        # Surface "too narrow" before "width change unsupported" — it's a user-facing error that's always wrong
        # regardless of whether width changing is implemented.
        raise ValueError(
            f"width {width} bits cannot encode {n} states (need at least {(n - 1).bit_length()} bits)")

    if width != state_cls._width:
        raise NotImplementedError(f"width change ({state_cls._width} → {width}) not yet supported")

    if strategy == "anneal":
        raise NotImplementedError("anneal strategy is future work")

    if strategy == "auto":
        if n <= 2:
            strategy = "predefined"
        elif factorial(n) <= exhaustive_budget:
            strategy = "exhaustive"
        else:
            strategy = "swap"

    if strategy == "predefined":
        best, _cost = _try_predefined(state_cls, cost_fn)
        return best
    if strategy == "exhaustive":
        best, _cost = _exhaustive(state_cls, cost_fn, width=width)
        return best
    if strategy == "swap":
        best, _cost = _swap(
            state_cls, cost_fn,
            width=width,
            max_iters=swap_iters,
            restarts=swap_restarts,
            rng=rng,
        )
        return best

    raise ValueError(f"unknown strategy {strategy!r}")
