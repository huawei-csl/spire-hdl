"""``optimized_encoding`` — bit-assignment search wrapper.

Usage::

    with optimized_encoding(Op, module=m, objective="cells"):
        # body that uses Op.ADD, Op.SUB, ...
        ...

On ``__exit__``:

1. Verify the State class was actually referenced inside the with-block by walking
   ``_SharedCache.new_wires + state_cls Const objects`` (the ``_SharedCache`` snapshot/diff capture from
   ``_capture``).
2. Build a cost function via ``_cost_oracle.make_cost_fn`` (pyosys + aigverse, in-process — no `yosys` binary
   required).
3. Run ``search_encoding`` with the chosen strategy.
4. Apply the winning bit-assignment via ``apply_encoding``.

The wrapper does *not* run Hopcroft minimisation — that's ``optimized_fsm``'s job. Nest both if both passes are
wanted; inner ``__exit__`` runs first, so an inner ``optimized_fsm`` shrinks the state set before the outer
``optimized_encoding`` searches over the survivors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

from spirehdl.optimize.fsm._capture import SharedCacheSnapshot
from spirehdl.optimize.fsm._cost_oracle import make_cost_fn
from spirehdl.optimize.fsm._encoding_search import search_encoding
from spirehdl.optimize.fsm._walker import find_state_consts

if TYPE_CHECKING:
    from spirehdl.spirehdl_module import Module
    from spirehdl.spirehdl_state import State


Strategy = Literal["predefined", "exhaustive", "swap", "anneal", "auto", "adjacency"]
Objective = Literal["cells", "wires", "transistors", "aig_gates", "aig_depth", "adp_proxy"]


class optimized_encoding:
    """Bit-assignment search wrapper for a ``State`` subclass.

    Parameters
    ----------
    state_cls : subclass of State
        The state set whose encoding is being optimised.
    module : Module
        The Module the user populates inside the ``with`` block. Required so
        the cost oracle can synthesise candidate encodings.
    objective : "cells" | "wires" | "transistors" | "aig_gates" | "aig_depth" | "adp_proxy"
        Synthesis metric to minimise. ``adp_proxy`` = ``aig_gates × aig_depth``,
        a PDK-free area×delay proxy (no liberty / STA needed); it implies
        ``bit_level_emit`` since it is measured on the combinational cone.
    search : strategy name (see ``_encoding_search.search_encoding``)
    width : optional int (currently must equal ``state_cls._width``)
    cost_fn : optional Callable[[assignment], float]
        Custom cost function. When ``None`` (default), uses
        ``_cost_oracle.make_cost_fn`` (in-process pyosys + aigverse).
    bit_level_emit : bool
        When True, after the encoding is chosen the FSM's next-state bits and
        outputs are rewritten as **minimised bit-level Boolean expressions** over
        the state-register bits (via ``_minimize_emit.minimize_and_rewrite``),
        instead of leaving the ``state == CODE`` mux tree. This stops Yosys from
        re-encoding the FSM (which would discard the search's encoding) and lets
        the chosen encoding reach realised PPA. The search is also measured on
        this emitted form. Requires sympy. See ``INVESTIGATION_fsm_encoding_api.md``.
    dont_cares : bool
        Feed unused state codes to the minimiser as don't-cares. Objective-
        dependent: shrinks gate/area count but can lengthen the critical path
        (helps cells/area, can hurt ADP). Default False.
    top_k : int
        For ``search="adjacency"``: how many screen-ranked candidates to verify
        with the real objective. Default 64. (Ignored by other strategies.)

    Search strategies
    -----------------
    ``"adjacency"`` is a fast two-stage search (see ``_adjacency``): it screens
    every encoding with a synthesis-free weighted-Hamming cost in <1 s, then
    runs the real ``cost_fn`` only on the ``top_k`` best — far faster than
    ``exhaustive`` and, because it verifies the real objective on the screened
    set, more robust than ``swap`` for noisy objectives like ``adp_proxy``.
    """

    def __init__(
        self,
        state_cls: "type[State]",
        module: "Module",
        *,
        objective: Objective = "cells",
        search: Strategy = "auto",
        width: int | None = None,
        cost_fn: Callable[[dict[str, int]], float] | None = None,
        bit_level_emit: bool = False,
        dont_cares: bool = False,
        top_k: int = 64,
    ) -> None:
        self.state_cls = state_cls
        self.module = module
        self.objective = objective
        self.search = search
        self.width = width
        self.cost_fn = cost_fn
        # adp_proxy is an AIG (combinational-cone) metric, so it requires the
        # bit-level emit path both to measure and to make the win real.
        self.bit_level_emit = bit_level_emit or objective == "adp_proxy"
        self.dont_cares = dont_cares
        self.top_k = top_k
        self._snap: SharedCacheSnapshot | None = None

    def __enter__(self) -> "optimized_encoding":
        self._snap = SharedCacheSnapshot()
        self._snap.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._snap is not None
        self._snap.__exit__(exc_type, exc, tb)
        if exc_type is not None:
            return False

        # Verify the State class was referenced inside the with-block. We walk:
        #   - every wire added to _SharedCache during the block (covers non-trivial Exprs the user built);
        #   - every Module signal's _driver (covers register / output assignments to bare state Consts that bypass
        #     _maybe_share).
        # If none of those reference the State class, there's nothing to do.
        roots = list(self._snap.new_wires) + [s._driver for s in self.module._signals if s._driver is not None]
        found = find_state_consts(roots, self.state_cls)
        if not found:
            return False

        cost_fn = self.cost_fn or make_cost_fn(
            self.module, self.state_cls, objective=self.objective,
            bit_level_emit=self.bit_level_emit, dont_cares=self.dont_cares)

        # Detect existing equivalence groups (e.g. produced by a nested ``optimized_fsm`` that merged states via
        # Hopcroft). Names sharing the same current value are in the same group and must continue to share a value
        # during the encoding search — otherwise the search would re-spread merged states.
        groups = _equivalence_groups(self.state_cls)

        best = None
        if self.search == "adjacency" and all(len(g) == 1 for g in groups):
            # Two-stage: synthesis-free adjacency screen → verify top_k with cost_fn.
            from spirehdl.optimize.fsm._adjacency import adjacency_search
            best = adjacency_search(self.state_cls, cost_fn, self.module,
                                    width=self.width, top_k=self.top_k)
            # Fall back to swap if the table couldn't be extracted (best is None).
            if best is None:
                best = _search_group_aware(self.state_cls, cost_fn, groups,
                                           strategy="swap", width=self.width)
        else:
            strat = "swap" if self.search == "adjacency" else self.search
            best = _search_group_aware(self.state_cls, cost_fn, groups, strategy=strat, width=self.width)

        # Commit the chosen encoding. The cost_fn already restored after the final trial — re-apply the winner here
        # so it sticks.
        from spirehdl.optimize.fsm._emit import apply_encoding
        apply_encoding(self.state_cls, best, width=self.width)

        # Bit-level emit: rewrite the FSM logic as minimised Boolean expressions
        # over the state-register bits so Yosys can't re-encode it away.
        if self.bit_level_emit:
            from spirehdl.optimize.fsm._minimize_emit import minimize_and_rewrite
            minimize_and_rewrite(self.module, self.state_cls,
                                 dont_cares=self.dont_cares)
        return False


def _equivalence_groups(state_cls: "type[State]") -> list[list[str]]:
    """Return groups of state names that currently share a value.

    Useful for composing ``optimized_encoding`` after ``optimized_fsm``: the inner wrapper merged equivalent states
    by assigning them the same Const value, and the outer search must respect that grouping.
    """
    by_value: dict[int, list[str]] = {}
    for name in state_cls.names:
        by_value.setdefault(state_cls._values[name], []).append(name)
    # Stable order: groups sorted by their representative's name.
    return sorted(by_value.values(), key=lambda g: g[0])


def _search_group_aware(
    state_cls: "type[State]",
    cost_fn,
    groups: list[list[str]],
    *,
    strategy: Strategy,
    width: int | None,
) -> dict[str, int]:
    """Run encoding search where names in the same group must share a value.

    When ``len(groups) == len(state_cls.names)`` (no merging) this collapses to ``search_encoding`` — same
    semantics, same result.
    """
    # Trivial case: every group is a singleton → defer to search_encoding.
    if all(len(g) == 1 for g in groups):
        return search_encoding(state_cls, cost_fn, strategy=strategy, width=width)

    # Non-trivial: do a group-level search inline. We enumerate assignments
    # of distinct codes (one per group) and let cost_fn judge the spread.
    if width is None:
        width = state_cls._width
    n_groups = len(groups)
    if (1 << width) < n_groups:
        raise ValueError(f"width {width} bits cannot encode {n_groups} groups")

    # The dispatch ladder mirrors search_encoding's, but iterates over
    # *group* permutations rather than per-name permutations.
    from itertools import permutations
    from math import factorial

    code_universe = list(range(1 << width))

    def expand(group_codes: tuple[int, ...]) -> dict[str, int]:
        out: dict[str, int] = {}
        for grp, code in zip(groups, group_codes):
            for name in grp:
                out[name] = code
        return out

    best: tuple[dict[str, int], float] | None = None

    if strategy in ("predefined",):
        # Try BINARY codes for groups (0, 1, 2, …) and a Gray-code variant.
        candidates = [
            tuple(range(n_groups)),
            tuple(i ^ (i >> 1) for i in range(n_groups)),
        ]
        for codes in candidates:
            full = expand(codes)
            cost = cost_fn(full)
            if best is None or cost < best[1]:
                best = (full, cost)
    elif strategy == "anneal":
        raise NotImplementedError("anneal strategy is future work")
    elif strategy in ("auto", "exhaustive", "swap"):
        # Auto: pick exhaustive when feasible.
        n_perms = factorial(1 << width) // factorial((1 << width) - n_groups)
        do_exhaustive = strategy == "exhaustive" or (strategy == "auto" and n_perms <= 5040)
        if do_exhaustive:
            for codes in permutations(code_universe, n_groups):
                full = expand(codes)
                cost = cost_fn(full)
                if best is None or cost < best[1]:
                    best = (full, cost)
        else:
            # Swap-based local search over group code assignments.
            import random
            rng = random.Random(0)
            for _restart in range(4):
                codes = list(rng.sample(code_universe, n_groups))
                full = expand(tuple(codes))
                cost = cost_fn(full)
                for _ in range(200):
                    i, j = rng.sample(range(n_groups), 2)
                    codes[i], codes[j] = codes[j], codes[i]
                    new_full = expand(tuple(codes))
                    new_cost = cost_fn(new_full)
                    if new_cost < cost:
                        cost = new_cost
                        full = new_full
                    else:
                        codes[i], codes[j] = codes[j], codes[i]
                if best is None or cost < best[1]:
                    best = (full, cost)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    assert best is not None
    return best[0]
