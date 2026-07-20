"""Syntactic analysis of selection cascades and their arm conditions.

This is the neutral base layer shared by two consumers with opposite roles:

  * :mod:`spire.control_structures` (language semantics) uses
    :class:`DisjointnessTracker` to skip emitting redundant first-match
    gating (``cond & ~covered``) exactly where it is provably dead;
  * :mod:`spire.selection_emission` (optional lowering optimization) uses
    :func:`collect_chain` / :func:`analyze_chain` to validate and rebuild
    cascade topologies from the expression graph.

Keeping the analysis here keeps the dependency graph one-way:
``expr -> selection_analysis -> control_structures -> selection_emission``.

Selects are classified into a small lattice over label sets of ONE selector:

  ("consts", sel, S)      — true iff selector value is in S
  ("complement", sel, T)  — true iff selector value is NOT in T
  ("zero", None, {})      — constant false
  ("opaque", None, {})    — anything else

The boolean connectives &, |, ~ are evaluated set-theoretically where both
sides classify against the same selector — which makes redundant first-match
gating (`cond & ~covered`) provably collapse: consts(S) & complement(T) is
consts(S - T). This is what lets gated if_/elif_ eq-chains qualify for the
one-hot emission forms with no construct-specific handling at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from spire.expr import Const, Expr, Op1, Op2, Resize, Signal, Ternary


# ---------------------------------------------------------------------------
# chain collection
# ---------------------------------------------------------------------------

def _deref(e: Expr, spine: Optional[set] = None) -> Expr:
    """Look through auto-generated wrapper Signals (``_maybe_share``) and
    width-preserving Resize nodes. Without this, chains built through spire's
    expression API are invisible: every mux is wrapped in a shared wire."""
    while True:
        if spine is not None:
            spine.add(id(e))
        if (isinstance(e, Signal) and getattr(e, "_auto_generated", False)
                and e._driver is not None):
            e = e._driver
            continue
        if isinstance(e, Resize) and e.typ.width == e.a.typ.width:
            e = e.a
            continue
        return e


def collect_chain(head: Expr,
                  spine: Optional[set] = None) -> Tuple[List[Tuple[Expr, Expr]], Expr]:
    """Collect the maximal priority cascade starting at ``head``.

    Returns ``(pairs, default)`` where ``pairs`` is ``[(sel, val), ...]`` in
    priority order (head arm first) and ``default`` is the chain tail.
    ``spine`` (optional set) records the ids of every wrapper/Ternary node
    traversed — callers use it to mark the cascade as visited.
    """
    pairs: List[Tuple[Expr, Expr]] = []
    cur = _deref(head, spine)
    while isinstance(cur, Ternary) and cur.sel.typ.width == 1:
        pairs.append((cur.sel, cur.a))
        cur = _deref(cur.b, spine)
    return pairs, cur


# ---------------------------------------------------------------------------
# arm-select classification
# ---------------------------------------------------------------------------

_CONSTS = "consts"
_COMPLEMENT = "complement"
_ZERO = "zero"      # constant false
_ONE = "one"        # constant true
_OPAQUE = "opaque"

_OPAQUE_R = (_OPAQUE, None, frozenset())
_ZERO_R = (_ZERO, None, frozenset())
_ONE_R = (_ONE, None, frozenset())


class _SelectorKeys:
    """Classification context: structural identity for selector expressions
    (two `x[0:2]` slices are different objects but the same selector), plus an
    optional node budget bounding the walk — construction-time callers reset
    it per arm; when exhausted, classification bails to opaque. ``None`` means
    unlimited (analysis-time callers)."""

    def __init__(self, budget: Optional[int] = None) -> None:
        from spire.simplify import _KeyWalker
        self._walker = _KeyWalker()
        self._reps: Dict[tuple, Expr] = {}
        self.budget = budget

    def key(self, e: Expr) -> tuple:
        k = self._walker.visit(e)
        self._reps.setdefault(k, e)
        return k

    def rep(self, k: tuple) -> Expr:
        return self._reps[k]

    def spend(self) -> bool:
        """Consume one node visit; False once the budget is exhausted."""
        if self.budget is None:
            return True
        self.budget -= 1
        return self.budget >= 0


def _classify_sel(e: Expr, keys: _SelectorKeys):
    """-> (kind, selector_key | None, frozenset(labels))"""
    if not keys.spend():
        return _OPAQUE_R
    e = _deref(e)
    if isinstance(e, Const):
        return _ZERO_R if e.value == 0 else _ONE_R
    if isinstance(e, Op1) and e.op == "~":
        kind, sel, labels = _classify_sel(e.a, keys)
        if kind == _CONSTS:
            return (_COMPLEMENT, sel, labels)
        if kind == _COMPLEMENT:
            return (_CONSTS, sel, labels)
        if kind == _ZERO:
            return _ONE_R
        if kind == _ONE:
            return _ZERO_R
        return _OPAQUE_R
    if isinstance(e, Op2) and e.op in ("|", "&"):
        lk, ls, ll = _classify_sel(e.a, keys)
        rk, rs, rl = _classify_sel(e.b, keys)
        if e.op == "|":
            if lk == _ONE or rk == _ONE:
                return _ONE_R
            if lk == _ZERO:
                return (rk, rs, rl)
            if rk == _ZERO:
                return (lk, ls, ll)
            if ls != rs or lk == _OPAQUE or rk == _OPAQUE:
                return _OPAQUE_R
            if lk == _CONSTS and rk == _CONSTS:
                return (_CONSTS, ls, ll | rl)
            return _OPAQUE_R  # unions involving complements: not needed
        # "&"
        if lk == _ZERO or rk == _ZERO:
            return _ZERO_R
        if lk == _ONE:
            return (rk, rs, rl)
        if rk == _ONE:
            return (lk, ls, ll)
        if ls != rs or lk == _OPAQUE or rk == _OPAQUE:
            return _OPAQUE_R
        if lk == _CONSTS and rk == _CONSTS:
            inter = ll & rl
            return (_CONSTS, ls, inter) if inter else _ZERO_R
        if lk == _CONSTS and rk == _COMPLEMENT:
            diff = ll - rl
            return (_CONSTS, ls, diff) if diff else _ZERO_R
        if lk == _COMPLEMENT and rk == _CONSTS:
            diff = rl - ll
            return (_CONSTS, ls, diff) if diff else _ZERO_R
        return (_COMPLEMENT, ls, ll | rl)  # complement & complement
    if isinstance(e, Op2) and e.op == "==":
        a, b = e.a, e.b
        if isinstance(b, Const) and not isinstance(a, Const):
            return (_CONSTS, keys.key(a), frozenset([int(b.value)]))
        if isinstance(a, Const) and not isinstance(b, Const):
            return (_CONSTS, keys.key(b), frozenset([int(a.value)]))
    return _OPAQUE_R


class DisjointnessTracker:
    """Incremental disjointness proof over the arms of one selection construct
    (a switch_ or an if_/elif_ chain — both use this identically).

    ``claim(cond)`` returns True iff ``cond`` is provably disjoint from every
    earlier claimed arm: it classifies as eq-const labels on the same single
    selector, with no label collision. Labels are tracked even on a collision
    (only the colliding arm loses its claim; later fresh-label arms stay
    provable). An unclassifiable (opaque) condition ends tracking — its labels
    are unknown, so no later disjointness claim can be proven.

    Used by the constructs to skip emitting redundant first-match gating
    (``cond & ~covered``) exactly where it is provably dead.
    """

    def __init__(self, budget: int = 256):
        self._keys: Optional[_SelectorKeys] = None
        self._sel_key = None
        self._seen: frozenset = frozenset()
        self._trackable = True
        self._budget = budget

    @property
    def trackable(self) -> bool:
        return self._trackable

    @property
    def seen_labels(self) -> frozenset:
        return self._seen

    def claim(self, cond_expr: Expr) -> bool:
        if not self._trackable:
            return False
        if self._keys is None:
            self._keys = _SelectorKeys(budget=self._budget)
        self._keys.budget = self._budget  # fresh per-arm node budget
        kind, key, labels = _classify_sel(cond_expr, self._keys)
        if kind != _CONSTS or (self._sel_key is not None and key != self._sel_key):
            self._trackable = False
            return False
        self._sel_key = key
        fresh = not (labels & self._seen)
        self._seen = self._seen | labels
        return fresh


# ---------------------------------------------------------------------------
# whole-cascade analysis
# ---------------------------------------------------------------------------

@dataclass
class ChainAnalysis:
    pairs: List[Tuple[Expr, Expr]]
    default: Expr
    n: int
    disjoint: bool                       # all arms const-eq, one selector, disjoint
    selector: Optional[Expr]             # representative selector expr
    selector_width: int
    arm_labels: List[Optional[frozenset]]  # per arm; None for a complement arm
    complement_arm: Optional[int]        # index of the (single) complement arm


def analyze_chain(pairs: List[Tuple[Expr, Expr]], default: Expr) -> ChainAnalysis:
    keys = _SelectorKeys()
    sel_key = None
    labels: List[Optional[frozenset]] = []
    complement_arm: Optional[int] = None
    seen: set = set()
    disjoint = len(pairs) > 0

    for i, (sel, _val) in enumerate(pairs):
        kind, k, ls = _classify_sel(sel, keys)
        if kind == _CONSTS:
            if sel_key is None:
                sel_key = k
            if k != sel_key or (ls & seen):
                disjoint = False
                labels.append(None)
                continue
            seen |= ls
            labels.append(ls)
        elif kind == _COMPLEMENT and complement_arm is None:
            # candidate default-style arm (switch_ `default()`): valid iff its
            # label set equals the union of all *other* arms' labels — checked
            # after the loop, since later arms may still add labels.
            complement_arm = i
            labels.append(None)
            if sel_key is None:
                sel_key = k
            elif k != sel_key:
                disjoint = False
        else:
            disjoint = False
            labels.append(None)

    if complement_arm is not None and disjoint:
        kind, k, ls = _classify_sel(pairs[complement_arm][0], keys)
        if not (k == sel_key and ls == seen):
            disjoint = False

    selector = keys.rep(sel_key) if (sel_key is not None and disjoint) else None
    return ChainAnalysis(
        pairs=pairs,
        default=default,
        n=len(pairs),
        disjoint=disjoint and selector is not None,
        selector=selector,
        selector_width=selector.typ.width if selector is not None else 0,
        arm_labels=labels,
        complement_arm=complement_arm,
    )
