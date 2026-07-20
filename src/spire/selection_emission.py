"""Selection-cascade emission rewrites: chain / tournament / andor / bittree.

Spire lowers ``switch_``/``case_``, ``if_``/``elif_`` and hand-written nested
``mux()`` calls to *linear chains* of binary ``Ternary`` nodes at construction
time. Downstream synthesis (yosys + ABC) cannot rebalance those chains: a mux
chain is not associative, and the mutual exclusivity of case arms — which lets
a Verilog ``case`` lower to a flat parallel ``$pmux`` network — is erased by
the lowering. The result is O(N) logic depth where O(log N) is available
(measured on the CV32E40P ALU port: find-first-one chain 31 vs 9 levels at
identical area; full design 9342 AND / 91 levels vs 8901 / 79 restructured).

One concept — *set the emission style for a scope* — at three granularities:

  * region:        ``with mux_emission("andor"): ...`` — captures every signal
                    assigned inside (switch_/if_ arms and hand-built chains
                    alike) and eagerly rewrites their final cascades on exit;
  * function:      ``@mux_emission("tournament")`` — the same object as a
                    decorator; eagerly rewrites the returned cascade;
  * whole design:  ``to_verilog_file(..., selection_emission=True)`` — the
                    :func:`apply_selection_emission` pass auto-detects untagged
                    cascades above the config thresholds.

Region and function forms are **eager**: the rewrite is baked into the
expression graph at construction, so every backend (Verilog, AIGER export,
Simulator, analyze) sees the same structure.

Modes and their prerequisites (validation is *shape-based* — the analyzer
judges the cascade, never the construct it came from):

  * ``"chain"``      — leave as-is. O(N) depth. Always legal.
  * ``"tournament"`` — parallel-prefix first-match tree, node
                       ``(sl | sr, mux(sl, vl, vr))``. Preserves priority
                       universally, O(log N) depth. Always legal.
  * ``"andor"``      — one-hot AND-mask + balanced OR (the ``$pmux`` form).
                       Requires provably disjoint arm selects: ``sel == const``
                       terms (or ORs of them) on one selector with pairwise-
                       distinct constants. Redundant first-match gating
                       (``cond & ~covered``) is seen through when provably
                       dead, so eq-const ``if_``/``elif_`` chains qualify too.
  * ``"bittree"``    — balanced mux tree indexed by the *selector bits* (the
                       arm compares vanish). Additionally needs a value for
                       all 2**K leaves (missing slots fill from the default).
  * ``"auto"``       — best legal form per cascade, subject to the config
                       thresholds; never raises.

See rtl_scout's metadocuments/spire_selection_emission.md for measurements and
the applicability table.
"""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from spire.expr import (
    Concat,
    Const,
    Expr,
    Op1,
    Op2,
    Resize,
    Signal,
    Slice,
    Ternary,
    as_expr,
    cat,
    fit_width,
    mux,
)

MODES = ("chain", "tournament", "andor", "bittree")
REGION_MODES = MODES + ("auto",)


@dataclass
class SelectionEmissionConfig:
    enabled: bool = False        # auto-detect untagged chains in the pass
    andor_min_n: int = 16        # auto: disjoint cascades at/above this -> andor
    tournament_min_n: int = 16   # auto: priority cascades at/above this -> tournament
    bittree_max_sel_bits: int = 6  # refuse bittree beyond 2**6 = 64 leaves


DEFAULT_CONFIG = SelectionEmissionConfig()


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
# arm-select classification (syntactic disjointness analysis)
# ---------------------------------------------------------------------------
#
# Selects are classified into a small lattice over label sets of ONE selector:
#   ("consts", sel, S)      — true iff selector value is in S
#   ("complement", sel, T)  — true iff selector value is NOT in T
#   ("zero", None, {})      — constant false
#   ("opaque", None, {})    — anything else
# The boolean connectives &, |, ~ are evaluated set-theoretically where both
# sides classify against the same selector — which makes redundant first-match
# gating (`cond & ~covered`) provably collapse: consts(S) & complement(T) is
# consts(S - T). This is what lets gated if_/elif_ eq-chains qualify for the
# one-hot forms with no construct-specific handling at all.

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


class _DisjointnessTracker:
    """Incremental disjointness proof over the arms of one selection construct
    (a switch_ or an if_/elif_ chain — both use this identically).

    ``claim(cond)`` returns True iff ``cond`` is provably disjoint from every
    earlier claimed arm: it classifies as eq-const labels on the same single
    selector, with no label collision. Labels are tracked even on a collision
    (only the colliding arm loses its claim; later fresh-label arms stay
    provable). An unclassifiable (opaque) condition ends tracking — its labels
    are unknown, so no later disjointness claim can be proven.

    Used by the constructs to skip emitting redundant first-match gating
    (``cond & ~covered``) exactly where it is provably dead, and to anchor
    early validation errors for the one-hot emission modes.
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


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _or_tree(terms: List[Expr]) -> Expr:
    assert terms
    while len(terms) > 1:
        terms = [terms[i] | terms[i + 1] if i + 1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    return terms[0]


def _rep_mask(bit: Expr, width: int) -> Expr:
    return cat(*([bit] * width)) if width > 1 else bit


def build_tournament(pairs: List[Tuple[Expr, Expr]], default: Expr) -> Expr:
    """Parallel-prefix first-match: universally valid, preserves priority."""
    items = list(pairs)
    while len(items) > 1:
        nxt = []
        for j in range(0, len(items) - 1, 2):
            (sl, vl), (sr, vr) = items[j], items[j + 1]
            nxt.append((sl | sr, mux(sl, vl, vr)))
        if len(items) % 2:
            nxt.append(items[-1])
        items = nxt
    s, v = items[0]
    return Ternary(s, v, default)


def build_andor(analysis: ChainAnalysis, out_typ) -> Expr:
    """One-hot AND-OR network (the `$pmux` form). Caller must have verified
    ``analysis.disjoint``. Arm conditions are REBUILT as balanced OR-trees of
    fresh ``selector == const`` compares — the originals may drag serial
    first-match gating behind them."""
    sel = analysis.selector
    w = out_typ.width
    named_conds: List[Expr] = []
    terms: List[Expr] = []
    complement_val: Optional[Expr] = None

    for i, (orig_sel, val) in enumerate(analysis.pairs):
        if i == analysis.complement_arm:
            complement_val = val
            continue
        labels = analysis.arm_labels[i]
        eqs = [sel == Const(c, sel.typ) for c in sorted(labels)]
        cond = _or_tree(eqs)
        named_conds.append(cond)
        terms.append(_rep_mask(cond, w) & fit_width(as_expr(val), out_typ))

    any_named = _or_tree(named_conds)
    # fallback: a complement arm (switch default()) wins over the chain tail
    # for the un-matched select space — the tail is unreachable then.
    fallback = complement_val if complement_val is not None else analysis.default
    fb = _deref(as_expr(fallback))
    if not (isinstance(fb, Const) and fb.value == 0):
        terms.append(_rep_mask(~any_named, w) & fit_width(as_expr(fallback), out_typ))
    return _or_tree(terms)


def build_bittree(analysis: ChainAnalysis, out_typ,
                  max_sel_bits: int) -> Expr:
    """Balanced mux tree indexed by the selector bits; missing label slots are
    filled with the chain default (or the complement arm's value)."""
    from spire.simplify import _build_bit_tree

    sel = analysis.selector
    k = analysis.selector_width
    if k > max_sel_bits:
        raise ValueError(
            f"bittree: selector has {k} bits -> {1 << k} leaves exceeds the "
            f"limit of 2**{max_sel_bits}; use 'andor' for sparse label sets")
    fallback = (analysis.pairs[analysis.complement_arm][1]
                if analysis.complement_arm is not None else analysis.default)
    leaves: List[Expr] = [fit_width(as_expr(fallback), out_typ)] * (1 << k)
    for i, (_sel, val) in enumerate(analysis.pairs):
        if i == analysis.complement_arm:
            continue
        for c in analysis.arm_labels[i]:
            if c >= (1 << k):
                raise ValueError(f"bittree: label {c} out of range for "
                                 f"{k}-bit selector")
            leaves[c] = fit_width(as_expr(val), out_typ)
    return _build_bit_tree(sel, leaves)


# ---------------------------------------------------------------------------
# mode choice + rewrite entry point
# ---------------------------------------------------------------------------

def choose_mode(analysis: ChainAnalysis, cfg: SelectionEmissionConfig) -> str:
    if analysis.disjoint and analysis.n >= cfg.andor_min_n:
        return "andor"
    if analysis.n >= cfg.tournament_min_n:
        return "tournament"
    return "chain"


def rewrite(head: Expr, mode: Optional[str] = None,
            cfg: SelectionEmissionConfig = DEFAULT_CONFIG) -> Expr:
    """Rewrite the priority cascade headed at ``head`` into ``mode``
    (None/"auto" = pick via ``choose_mode``). Returns ``head`` unchanged when
    the mode resolves to "chain" or no cascade is found under an auto mode.
    Raises ValueError when an explicitly requested mode's prerequisites don't
    hold (shape-based: provable disjointness for "andor"/"bittree")."""
    if mode == "auto":
        mode = None
    if mode is not None and mode not in MODES:
        raise ValueError(f"unknown emission mode {mode!r}; expected one of {REGION_MODES}")

    head_e = as_expr(head)
    pairs, default = collect_chain(head_e)
    if len(pairs) < 2:
        if mode is not None and mode != "chain":
            raise ValueError(f"emission mode {mode!r}: no mux cascade found at this expression")
        return head_e

    analysis = analyze_chain(pairs, default)
    out_typ = _deref(head_e).typ

    if mode is None:
        mode = choose_mode(analysis, cfg)
    elif mode in ("andor", "bittree") and not analysis.disjoint:
        raise ValueError(
            f"emission mode {mode!r} requires pairwise-disjoint constant-label "
            f"arm selects on a single selector; this cascade's selects are not "
            f"provably disjoint (use 'tournament', which preserves priority "
            f"universally)")

    if mode == "chain":
        return head_e
    if mode == "tournament":
        return build_tournament(pairs, default)
    if mode == "andor":
        return build_andor(analysis, out_typ)
    return build_bittree(analysis, out_typ, cfg.bittree_max_sel_bits)


# ---------------------------------------------------------------------------
# the user-facing scope object: `mux_emission`
# ---------------------------------------------------------------------------

class mux_emission:
    """Set the selection-emission style for a scope — one object, two roles.

    Region (context manager)::

        with mux_emission("andor"):
            with switch_(op):
                with case_(A, B): y <<= ...

        with mux_emission("tournament"):
            with if_(c0):   y <<= 1
            with elif_(c1): y <<= 2

        with mux_emission("tournament"):
            y <<= hand_built_mux_chain      # hand chains count too

    Every signal assigned inside the region is captured; on exit each captured
    signal's final selection cascade is eagerly rewritten to ``mode``. Signals
    without a cascade (plain assignments) are skipped. Explicit one-hot modes
    ("andor"/"bittree") raise if a captured cascade cannot provably satisfy
    them; "auto" picks the best legal form per cascade and never raises.
    An if_/elif_ chain may not straddle the region boundary (the region is
    part of the chain-claim scope, so a straddling elif_ fails loudly).

    Function (decorator)::

        @mux_emission("tournament")
        def ff1_index(bits):
            chain = Const(0, UInt(5))
            for k in reversed(range(32)):
                chain = mux(bits[k], Const(k, UInt(5)), chain)
            return chain

    The returned cascade is rewritten eagerly. In both roles the rewrite is
    baked into the expression graph, so all backends see the same structure.
    """

    def __init__(self, mode: str):
        if mode not in REGION_MODES:
            raise ValueError(
                f"mux_emission: unknown mode {mode!r}; expected one of {REGION_MODES}")
        self._mode = mode
        self._entered = False
        self._signals: List[Signal] = []
        self._seen_ids: set = set()

    # -- decorator role ------------------------------------------------------
    def __call__(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return rewrite(as_expr(fn(*args, **kwargs)), self._mode, DEFAULT_CONFIG)
        return wrapper

    # -- region role ---------------------------------------------------------
    def _register(self, signal: Signal) -> None:
        if id(signal) not in self._seen_ids:
            self._seen_ids.add(id(signal))
            self._signals.append(signal)

    def __enter__(self) -> "mux_emission":
        from spire.control_structures import _ConditionState
        if self._entered:
            raise RuntimeError("mux_emission region cannot be re-entered while active")
        self._signals, self._seen_ids = [], set()
        _ConditionState.mux_emission_stack.append(self)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        from spire.control_structures import _ConditionState, fresh_condition_scope
        stack = _ConditionState.mux_emission_stack
        if not self._entered or not stack or stack[-1] is not self:
            raise RuntimeError("mux_emission region stack corruption detected")
        stack.pop()
        self._entered = False
        if exc_type is not None:
            return False
        # The rewrite builds new expressions; run it outside any enclosing
        # conditions so helper wires (fit_type etc.) aren't condition-wrapped.
        with fresh_condition_scope():
            for sig in self._signals:
                self._finalize(sig)
        return False

    def _finalize(self, sig: Signal) -> None:
        drv = getattr(sig, "_driver", None)
        if isinstance(drv, Expr):
            new = _rewrite_outermost_cascade(drv, self._mode, sig.name)
            if new is not drv:
                sig._driver = new


def _rewrite_outermost_cascade(e: Expr, mode: str, signal_name: str) -> Expr:
    """Rewrite the outermost cascade reachable through wrapper nodes
    (width-changing Resize, auto-generated Signals). Non-cascades are left
    alone — a region legitimately contains plain assignments."""
    pairs, _default = collect_chain(e)
    if len(pairs) >= 2:
        try:
            return rewrite(e, mode, DEFAULT_CONFIG)
        except ValueError as err:
            raise ValueError(f"mux_emission: signal '{signal_name}': {err}") from None
    if isinstance(e, Resize):
        e.a = _rewrite_outermost_cascade(e.a, mode, signal_name)
        return e
    if (isinstance(e, Signal) and getattr(e, "_auto_generated", False)
            and e._driver is not None):
        e._driver = _rewrite_outermost_cascade(e._driver, mode, signal_name)
        return e
    return e


# ---------------------------------------------------------------------------
# emission pass: whole-design auto-detection (opt-in)
# ---------------------------------------------------------------------------

def apply_selection_emission(module,
                             cfg: SelectionEmissionConfig = DEFAULT_CONFIG) -> int:
    """Auto-detect and rewrite selection cascades across a netlist per the
    config thresholds (``choose_mode``). Returns the number of cascades
    rewritten. Scope-level requests (``mux_emission`` regions/decorators) are
    eager and independent of this pass."""
    if not cfg.enabled:
        return 0

    limit = sys.getrecursionlimit()
    if limit < 20000:
        sys.setrecursionlimit(20000)

    n_changed = 0
    visited: set = set()

    def try_rewrite(e: Expr) -> Expr:
        nonlocal n_changed
        new_e = rewrite(e, None, cfg)  # mode None = auto: "chain" -> unchanged
        if new_e is not e:
            n_changed += 1
        return new_e

    def walk(e: Expr) -> Expr:
        if id(e) in visited:
            return e
        visited.add(id(e))
        if isinstance(e, Ternary):
            spine: set = set()
            pairs, default = collect_chain(e, spine=spine)
            if len(pairs) >= 2:
                visited.update(spine)  # the whole cascade is handled here
                # walk leaves first so nested cascades inside arm values get
                # their own treatment
                for s, v in pairs:
                    walk(s)
                    walk(v)
                walk(default)
                return try_rewrite(e)
            e.sel = walk(e.sel)
            e.a = walk(e.a)
            e.b = walk(e.b)
        elif isinstance(e, Op1):
            e.a = walk(e.a)
        elif isinstance(e, Op2):
            e.a = walk(e.a)
            e.b = walk(e.b)
        elif isinstance(e, Concat):
            e.parts = [walk(p) for p in e.parts]
        elif isinstance(e, (Slice, Resize)):
            e.a = walk(e.a)
        elif isinstance(e, Signal):
            if getattr(e, "_auto_generated", False) and e._driver is not None:
                nd = walk(e._driver)
                if nd is not e._driver:
                    e._driver = nd
        return e

    for s in module._signals:
        drv = getattr(s, "_driver", None)
        if not isinstance(drv, Expr):
            continue
        nd = walk(drv)
        if nd is not drv:
            s._driver = nd

    return n_changed
