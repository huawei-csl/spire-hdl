"""Selection-cascade emission rewrites: chain / tournament / onehot / bittree.

Spire lowers ``switch_``/``case_``, ``if_``/``elif_`` and hand-written nested
``mux()`` calls to *linear chains* of binary ``Ternary`` nodes at construction
time. Downstream synthesis (yosys + ABC) cannot rebalance those chains: a mux
chain is not associative, and the mutual exclusivity of case arms — which lets
a Verilog ``case`` lower to a flat parallel ``$pmux`` network — is erased by
the lowering. The result is O(N) logic depth where O(log N) is available
(measured on the CV32E40P ALU port: find-first-one chain 31 vs 9 levels at
identical area; full design 9342 AND / 91 levels vs 8901 / 79 restructured).

One concept — *set the emission style for a scope* — at three granularities:

  * region:        ``with selection_topology("onehot"): ...`` — captures every signal
                    assigned inside (switch_/if_ arms and hand-built chains
                    alike) and eagerly rewrites their final cascades on exit;
  * function:      ``@selection_topology("tournament")`` — the same object as a
                    decorator; eagerly rewrites the returned cascade;
  * whole design:  ``to_verilog_file(..., selection_emission=True)`` — the
                    :func:`apply_selection_emission` pass auto-detects untagged
                    cascades above the config thresholds.

Region and function forms are **eager**: the rewrite is baked into the
expression graph at construction, so every backend (Verilog, AIGER export,
Simulator, analyze) sees the same structure.

Modes and their prerequisites (validation is *shape-based* — the analyzer in
:mod:`spire.selection_analysis` judges the final cascade, never the construct
it came from, so hand-built chains and constructs are treated identically):

  * ``"chain"``      — leave as-is. O(N) depth. Always legal.
  * ``"tournament"`` — parallel-prefix first-match tree, node
                       ``(sl | sr, mux(sl, vl, vr))``. Preserves priority
                       universally, O(log N) depth. Always legal.
  * ``"onehot"``     — one-hot AND-mask + balanced OR (the ``$pmux`` form).
                       Requires provably disjoint arm selects: ``sel == const``
                       terms (or ORs of them) on one selector with pairwise-
                       distinct constants. Redundant first-match gating
                       (``cond & ~covered``) is seen through when provably
                       dead, so eq-const ``if_``/``elif_`` chains qualify too.
  * ``"bittree"``    — balanced mux tree indexed by the *selector bits* (the
                       arm compares vanish). Missing labels fill from the
                       fallback (the chain tail or the ``default()`` value);
                       selector width is capped by ``bittree_max_sel_bits``.
  * ``"auto"``       — best legal form per cascade, subject to the config
                       thresholds; never raises.

See rtl_scout's metadocuments/spire_selection_emission.md for measurements and
the applicability table.
"""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from spire.control_structures import _ConditionState, fresh_condition_scope
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
from spire.selection_analysis import (
    ChainAnalysis,
    _deref,
    analyze_chain,
    collect_chain,
)

MODES = ("chain", "tournament", "onehot", "bittree")
REGION_MODES = MODES + ("auto",)


@dataclass
class SelectionEmissionConfig:
    enabled: bool = False        # auto-detect untagged chains in the pass
    onehot_min_n: int = 16       # auto: disjoint cascades at/above this -> onehot
    tournament_min_n: int = 16   # auto: priority cascades at/above this -> tournament
    bittree_max_sel_bits: int = 6  # refuse bittree beyond 2**6 = 64 leaves


DEFAULT_CONFIG = SelectionEmissionConfig()


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


def build_onehot(analysis: ChainAnalysis, out_typ) -> Expr:
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
            f"limit of 2**{max_sel_bits}; use 'onehot' for sparse label sets")
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
    if analysis.disjoint and analysis.n >= cfg.onehot_min_n:
        return "onehot"
    if analysis.n >= cfg.tournament_min_n:
        return "tournament"
    return "chain"


def rewrite(head: Expr, mode: Optional[str] = None,
            cfg: SelectionEmissionConfig = DEFAULT_CONFIG) -> Expr:
    """Rewrite the priority cascade headed at ``head`` into ``mode``
    (None/"auto" = pick via ``choose_mode``). Returns ``head`` unchanged when
    the mode resolves to "chain" or no cascade is found under an auto mode.
    Raises ValueError when an explicitly requested mode's prerequisites don't
    hold (shape-based: provable disjointness for "onehot"/"bittree")."""
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
    elif mode in ("onehot", "bittree") and not analysis.disjoint:
        raise ValueError(
            f"emission mode {mode!r} requires provably disjoint arm selects — "
            f"the analyzer recognizes pairwise-distinct constant labels on a "
            f"single selector (the only disjointness it can prove); this "
            f"cascade's selects don't qualify (use 'tournament', which "
            f"preserves priority universally)")

    if mode == "chain":
        return head_e
    if mode == "tournament":
        return build_tournament(pairs, default)
    if mode == "onehot":
        return build_onehot(analysis, out_typ)
    return build_bittree(analysis, out_typ, cfg.bittree_max_sel_bits)


# ---------------------------------------------------------------------------
# the user-facing scope object: `selection_topology`
# ---------------------------------------------------------------------------

class selection_topology:
    """Set the selection-emission style for a scope — one object, two roles.

    Region (context manager)::

        with selection_topology("onehot"):
            with switch_(op):
                with case_(A, B): y <<= ...

        with selection_topology("tournament"):
            with if_(c0):   y <<= 1
            with elif_(c1): y <<= 2

        with selection_topology("tournament"):
            y <<= hand_built_mux_chain      # hand chains count too

    Every signal assigned inside the region is captured by the *innermost*
    active region; on exit each captured signal's final selection cascade is
    eagerly rewritten to ``mode``. Signals without a cascade (plain
    assignments) are skipped. Explicit one-hot modes ("onehot"/"bittree")
    raise at region exit if a captured cascade cannot provably satisfy them;
    "auto" picks the best legal form per cascade and never raises.
    An if_/elif_ chain may not straddle the region boundary (the region is
    part of the chain-claim scope, so a straddling elif_ fails loudly).

    Caveat: the rewrite covers a captured signal's ENTIRE final cascade —
    conditional arms assigned *before* the region count too, and can fail a
    one-hot proof the in-region arms alone would pass. Workaround: give the
    signal an unconditional driver before the region (it becomes the chain
    tail, which needs no proof). A proper fix exists but is not implemented:
    snapshot the driver at a signal's first in-region assignment and stop
    the rewrite there.

    Function (decorator)::

        @selection_topology("tournament")
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
                f"selection_topology: unknown mode {mode!r}; expected one of {REGION_MODES}")
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

    def __enter__(self) -> "selection_topology":
        if self._entered:
            raise RuntimeError("selection_topology region cannot be re-entered while active")
        self._signals, self._seen_ids = [], set()
        _ConditionState.selection_topology_stack.append(self)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        stack = _ConditionState.selection_topology_stack
        if not self._entered or not stack or stack[-1] is not self:
            raise RuntimeError("selection_topology region stack corruption detected")
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
            raise ValueError(
                f"selection_topology: signal '{signal_name}': {err} (note: the rewrite covers the "
                f"signal's entire final cascade — pre-region conditional arms count too)") from None
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
    rewritten. Scope-level requests (``selection_topology`` regions/decorators) are
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
