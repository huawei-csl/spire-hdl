"""Balanced reduction trees — log-depth folds of associative operators.

Reductions written as linear loops (``acc = mux(x > acc, x, acc)``) are O(N)
deep, and downstream synthesis cannot rebalance them: no netlist cell
represents word-level max/min, so the associativity is invisible after
lowering (measured on a 20-input max: 171 vs 45 gate levels). These helpers
build the balanced structure directly — stating the intent beats recognizing it.

Topologies (``topology=`` on each helper):

  * ``"tree"``      — balanced binary tree, depth O(log N). Default.
  * ``"chain"``     — serial left fold, depth O(N). Baseline/reference.
  * ``"huffman"``   — arrival-aware tree: shallow inputs merge first (unit-cost
                      expression-depth heuristic). Reorders operands, so ``fn``
                      must also be commutative; not offered for argmax/argmin.
  * ``"matrix"``    — max/min/argmax/argmin only: all N(N-1)/2 comparisons in
                      parallel, one-hot winner select. Shallowest, O(N^2) area.

``prefix_scan`` returns all N running prefixes at O(log N) depth
(``"sklansky"`` / ``"brentkung"`` / ``"koggestone"``) — use it when partial
results are tapped, instead of duplicating a chain.

All order-preserving forms pair left-to-right; ``argmax_``/``argmin_`` resolve
ties to the leftmost element in every topology. Integer/bitwise only.
See docs/README_reductions.md and metadocs/REDUCTION_TOPOLOGY_PLAN.md.
"""

from __future__ import annotations

import functools
import heapq
from typing import Callable, Dict, List, Sequence, Tuple

from spire.expr import (Concat, Const, Expr, ExprLike, Op1, Op2, Resize,
                        Signal, Slice, Ternary, UInt, as_expr, cat, mux)

_FOLDS = ("tree", "chain", "huffman")
_SCANS = ("sklansky", "brentkung", "koggestone")


def reduce_tree(fn: Callable[[Expr, Expr], Expr], items: Sequence[ExprLike],
                topology: str = "tree") -> Expr:
    """Fold ``items`` with binary ``fn``. ``fn`` must be associative (and for
    ``"huffman"`` commutative) — the caller asserts this, it is not checked."""
    if not items:
        raise ValueError("reduce_tree() requires at least one item")
    if topology not in _FOLDS:
        raise ValueError(f"unknown fold topology {topology!r}; expected one of {_FOLDS}")
    layer: List[Expr] = [as_expr(x) for x in items]
    if topology == "chain":
        return functools.reduce(fn, layer)
    if topology == "huffman":
        return _huffman(fn, layer)
    while len(layer) > 1:
        layer = [fn(layer[i], layer[i + 1]) if i + 1 < len(layer) else layer[i]
                 for i in range(0, len(layer), 2)]
    return layer[0]


def max_(items: Sequence[ExprLike], topology: str = "tree") -> Expr:
    """Max of ``items``."""
    if topology == "matrix":
        return _matrix_select(items, _ge)[0]
    return reduce_tree(lambda a, b: mux(a >= b, a, b), items, topology)


def min_(items: Sequence[ExprLike], topology: str = "tree") -> Expr:
    """Min of ``items``."""
    if topology == "matrix":
        return _matrix_select(items, _le)[0]
    return reduce_tree(lambda a, b: mux(a <= b, a, b), items, topology)


def sum_(items: Sequence[ExprLike], topology: str = "tree") -> Expr:
    """Adder tree (widths widen per level; slice/fit the result)."""
    return reduce_tree(lambda a, b: a + b, items, topology)


def prod_(items: Sequence[ExprLike], topology: str = "tree") -> Expr:
    """Multiplier tree."""
    return reduce_tree(lambda a, b: a * b, items, topology)


def clamp_(x: ExprLike, lo: ExprLike, hi: ExprLike) -> Expr:
    """Clamp ``x`` into [lo, hi] (assumes lo <= hi)."""
    x = as_expr(x)
    m = mux(x >= as_expr(lo), x, lo)
    return mux(m <= as_expr(hi), m, hi)


def argmax_(items: Sequence[ExprLike], topology: str = "tree") -> Tuple[Expr, Expr]:
    """(value, index) of the max; leftmost of equal values wins."""
    return _arg_reduce(items, _ge, topology)


def argmin_(items: Sequence[ExprLike], topology: str = "tree") -> Tuple[Expr, Expr]:
    """(value, index) of the min; leftmost of equal values wins."""
    return _arg_reduce(items, _le, topology)


def prefix_scan(fn: Callable[[Expr, Expr], Expr], items: Sequence[ExprLike],
                topology: str = "sklansky") -> List[Expr]:
    """All N inclusive left-to-right prefixes of ``fn`` at O(log N) depth.
    ``"sklansky"``: N/2·logN ops, minimal depth, high fanout. ``"brentkung"``:
    ~2N ops, 2·logN depth. ``"koggestone"``: N·logN ops, minimal depth and
    fanout. Order is preserved, so ``fn`` only needs associativity."""
    if not items:
        raise ValueError("prefix_scan() requires at least one item")
    if topology not in _SCANS:
        raise ValueError(f"unknown scan topology {topology!r}; expected one of {_SCANS}")
    xs = [as_expr(x) for x in items]
    if topology == "koggestone":
        d = 1
        while d < len(xs):
            xs = [xs[i] if i < d else fn(xs[i - d], xs[i]) for i in range(len(xs))]
            d *= 2
        return xs
    if topology == "brentkung":
        return _brent_kung(fn, xs)
    return _sklansky(fn, xs)


# -- internals ---------------------------------------------------------------

def _sklansky(fn, xs: List[Expr]) -> List[Expr]:
    if len(xs) == 1:
        return xs
    m = (len(xs) + 1) // 2
    left, right = _sklansky(fn, xs[:m]), _sklansky(fn, xs[m:])
    total = left[-1]
    return left + [fn(total, r) for r in right]


def _brent_kung(fn, xs: List[Expr]) -> List[Expr]:
    n = len(xs)
    if n == 1:
        return xs
    pairs = [fn(xs[i], xs[i + 1]) if i + 1 < n else xs[i] for i in range(0, n, 2)]
    ps = _brent_kung(fn, pairs)  # prefixes of the pair sums
    return [xs[0] if i == 0 else ps[i // 2] if i % 2 else fn(ps[i // 2 - 1], xs[i])
            for i in range(n)]


def _ge(a: Expr, b: Expr) -> Expr:
    return a >= b


def _le(a: Expr, b: Expr) -> Expr:
    return a <= b


def _huffman(fn, layer: List[Expr]) -> Expr:
    heap = [(_expr_depth(e), i, e) for i, e in enumerate(layer)]
    heapq.heapify(heap)
    seq = len(layer)
    while len(heap) > 1:
        d1, _, a = heapq.heappop(heap)
        d2, _, b = heapq.heappop(heap)
        heapq.heappush(heap, (max(d1, d2) + 1, seq, fn(a, b)))
        seq += 1
    return heap[0][2]


def _expr_depth(e: Expr, memo: Dict[int, int] = None) -> int:
    """Unit-cost depth of the expression cone (registers and inputs are 0)."""
    if memo is None:
        memo = {}
    if id(e) in memo:
        return memo[id(e)]
    if isinstance(e, (Op1, Slice, Resize)):
        d = 1 + _expr_depth(e.a, memo)
    elif isinstance(e, Op2):
        d = 1 + max(_expr_depth(e.a, memo), _expr_depth(e.b, memo))
    elif isinstance(e, Ternary):
        d = 1 + max(_expr_depth(e.sel, memo), _expr_depth(e.a, memo),
                    _expr_depth(e.b, memo))
    elif isinstance(e, Concat):
        d = max((_expr_depth(p, memo) for p in e.parts), default=0)
    elif isinstance(e, Signal) and e._driver is not None and e.kind != "reg":
        d = _expr_depth(e._driver, memo)
    else:
        d = 0
    memo[id(e)] = d
    return d


def _arg_reduce(items, better, topology) -> Tuple[Expr, Expr]:
    if not items:
        raise ValueError("argmax_/argmin_ require at least one item")
    if topology == "huffman":
        raise ValueError("argmax_/argmin_ do not support 'huffman': it reorders "
                         "elements, breaking the leftmost-wins tie rule")
    if topology == "matrix":
        return _matrix_select(items, better)
    if topology not in ("tree", "chain"):
        raise ValueError(f"unknown argmax/argmin topology {topology!r}")
    iw = max(1, (len(items) - 1).bit_length())
    layer = [(as_expr(x), Const(i, UInt(iw))) for i, x in enumerate(items)]
    if topology == "chain":
        return functools.reduce(lambda l, r: _arg_node(l, r, better), layer)
    while len(layer) > 1:
        nxt = [_arg_node(layer[i], layer[i + 1], better) if i + 1 < len(layer)
               else layer[i] for i in range(0, len(layer), 2)]
        layer = nxt
    return layer[0]


def _arg_node(left, right, better):
    (lv, li), (rv, ri) = left, right
    keep = better(lv, rv)  # ties keep the lower (leftmost) index
    return (mux(keep, lv, rv), mux(keep, li, ri))


def _matrix_select(items, better) -> Tuple[Expr, Expr]:
    """All-pairs compare, one-hot winner: (value, index). ``better(a, b)`` with
    a left of b must be true on ties, making exactly one winner (leftmost)."""
    if not items:
        raise ValueError("matrix topology requires at least one item")
    xs = [as_expr(x) for x in items]
    n = len(xs)
    g = {(i, j): better(xs[i], xs[j]) for i in range(n) for j in range(i + 1, n)}
    wins = []
    for i in range(n):
        terms = [~g[(k, i)] for k in range(i)] + [g[(i, j)] for j in range(i + 1, n)]
        wins.append(reduce_tree(lambda a, b: a & b, terms) if terms else as_expr(True))
    iw = max(1, (n - 1).bit_length())
    value = _onehot_pick(wins, xs, xs[0].typ.width)
    index = _onehot_pick(wins, [Const(i, UInt(iw)) for i in range(n)], iw)
    return value, index


def _onehot_pick(flags, values, width):
    rep = (lambda f: cat(*([f] * width))) if width > 1 else (lambda f: f)
    return reduce_tree(lambda a, b: a | b, [rep(f) & v for f, v in zip(flags, values)])
