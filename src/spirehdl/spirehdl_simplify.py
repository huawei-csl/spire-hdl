"""Post-construction expression simplification — analogous to yosys ``opt_expr`` / ``opt_muxtree``.

Walks every driver-attached Expr DAG bottom-up and applies local rewrites:

1. **Constant folding** — ``Const ⊕ Const → Const(folded)`` for the bit-logic, arithmetic, and comparison operators.
   Width-masked so widening / wraparound matches Verilog semantics.

2. **Boolean / arithmetic identities** — ``x|0→x``, ``x|1→1``, ``x|x→x``; ``x&0→0``, ``x&1→x``, ``x&x→x``;
   ``x^0→x``, ``x^x→0``; ``x+0→x``, ``x-0→x``, ``x*0→0``, ``x*1→x``; ``~~x→x``; ``~Const → Const``.

3. **Trivial mux** — ``mux(Const(1), a, b) → a``; ``mux(Const(0), a, b) → b``; ``mux(c, x, x) → x`` (structural
   equality on x).

4. **Mux-tree guard substitution** — ``mux(g, mux(g, A, B), F) → mux(g, A, F)`` and the symmetric
   ``mux(g, T, mux(g, A, B)) → mux(g, T, B)``. Catches the redundant-inner-mux pattern that motivates yosys's
   ``opt_muxtree`` pass.

This is structural / syntactic only — no SAT, no full symbolic reasoning. The guard substitution requires the inner
and outer mux selectors to be **structurally identical** (same canonical key), so it catches cases like
``mux(x, mux(x, …), F)`` but not ``mux(x, mux(x|sel, …), F)`` directly. The latter reduces in two iterations: first
constant-folding through a substituted guard (only fires if the inner guard's operands become Const), then the mux
collapse. For the case where the inner guard is a syntactically different expression but provably the same value
under the outer guard, the user (or a fixed-point loop) can repeat the pass.

Returns the number of nodes rewritten — useful for callers that want to know whether this pass changed the design.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from spirehdl.spirehdl import (
    Concat,
    Const,
    Expr,
    HDLType,
    Op1,
    Op2,
    Resize,
    Signal,
    Slice,
    Ternary,
)
from spirehdl.spirehdl_visitor import ExprVisitor


_SYMMETRIC_OPS = frozenset({"&", "|", "^", "==", "!=", "nand"})


# ---------------------------------------------------------------------------
# Canonical-key walker for structural equality
# ---------------------------------------------------------------------------

class _KeyWalker(ExprVisitor[tuple]):
    """Compute a canonical key per Expr. Two Exprs with the same key are structurally equivalent.

    Mirrors ``spirehdl_cse._CseWalker`` but is a separate instance because mux-tree opt may construct new Ternary
    nodes whose keys we want to compute on demand.
    """

    def visit_const(self, e: Const) -> tuple:
        return ("const", e.value, e.typ.width, e.typ.signed)

    def visit_signal(self, e: Signal) -> tuple:
        # Auto-shared wires (created by ``_maybe_share``) are transparent for the purposes of structural equality —
        # their key is the key of their driver. User-named Signals stay opaque (keyed by Python id) so that the pass
        # never equates e.g. two distinct user inputs that happen to be wired to the same expression elsewhere.
        if getattr(e, "_auto_generated", False) and e._driver is not None:
            return self.visit(e._driver)
        return ("sig", id(e))

    def visit_op1(self, e: Op1) -> tuple:
        return ("op1", e.op, self.visit(e.a), e.typ.width, e.typ.signed)

    def visit_op2(self, e: Op2) -> tuple:
        ka, kb = self.visit(e.a), self.visit(e.b)
        if e.op in _SYMMETRIC_OPS and ka > kb:
            ka, kb = kb, ka
        return ("op2", e.op, ka, kb, e.typ.width, e.typ.signed)

    def visit_ternary(self, e: Ternary) -> tuple:
        return ("tern", self.visit(e.sel), self.visit(e.a), self.visit(e.b),
                e.typ.width, e.typ.signed)

    def visit_concat(self, e: Concat) -> tuple:
        return ("cat", tuple(self.visit(p) for p in e.parts))

    def visit_slice(self, e: Slice) -> tuple:
        return ("slice", self.visit(e.a), e.start, e.msb)

    def visit_resize(self, e: Resize) -> tuple:
        return ("resize", self.visit(e.a), e.to_width, e.typ.signed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask(width: int) -> int:
    """Bit-mask for `width` bits — 0xFF for 8, 0xFFFF for 16, etc."""
    return (1 << width) - 1 if width > 0 else 0


def _through(e: Expr) -> Expr:
    """Follow auto-shared (sig_N) wires to their driver. Stops at user-named Signals.

    ``_maybe_share`` wraps every fresh ``Op``-typed expression in an auto-generated wire on first use, so by the time
    our peepholes see a tree node's children most non-leaf expressions are actually ``Signal(_auto_generated=True)``
    whose ``_driver`` is the real expression. Looking through them lets the same peephole logic catch
    ``Resize(sig_0)`` where ``sig_0._driver`` simplified to a ``Const``, without us having to worry about iteration
    order or run a fixed-point loop.
    """
    while isinstance(e, Signal) and getattr(e, "_auto_generated", False) \
            and e._driver is not None:
        e = e._driver
    return e


def _is_const_value(e: Expr, v: int) -> bool:
    e = _through(e)
    return isinstance(e, Const) and e.value == v


def _is_const_all_ones(e: Expr) -> bool:
    e = _through(e)
    return isinstance(e, Const) and e.value == _mask(e.typ.width)


def _fold_op2(a: Const, b: Const, op: str, typ: HDLType) -> Expr:
    """Constant-fold a binary op. Returns a Const, or the original Op2 if op is unhandled."""
    av, bv = a.value, b.value
    m = _mask(typ.width)
    if op == "&":   return Const(av & bv, typ)
    if op == "|":   return Const(av | bv, typ)
    if op == "^":   return Const(av ^ bv, typ)
    if op == "nand": return Const(m ^ (av & bv), typ)
    if op == "+":   return Const((av + bv) & m, typ)
    if op == "-":   return Const((av - bv) & m, typ)
    if op == "*":   return Const((av * bv) & m, typ)
    if op == "==":  return Const(1 if av == bv else 0, typ)
    if op == "!=":  return Const(0 if av == bv else 1, typ)
    if op == "<":   return Const(1 if av <  bv else 0, typ)
    if op == "<=":  return Const(1 if av <= bv else 0, typ)
    if op == ">":   return Const(1 if av >  bv else 0, typ)
    if op == ">=":  return Const(1 if av >= bv else 0, typ)
    if op == "<<":  return Const((av << bv) & m, typ)
    if op == ">>":  return Const((av >> bv) & m, typ)
    return Op2(a, b, op, typ)


# ---------------------------------------------------------------------------
# Per-node simplifiers
# ---------------------------------------------------------------------------

def _simplify_op1(e: Op1) -> Expr:
    """Op1 peepholes: const fold, double-negation."""
    a = _through(e.a)
    if isinstance(a, Const) and e.op == "~":
        return Const(_mask(e.typ.width) ^ a.value, e.typ)
    if e.op == "~" and isinstance(a, Op1) and a.op == "~":
        return _through(a.a)
    return e


def _simplify_op2(e: Op2, key_of: Callable[[Expr], tuple]) -> Expr:
    """Op2 peepholes: const fold + boolean/arithmetic identity rules.

    We pass ``e.a`` / ``e.b`` through ``_through`` for Const-and-identity checks but return either a Const literal or
    one of ``e.a`` / ``e.b`` verbatim (preserving the auto-shared wire if any) when the rule fires — that keeps the
    wire references in parent nodes pointing at the same Signal across before/after the pass and avoids surprising
    CSE downstream.
    """
    op, typ = e.op, e.typ
    a_t, b_t = _through(e.a), _through(e.b)

    if isinstance(a_t, Const) and isinstance(b_t, Const):
        return _fold_op2(a_t, b_t, op, typ)

    if op == "|":
        if _is_const_value(a_t, 0): return e.b
        if _is_const_value(b_t, 0): return e.a
        if _is_const_all_ones(a_t): return e.a
        if _is_const_all_ones(b_t): return e.b
        if key_of(e.a) == key_of(e.b): return e.a
    elif op == "&":
        if _is_const_value(a_t, 0): return e.a
        if _is_const_value(b_t, 0): return e.b
        if _is_const_all_ones(a_t): return e.b
        if _is_const_all_ones(b_t): return e.a
        if key_of(e.a) == key_of(e.b): return e.a
    elif op == "^":
        if _is_const_value(a_t, 0): return e.b
        if _is_const_value(b_t, 0): return e.a
        if key_of(e.a) == key_of(e.b): return Const(0, typ)
    elif op == "+":
        if _is_const_value(a_t, 0): return e.b
        if _is_const_value(b_t, 0): return e.a
    elif op == "-":
        if _is_const_value(b_t, 0): return e.a
        if key_of(e.a) == key_of(e.b): return Const(0, typ)
    elif op == "*":
        if _is_const_value(a_t, 0) or _is_const_value(b_t, 0):
            return Const(0, typ)
        if _is_const_value(a_t, 1): return e.b
        if _is_const_value(b_t, 1): return e.a
    elif op == "==":
        if key_of(e.a) == key_of(e.b): return Const(1, typ)
    elif op == "!=":
        if key_of(e.a) == key_of(e.b): return Const(0, typ)
    return e


def _simplify_resize(e: Resize) -> Expr:
    """Resize peepholes: no-op resize is identity, Const operand re-types in place."""
    a_t = _through(e.a)
    if a_t.typ.width == e.to_width and a_t.typ.signed == e.typ.signed:
        return a_t
    if isinstance(a_t, Const):
        new_typ = HDLType(e.to_width, signed=a_t.typ.signed,
                          is_bool=(e.to_width == 1))
        return Const(a_t.value & _mask(e.to_width), new_typ)
    return e


def _simplify_slice(e: Slice) -> Expr:
    """Slice peepholes: full-width slice is identity, Const operand evaluates the bit-select."""
    a_t = _through(e.a)
    if e.start == 0 and e.typ.width == a_t.typ.width:
        return a_t
    if isinstance(a_t, Const):
        extracted = (a_t.value >> e.start) & _mask(e.typ.width)
        return Const(extracted, e.typ)
    return e


def _substitute(expr: Expr, key_to_val: Dict[tuple, Const],
                key_of: Callable[[Expr], tuple]) -> Expr:
    """Return a tree with all subtrees whose canonical key is in ``key_to_val`` replaced by the corresponding
    ``Const``. Rebuilt nodes go through the per-type simplifier so any newly-exposed peephole rule fires automatically.

    Only used by the guard-substitution path in ``_simplify_ternary`` — under the outer mux's true-side we substitute
    ``sel → Const(1)`` into the inner expression, then resimplify; under the false-side we substitute ``sel → 0``.
    """
    if not key_to_val:
        return expr

    k = key_of(expr)
    if k in key_to_val:
        return key_to_val[k]

    if isinstance(expr, Const):
        return expr

    if isinstance(expr, Signal):
        # Auto-shared wires are transparent — substitute into their driver. User-named Signals stay opaque (we don't
        # peek inside named Wire/Register definitions).
        if getattr(expr, "_auto_generated", False) and expr._driver is not None:
            new_driver = _substitute(expr._driver, key_to_val, key_of)
            return new_driver if new_driver is not expr._driver else expr
        return expr

    if isinstance(expr, Op1):
        new_a = _substitute(expr.a, key_to_val, key_of)
        if new_a is expr.a:
            return expr
        return _simplify_op1(Op1(new_a, expr.op, expr.typ))

    if isinstance(expr, Op2):
        new_a = _substitute(expr.a, key_to_val, key_of)
        new_b = _substitute(expr.b, key_to_val, key_of)
        if new_a is expr.a and new_b is expr.b:
            return expr
        return _simplify_op2(Op2(new_a, new_b, expr.op, expr.typ), key_of)

    if isinstance(expr, Ternary):
        new_sel = _substitute(expr.sel, key_to_val, key_of)
        new_a = _substitute(expr.a, key_to_val, key_of)
        new_b = _substitute(expr.b, key_to_val, key_of)
        if new_sel is expr.sel and new_a is expr.a and new_b is expr.b:
            return expr
        return _simplify_ternary(Ternary(new_sel, new_a, new_b), key_of)

    if isinstance(expr, Concat):
        new_parts = [_substitute(p, key_to_val, key_of) for p in expr.parts]
        if all(np is op for np, op in zip(new_parts, expr.parts)):
            return expr
        return Concat(new_parts)

    if isinstance(expr, Slice):
        new_a = _substitute(expr.a, key_to_val, key_of)
        if new_a is expr.a:
            return expr
        return _simplify_slice(Slice(new_a, expr.start, expr.msb + 1))

    if isinstance(expr, Resize):
        new_a = _substitute(expr.a, key_to_val, key_of)
        if new_a is expr.a:
            return expr
        return _simplify_resize(Resize(new_a, expr.to_width))

    return expr


def _simplify_ternary(e: Ternary, key_of: Callable[[Expr], tuple]) -> Expr:
    """Ternary peepholes: const selector, equal branches, mux-tree guard substitution.

    The guard-substitution rules are:

    - ``mux(g, mux(g, A, B), F) → mux(g, A, F)`` — when the outer mux picks the true-branch, ``g`` is also true for
      the inner mux, which picks its own A.

    - ``mux(g, T, mux(g, A, B)) → mux(g, T, B)`` — symmetric for the false side.

    After substitution we re-check the trivial-mux peepholes in case the rewrite exposed them.
    """
    sel_t = _through(e.sel)
    a_t, b_t = _through(e.a), _through(e.b)

    # Constant selector — pick the surviving branch.
    if isinstance(sel_t, Const):
        return e.a if sel_t.value != 0 else e.b

    # Equal branches — both sides emit the same value, so the selector is irrelevant.
    if key_of(e.a) == key_of(e.b):
        return e.a

    # Guard substitution: look through auto-shared wires to discover nested Ternary.
    sel_key = key_of(e.sel)
    new_a = e.a
    new_b = e.b
    if isinstance(a_t, Ternary) and key_of(a_t.sel) == sel_key:
        new_a = a_t.a
    if isinstance(b_t, Ternary) and key_of(b_t.sel) == sel_key:
        new_b = b_t.b

    # Symbolic guard substitution: under the outer mux's true-side, every reference to ``sel`` is provably 1;
    # substitute and re-simplify. Symmetric under false-side with 0. This handles cases like
    # ``mux(x, mux(x|sel, A, B), F)`` — under x=1 the inner guard ``x|sel`` folds to ``Const(1)``, collapsing the
    # inner mux to ``A``. Only safe to do when sel is 1-bit (the only case where Const(0)/Const(1) values are
    # unambiguous regardless of signed/unsigned interpretation).
    if e.sel.typ.width == 1:
        if isinstance(a_t, Ternary):
            sub_a = _substitute(new_a, {sel_key: Const(1, e.sel.typ)}, key_of)
            if sub_a is not new_a:
                new_a = sub_a
        if isinstance(b_t, Ternary):
            sub_b = _substitute(new_b, {sel_key: Const(0, e.sel.typ)}, key_of)
            if sub_b is not new_b:
                new_b = sub_b

    if new_a is e.a and new_b is e.b:
        return e

    if key_of(new_a) == key_of(new_b):
        return new_a

    return Ternary(e.sel, new_a, new_b)


# ---------------------------------------------------------------------------
# Bottom-up DAG rewriter
# ---------------------------------------------------------------------------

_MAX_ITERS = 8


def apply_simplify(module) -> int:
    """Run the peephole pass until fixpoint. Returns the total number of nodes rewritten across all iterations
    (useful for callers that want to skip downstream work when nothing changed).

    A fixpoint loop is necessary because user signal drivers can wrap auto-shared wires (created by ``_maybe_share``)
    whose own drivers get simplified mid-pass — the enclosing Resize/Slice/Op then sees a Const child only on the next
    iteration. In practice 2–3 iterations are plenty.
    """
    total = 0
    for _ in range(_MAX_ITERS):
        n = _apply_simplify_once(module)
        total += n
        if n == 0:
            break
    return total


def _apply_simplify_once(module) -> int:
    """One pass of the peephole rewriter — bottom-up DAG walk with in-place rewrites."""
    roots: List[Expr] = []
    for s in module._signals:
        drv = getattr(s, "_driver", None)
        if isinstance(drv, Expr):
            roots.append(drv)
    if not roots:
        return 0

    keyer = _KeyWalker()

    def key_of(e: Expr) -> tuple:
        return keyer.visit(e)

    # Pre-warm the canonical-key cache so structural-equality checks are O(1).
    for r in roots:
        keyer.visit(r)

    replaced: Dict[int, Expr] = {}
    visiting: set = set()
    n_changed = 0

    def simp(e: Expr) -> Expr:
        nonlocal n_changed
        eid = id(e)
        if eid in replaced:
            return replaced[eid]
        if eid in visiting:
            # Cycle guard — should not happen on a DAG, but be defensive.
            return e
        visiting.add(eid)

        new_e: Expr = e
        if isinstance(e, Op1):
            e.a = simp(e.a)
            new_e = _simplify_op1(e)
        elif isinstance(e, Op2):
            e.a = simp(e.a)
            e.b = simp(e.b)
            new_e = _simplify_op2(e, key_of)
        elif isinstance(e, Ternary):
            e.sel = simp(e.sel)
            e.a = simp(e.a)
            e.b = simp(e.b)
            new_e = _simplify_ternary(e, key_of)
        elif isinstance(e, Concat):
            e.parts = [simp(p) for p in e.parts]
        elif isinstance(e, Slice):
            e.a = simp(e.a)
            new_e = _simplify_slice(e)
        elif isinstance(e, Resize):
            e.a = simp(e.a)
            new_e = _simplify_resize(e)
        # Const / Signal: leaves, nothing to recurse into.

        visiting.discard(eid)
        if new_e is not e:
            n_changed += 1
        replaced[eid] = new_e
        return new_e

    # Rewrite each Signal's driver.
    for s in module._signals:
        drv = getattr(s, "_driver", None)
        if isinstance(drv, Expr):
            new_drv = simp(drv)
            if new_drv is not drv:
                s._driver = new_drv

    return n_changed


# ---------------------------------------------------------------------------
# Mux-tree balance pass: detect the linear-cascade lookup anti-pattern
#   mux(sel == Const(0), v_0,
#     mux(sel == Const(1), v_1,
#       ...
#         mux(sel == Const(N-1), v_{N-1}, default)))
#
# and replace with a balanced binary mux tree using BITS of `sel`:
#   leaves = [v_0, v_1, ..., v_{N-1}]
#   for bit in range(log2(N)):
#       leaves = [mux(sel[bit], leaves[i+1], leaves[i]) for i in range(0, N, 2)]
#   return leaves[0]
#
# The cascade form (N comparators + N-deep mux chain) tends to synthesise to
# a deep AOI/OAI chain under yosys+abc, whereas the bit-tree form maps cleanly
# to log2(N) levels of native MUX2 cells. The win is largest for N ≥ 16 (on
# smaller N, yosys-abc usually finds an optimal MUX2/MUX4 structure on its own
# — forcing the bit tree there can even regress delay).
#
# We require:
#   1. Cascade length ≥ min_n (default 16).
#   2. All `==` guards reference the SAME sel signal.
#   3. The key values cover [0, 2^K) exactly once for K = sel.typ.width
#      (full-coverage power-of-2). The cascade's innermost else-branch
#      ("default") is then never taken and can be dropped.
#
# Returns the number of cascades rewritten.
# ---------------------------------------------------------------------------


def _eq_const_info(e: Expr, key_of: Callable[[Expr], tuple]):
    """If `e` is structurally `signal == Const(k)`, return (signal, k, sel_key).

    Handles `Const(k) == signal` symmetrically. Returns None otherwise.
    """
    e = _through(e)
    if not isinstance(e, Op2) or e.op != "==":
        return None
    a, b = _through(e.a), _through(e.b)
    if isinstance(b, Const) and not isinstance(a, Const):
        return (e.a, b.value, key_of(e.a))
    if isinstance(a, Const) and not isinstance(b, Const):
        return (e.b, a.value, key_of(e.b))
    return None


def _collect_cascade(root: Expr, key_of: Callable[[Expr], tuple]):
    """Walk down a Ternary chain collecting (k, value) pairs where every
    selector is `sel == Const(k)` for the SAME `sel`.

    Returns (sel_signal, [(k_0, v_0), (k_1, v_1), ...], default, depth_consumed).
    `depth_consumed` is the number of Ternary nodes consumed; if < 2 the caller
    should consider the cascade not worth rewriting.
    """
    e = root
    pairs: List = []
    sel_signal = None
    sel_key = None
    depth = 0
    while True:
        e_t = _through(e)
        if not isinstance(e_t, Ternary):
            break
        info = _eq_const_info(e_t.sel, key_of)
        if info is None:
            break
        signal, val, this_sel_key = info
        if sel_key is None:
            sel_signal = signal
            sel_key = this_sel_key
        elif this_sel_key != sel_key:
            break
        pairs.append((val, e_t.a))
        depth += 1
        e = e_t.b
    return sel_signal, pairs, e, depth


def _build_bit_tree(sel: Expr, leaves: List[Expr]) -> Expr:
    """Build a balanced binary mux tree using bits of `sel`.

    `leaves[i]` is selected when `sel == i`. `len(leaves)` must be a power of
    two and equal to `2 ** sel.typ.width`.
    """
    w = sel.typ.width
    n = len(leaves)
    assert n == (1 << w), f"leaves count {n} must equal 2^{w}"
    layer = list(leaves)
    for bit in range(w):
        sel_bit = Slice(sel, bit, bit + 1)
        layer = [Ternary(sel_bit, layer[i + 1], layer[i])
                 for i in range(0, len(layer), 2)]
    return layer[0]


def _try_balance_cascade(root: Expr, key_of: Callable[[Expr], tuple],
                          min_n: int) -> Expr:
    """If `root` heads a cascade of length M = 2^K-1 or M = 2^K with the keys
    {0, 1, ..., M-1} and the SAME sel signal at every level, return a balanced
    bit-tree replacement. Else return `root` unchanged.

    Two layout cases:
      - "Full cascade" (M = 2^K): every value in {0..2^K-1} appears as a key in
        the chain; the cascade's innermost `default` is dead code and ignored.
      - "Open cascade" (M = 2^K-1): keys cover {0..2^K-2} and the cascade ends
        in a `default` Expr that's the value for sel == 2^K-1. This is the
        idiomatic Python form `chain = items[last]; for i in reversed(...):
        chain = mux(sel == i, items[i], chain)`.
    """
    sel_signal, pairs, default, depth = _collect_cascade(root, key_of)
    if sel_signal is None:
        return root
    k = sel_signal.typ.width
    n_full = 1 << k
    # We bail out only if the FULL-COVERAGE size n_full is below the threshold —
    # depth itself can be n_full or n_full-1 (open cascade) which would be one
    # short of n_full but still produces a tree with n_full leaves.
    if n_full < min_n:
        return root

    keys = [k_ for (k_, _) in pairs]
    pair_count = depth
    if pair_count == n_full and sorted(keys) == list(range(n_full)):
        # Full cascade: build leaves from pairs only; default is dead.
        leaves: List[Expr] = [None] * n_full  # type: ignore[list-item]
        for k_, v in pairs:
            leaves[k_] = v
        return _build_bit_tree(sel_signal, leaves)
    if pair_count == n_full - 1 and sorted(keys) == list(range(n_full - 1)):
        # Open cascade: pairs cover 0..n_full-2, default fills slot n_full-1.
        leaves = [None] * n_full  # type: ignore[list-item]
        for k_, v in pairs:
            leaves[k_] = v
        leaves[n_full - 1] = default
        return _build_bit_tree(sel_signal, leaves)
    return root


def apply_mux_tree_balance(module, min_n: int = 16) -> int:
    """Detect and rewrite linear `mux(sel == Const(i), v_i, ...)` cascades of
    length >= min_n with full power-of-2 coverage into balanced bit-tree muxes.

    Returns the number of cascades rewritten. See header comment for context.
    """
    if min_n < 4:
        # below this threshold the bit-tree form is rarely a win — bail.
        return 0

    keyer = _KeyWalker()

    def key_of(e: Expr) -> tuple:
        return keyer.visit(e)

    n_changed = 0
    visited: set = set()

    def walk(e: Expr) -> Expr:
        nonlocal n_changed
        if id(e) in visited:
            return e
        visited.add(id(e))
        # Recurse into children FIRST so inner cascades have a chance to
        # rewrite before we look at the outer cascade.
        if isinstance(e, Op1):
            e.a = walk(e.a)
        elif isinstance(e, Op2):
            e.a = walk(e.a)
            e.b = walk(e.b)
        elif isinstance(e, Ternary):
            e.sel = walk(e.sel)
            e.a = walk(e.a)
            e.b = walk(e.b)
            # Now check whether *this* Ternary heads a cascade we should rewrite.
            new_e = _try_balance_cascade(e, key_of, min_n)
            if new_e is not e:
                n_changed += 1
                return new_e
        elif isinstance(e, Concat):
            e.parts = [walk(p) for p in e.parts]
        elif isinstance(e, Slice):
            e.a = walk(e.a)
        elif isinstance(e, Resize):
            e.a = walk(e.a)
        elif isinstance(e, Signal):
            if getattr(e, "_auto_generated", False) and e._driver is not None:
                new_drv = walk(e._driver)
                if new_drv is not e._driver:
                    e._driver = new_drv
        return e

    for s in module._signals:
        drv = getattr(s, "_driver", None)
        if isinstance(drv, Expr):
            new_drv = walk(drv)
            if new_drv is not drv:
                s._driver = new_drv

    return n_changed
