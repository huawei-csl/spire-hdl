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
