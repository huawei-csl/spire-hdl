"""Post-construction structural CSE (Common Subexpression Elimination) for Spire modules.

There are two kinds of duplication to collapse in a module's driver DAG:

1. **High fan-out on a single Expr instance.** `a & b` constructed once may be reused by N parents.
   ``_maybe_share`` in ``spire.py`` catches this via ``as_expr``, but only on operands that go
   through operator overloads — `self` references (the first operand of `a & b`) bypass ``as_expr``
   and stay inline. Each parent emits the inline expression, so N references produce N `(a & b)`
   substrings in the Verilog output.

2. **Structural duplicates across distinct Expr instances.** AIG re-importers (``@abc_optimized`` /
   ``@flowy_optimized``) can build N independent ``Op2(a, b, '&', ...)`` nodes over the same operand
   Signals. Each is a distinct Python instance, so ``_maybe_share`` — which is instance-identity
   based — doesn't see them as duplicates.

``apply_structural_cse`` handles both: a single ``ExprVisitor``-based DAG walk gathers canonical
keys and parent-reference counts in one pass, then creates one shared wire per equivalence class
(class >= 2 members, or any member with fan-out >= 2) and rewrites every parent edge to point at
the wire. The result is information-preserving across the Spire → Verilog boundary: the emitted
Verilog's AIG gate count exactly matches the spire-native AigerExporter count.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

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
    _create_new_shared_wire,
)
from spire.memory import _ArrayIndex
from spire.visitor import ExprVisitor

_SYMMETRIC_OPS = frozenset({"&", "|", "^", "==", "!=", "nand"})


class _CseWalker(ExprVisitor[int]):
    """Single-pass walker that computes canonical keys and gathers every op.

    * ``self.all_ops`` — every non-leaf Expr visited, in DFS order.
    * ``self._cache`` (inherited) — id(Expr) -> canonical key (an interned int).

    Keys are hash-consed: each key tuple is interned to a class-id and parent keys embed child
    ids, so every key is O(1) to hash/compare (nested-tuple keys made the pass O(n * depth)).

    Leaves (Const / Signal) are NOT recursed through: a wire-Signal's driver is owned by the Signal,
    not re-canonicalised here. This keeps wire identities stable (no renaming of user-visible named
    signals).
    """

    def __init__(self) -> None:
        super().__init__()
        self.all_ops: List[Expr] = []
        self._intern_tab: dict = {}
        self.key_of: dict = {}  # id(Expr) -> interned class-id, for non-leaf ops

    def _intern(self, key: tuple) -> int:
        cid = self._intern_tab.get(key)
        if cid is None:
            cid = len(self._intern_tab)
            self._intern_tab[key] = cid
        return cid

    def _op(self, e: Expr, key: tuple) -> int:
        self.all_ops.append(e)
        cid = self._intern(key)
        self.key_of[id(e)] = cid
        return cid

    def visit_const(self, e: Const) -> int:
        return self._intern(("const", e.value, e.typ.width, e.typ.signed))

    def visit_signal(self, e: Signal) -> int:
        return self._intern(("sig", id(e)))

    def visit_op1(self, e: Op1) -> int:
        return self._op(e, ("op1", e.op, self.visit(e.a), e.typ.width, e.typ.signed))

    def visit_op2(self, e: Op2) -> int:
        ka, kb = self.visit(e.a), self.visit(e.b)
        if e.op in _SYMMETRIC_OPS and ka > kb:
            ka, kb = kb, ka
        return self._op(e, ("op2", e.op, ka, kb, e.typ.width, e.typ.signed))

    def visit_ternary(self, e: Ternary) -> int:
        return self._op(e, ("tern", self.visit(e.sel), self.visit(e.a), self.visit(e.b),
                            e.typ.width, e.typ.signed))

    def visit_concat(self, e: Concat) -> int:
        return self._op(e, ("cat", tuple(self.visit(p) for p in e.parts)))

    def visit_slice(self, e: Slice) -> int:
        return self._op(e, ("slice", self.visit(e.a), e.start, e.msb))

    def visit_resize(self, e: Resize) -> int:
        return self._op(e, ("resize", self.visit(e.a), e.to_width, e.typ.signed))

    def visit_array_index(self, e: _ArrayIndex) -> int:
        # Leaf — address signal is reached via Memory's port traversal, not through
        # this Expr's fields. Canonical key uses id(mem) + id(addr_wire) since both
        # are user-named Signals; we never merge across distinct Memory instances.
        return self._intern(("array_index", id(e.mem), id(e.addr_wire), e.typ.width, e.typ.signed))


def _children_of(e: Expr) -> Tuple[Expr, ...]:
    """Return non-leaf children of e (those that could themselves be shared)."""
    if isinstance(e, Op1):
        return (e.a,) if not isinstance(e.a, (Const, Signal)) else ()
    if isinstance(e, Op2):
        cs = []
        if not isinstance(e.a, (Const, Signal)):
            cs.append(e.a)
        if not isinstance(e.b, (Const, Signal)):
            cs.append(e.b)
        return tuple(cs)
    if isinstance(e, Ternary):
        cs = []
        for x in (e.sel, e.a, e.b):
            if not isinstance(x, (Const, Signal)):
                cs.append(x)
        return tuple(cs)
    if isinstance(e, Concat):
        return tuple(p for p in e.parts if not isinstance(p, (Const, Signal)))
    if isinstance(e, Slice):
        return (e.a,) if not isinstance(e.a, (Const, Signal)) else ()
    if isinstance(e, Resize):
        return (e.a,) if not isinstance(e.a, (Const, Signal)) else ()
    return ()


def apply_structural_cse(module) -> int:
    """Post-construction CSE (Common Subexpression Elimination) pass — collapse duplicated subtrees.

    Two sharing criteria:

    1. **Fan-out**: any Expr with >= 2 parent references gets a shared wire.
    2. **Structural**: distinct Expr instances with identical canonical keys collapse to a single
       wire (the first one becomes its driver).

    The combined effect: every equivalence class (by either criterion) emits its expression once as
    a wire driver, and all N-1 other uses become simple wire references.

    Returns the number of shared wires created.
    """
    # 1. Gather roots: every driver currently attached to a signal. Suppressed drivers
    # (`_no_emit_drive`, e.g. the simulation model of a custom-Verilog component) never emit,
    # so sharing their subtrees would only mint dead wires.
    roots: List[Expr] = []
    for s in module._signals:
        if getattr(s, "_no_emit_drive", False):
            continue
        drv = getattr(s, "_driver", None)
        if isinstance(drv, Expr):
            roots.append(drv)

    if not roots:
        return 0

    # 2. Walk graph: compute canonical keys AND count parent references.
    walker = _CseWalker()
    for r in roots:
        walker.visit(r)

    # Root references count as one parent edge each.
    refcount: dict = defaultdict(int)
    for r in roots:
        refcount[id(r)] += 1
    for e in walker.all_ops:
        for child in _children_of(e):
            refcount[id(child)] += 1

    # 3. Build equivalence classes:
    #    - Group all_ops by canonical key (handles criterion 2).
    #    - Within a key's class, mark whether any member has fan-out >= 2 OR the class itself has
    #      >= 2 members (covers criterion 1).
    key_to_instances: dict = defaultdict(list)
    for e in walker.all_ops:
        key_to_instances[walker.key_of[id(e)]].append(e)  # interned class-id (see _CseWalker)

    # 4. For each class that qualifies (multi-member OR high fan-out on any member), create one
    #    shared wire. First instance in the class is the driver; every other instance in the class
    #    (or subsequent parent references to the first instance) redirects to the wire.
    redirect: dict = {}  # id(expr) -> Signal
    wires_created = 0
    for key, insts in key_to_instances.items():
        # Qualify if: class has multiple members, OR any single member has fan-out >= 2
        # (meaning it's referenced from multiple parents).
        any_high_fanout = any(refcount[id(e)] >= 2 for e in insts)
        if len(insts) < 2 and not any_high_fanout:
            continue
        first = insts[0]
        sig = _create_new_shared_wire(first.typ, getattr(first, "_suggested_name", None))
        sig._driver = first
        # Every OTHER instance in the class redirects to the wire.
        for other in insts[1:]:
            redirect[id(other)] = sig
        # The first instance stays as the driver of the wire; every parent reference to it (however many)
        # redirects to the wire, so the shared logic emits exactly once.
        redirect[id(first)] = sig
        wires_created += 1

    if not redirect:
        return 0

    # 5. Rewrite parent fields on every op visited.
    def _redir(child: Expr) -> Expr:
        r = redirect.get(id(child))
        return r if r is not None else child

    for e in walker.all_ops:
        if isinstance(e, Op1):
            e.a = _redir(e.a)
        elif isinstance(e, Op2):
            e.a = _redir(e.a)
            e.b = _redir(e.b)
        elif isinstance(e, Ternary):
            e.sel = _redir(e.sel)
            e.a = _redir(e.a)
            e.b = _redir(e.b)
        elif isinstance(e, Concat):
            e.parts = [_redir(p) for p in e.parts]
        elif isinstance(e, Slice):
            e.a = _redir(e.a)
        elif isinstance(e, Resize):
            e.a = _redir(e.a)

    # 6. Rewrite Signal drivers, EXCEPT for the shared wires themselves (a shared wire's driver is
    #    the `first` instance in its class and must stay that way — redirecting it to itself would
    #    form a cycle, and redirecting to another wire would lose the driver).
    our_wires = {id(sig) for sig in redirect.values()}
    for s in module._signals:
        if id(s) in our_wires:
            continue
        drv = getattr(s, "_driver", None)
        if drv is not None and id(drv) in redirect:
            s._driver = redirect[id(drv)]

    return wires_created
