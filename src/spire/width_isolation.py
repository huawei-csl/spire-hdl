"""Emission-time width isolation: named-wire boundaries around inline compound operands.

Spire evaluates every expression node at its own width and wraps there; IEEE-1364 re-sizes inline operands to the
enclosing expression's width/signedness with no intermediate wrap, so an inline compound operand can synthesize to
a different value than it simulates. This pass runs once per emission (after simplify/balance/CSE, which want the
original nesting) and rewires every remaining non-leaf operand through a declared wire, whose assignment evaluates
it at exactly its node width and signedness. Values are unchanged; only netlist structure changes — same contract
as CSE. Exception, kept for PPA: width-1 unsigned boolean logic stays inline in 1-bit/self-determined positions
(preserves the flat cones of the FSM bit-level emitter, independent of when `flat_emit` was active).
"""
from __future__ import annotations

from typing import Dict, List

from spire.expr import Concat, Const, Expr, Op1, Op2, Resize, Signal, Slice, Ternary, _create_new_shared_wire

_BOOL_OPS = frozenset({"&", "|", "^", "nand"})
_CMP_OPS = frozenset({"==", "!=", "<", "<=", ">", ">="})


def _is_leaf(e: Expr) -> bool:
    # Anything that is not one of the compound node classes emits as a named reference (Signal), a literal (Const),
    # or an opaque indexed form (e.g. memory reads) whose base operands are named signals already.
    return not isinstance(e, (Op1, Op2, Ternary, Concat, Slice, Resize))


def _flat_safe(e: Expr) -> bool:
    """Value-safe to emit inline in a 1-bit / self-determined position: width-1 unsigned boolean logic."""
    if _is_leaf(e):
        return True
    t = e.typ
    if t.width != 1 or t.signed:
        return False
    if isinstance(e, Op1):
        return e.op == "~" and e.a.typ.width == 1 and _flat_safe(e.a)
    if isinstance(e, Op2):
        if e.op in _BOOL_OPS or e.op in ("==", "!="):
            return e.a.typ.width == 1 and e.b.typ.width == 1 and _flat_safe(e.a) and _flat_safe(e.b)
        return False
    if isinstance(e, Ternary):
        return (e.sel.typ.width == 1 and e.a.typ.width == 1 and e.b.typ.width == 1
                and _flat_safe(e.sel) and _flat_safe(e.a) and _flat_safe(e.b))
    if isinstance(e, Slice):
        return isinstance(e.a, (Signal, Const))  # bit-select of a named base
    return False


def _inline_ok(parent: Expr, child: Expr) -> bool:
    """May `child` stay inline under `parent`? Only where its IEEE evaluation context is 1 bit wide or
    self-determined, and the child is flat-safe — then inline evaluation equals the child's node semantics."""
    if not _flat_safe(child):
        return False
    if isinstance(parent, Op1):
        return parent.op == "~" and parent.typ.width == 1
    if isinstance(parent, Op2):
        if parent.op in _BOOL_OPS:
            return parent.typ.width == 1
        if parent.op in ("==", "!="):  # comparison island width = max of both operands
            return parent.a.typ.width == 1 and parent.b.typ.width == 1
        if parent.op in ("<", "<=", ">", ">="):
            return False  # ordered compares get explicit treatment; keep operands named
        return False  # arithmetic/shift contexts are wider than the child
    if isinstance(parent, Ternary):
        if child is parent.sel:
            return True  # the ?: condition is self-determined
        return parent.typ.width == 1
    if isinstance(parent, Concat):
        return True  # concat parts are self-determined
    if isinstance(parent, Resize):
        return True  # truncation slice / extension concat evaluate the source self-determined
    if isinstance(parent, Slice):
        return False  # select bases must be named for lexically valid Verilog
    return False


def _children(e: Expr) -> List[Expr]:
    if isinstance(e, Op1):
        return [e.a]
    if isinstance(e, Op2):
        return [e.a, e.b]
    if isinstance(e, Ternary):
        return [e.sel, e.a, e.b]
    if isinstance(e, Concat):
        return list(e.parts)
    if isinstance(e, (Slice, Resize)):
        return [e.a]
    return []


def _replace_child(e: Expr, old: Expr, new: Expr) -> None:
    if isinstance(e, (Op1, Slice, Resize)):
        if e.a is old:
            e.a = new
    elif isinstance(e, Op2):
        if e.a is old:
            e.a = new
        if e.b is old:
            e.b = new
    elif isinstance(e, Ternary):
        if e.sel is old:
            e.sel = new
        if e.a is old:
            e.a = new
        if e.b is old:
            e.b = new
    elif isinstance(e, Concat):
        e.parts = [new if p is old else p for p in e.parts]


def apply_width_isolation(module) -> int:
    """Wire every inline compound operand that IEEE context rules could re-size (see module docstring).

    Returns the number of wires created. Idempotent: a second run finds only Signal-backed operands.
    """
    roots: List[Expr] = []
    for s in module._signals:
        drv = getattr(s, "_driver", None)
        if isinstance(drv, Expr) and not _is_leaf(drv):
            roots.append(drv)
    if not roots:
        return 0

    # Post-order over the compound DAG, treating Signals/Consts/etc. as boundaries (their drivers are separate
    # roots via module._signals). `done` is id-keyed but holds strong refs via `order`, so ids cannot recycle.
    order: List[Expr] = []
    done: set = set()
    stack: List[tuple] = [(r, False) for r in roots]
    while stack:
        node, expanded = stack.pop()
        if id(node) in done:
            continue
        if expanded:
            done.add(id(node))
            order.append(node)
            continue
        stack.append((node, True))
        for c in _children(node):
            if not _is_leaf(c) and id(c) not in done:
                stack.append((c, False))

    wire_for: Dict[int, Signal] = {}  # id(child) -> shared wire (one per instance, reused across parents)
    created = 0
    for parent in order:
        for child in _children(parent):
            if _is_leaf(child) or _inline_ok(parent, child):
                continue
            sig = wire_for.get(id(child))
            if sig is None:
                sig = _create_new_shared_wire(child.typ, getattr(child, "_suggested_name", None))
                sig._driver = child
                wire_for[id(child)] = sig
                created += 1
            _replace_child(parent, child, sig)
    return created
