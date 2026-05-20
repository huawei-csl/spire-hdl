"""Shared visitor infrastructure for walking SpireHDL expression trees.

Provides:
- ``expr_children(e)`` — return the immediate sub-expressions of any Expr node.
- ``ExprVisitor[T]`` — abstract base with per-type dispatch, id-based caching,
  and clearly-named ``visit_*`` hooks.  Subclasses only override the hooks they
  need; the dispatch + caching boilerplate lives here once.
"""

from __future__ import annotations

from typing import Generic, Tuple, TypeVar

from spirehdl.spirehdl import (
    Concat,
    Const,
    Expr,
    MemRead,
    Op1,
    Op2,
    Resize,
    Signal,
    Slice,
    Ternary,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Standalone utility
# ---------------------------------------------------------------------------

def expr_children(e: Expr) -> Tuple[Expr, ...]:
    """Return the immediate structural sub-expressions of *e*.

    * ``Const`` and leaf ``Signal`` nodes (inputs, registers) have no children.
    * Combinational ``Signal`` nodes (wires, outputs) follow through to their
      driver expression.
    """
    if isinstance(e, Const):
        return ()
    if isinstance(e, Signal):
        if e.kind in ("input", "reg") or e._driver is None:
            return ()
        return (e._driver,)
    if isinstance(e, Op1):
        return (e.a,)
    if isinstance(e, Op2):
        return (e.a, e.b)
    if isinstance(e, Ternary):
        return (e.sel, e.a, e.b)
    if isinstance(e, Concat):
        return tuple(e.parts)
    if isinstance(e, Slice):
        return (e.a,)
    if isinstance(e, Resize):
        return (e.a,)
    if isinstance(e, MemRead):
        # The Memory itself is a Signal (kind="mem") and not a sub-expression
        # that participates in normal Expr DAG traversal — only the address is.
        return (e.addr,)
    return ()


# ---------------------------------------------------------------------------
# Visitor base class
# ---------------------------------------------------------------------------

class ExprVisitor(Generic[T]):
    """Abstract base for walking Expr trees with per-type dispatch and caching.

    Subclasses override ``visit_const``, ``visit_signal``, … to define
    per-node-type behaviour.  The public ``visit(e)`` method handles type
    dispatch and caches results by object identity (``id(e)``).
    """

    def __init__(self) -> None:
        self._cache: dict[int, T] = {}

    def visit(self, e: Expr) -> T:
        """Dispatch *e* to the appropriate ``visit_*`` handler (cached)."""
        eid = id(e)
        if eid in self._cache:
            return self._cache[eid]

        if isinstance(e, Const):
            result = self.visit_const(e)
        elif isinstance(e, Signal):
            result = self.visit_signal(e)
        elif isinstance(e, Op1):
            result = self.visit_op1(e)
        elif isinstance(e, Op2):
            result = self.visit_op2(e)
        elif isinstance(e, Ternary):
            result = self.visit_ternary(e)
        elif isinstance(e, Concat):
            result = self.visit_concat(e)
        elif isinstance(e, Slice):
            result = self.visit_slice(e)
        elif isinstance(e, Resize):
            result = self.visit_resize(e)
        elif isinstance(e, MemRead):
            result = self.visit_memread(e)
        else:
            raise TypeError(f"Unsupported Expr subclass: {type(e)}")

        self._cache[eid] = result
        return result

    def clear_cache(self) -> None:
        """Drop all cached results (call when underlying values change)."""
        self._cache.clear()

    # -- Override in subclasses ------------------------------------------------

    def visit_const(self, e: Const) -> T:
        raise NotImplementedError

    def visit_signal(self, e: Signal) -> T:
        raise NotImplementedError

    def visit_op1(self, e: Op1) -> T:
        raise NotImplementedError

    def visit_op2(self, e: Op2) -> T:
        raise NotImplementedError

    def visit_ternary(self, e: Ternary) -> T:
        raise NotImplementedError

    def visit_concat(self, e: Concat) -> T:
        raise NotImplementedError

    def visit_slice(self, e: Slice) -> T:
        raise NotImplementedError

    def visit_resize(self, e: Resize) -> T:
        raise NotImplementedError

    def visit_memread(self, e: MemRead) -> T:
        raise NotImplementedError
