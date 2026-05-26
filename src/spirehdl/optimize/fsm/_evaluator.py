"""Symbolic-substitution evaluator.

Concrete-integer evaluation of an Expr DAG under a ``{Signal -> int}`` environment. Used by
``extract_transition_table`` to enumerate (state_value × input_combination) transitions when reconstructing the
FSM's transition table from ``reg._driver``.

The operator table mirrors ``spirehdl_simplify._fold_op2`` (same width-masked Verilog semantics) so symbolic eval
never diverges from what the peephole simplifier would compute on the same Const inputs.

Auto-shared CSE wires (``_auto_generated=True``) are transparently followed via ``signal._driver``; user-named
signals must appear in the ``env`` dict or raise.
"""
from __future__ import annotations

from typing import Iterable

from spirehdl.spirehdl import (
    Concat, Const, Expr, Op1, Op2, Resize, Signal, Slice, Ternary,
)
from spirehdl.spirehdl_visitor import ExprVisitor

# Bindings are passed as an iterable of (Signal, int) pairs because Signal overrides ``__eq__`` to build an Op2 Expr,
# so Signals can't be dict keys.
Bindings = Iterable[tuple[Signal, int]]


def _mask(width: int) -> int:
    return (1 << width) - 1 if width > 0 else 0


def _apply_op1(op: str, a: int, width: int) -> int:
    if op == "~":  return (~a) & _mask(width)
    if op == "!":  return 0 if a else 1
    if op == "-":  return (-a) & _mask(width)
    raise NotImplementedError(f"Op1 op={op!r} not supported by evaluator")


def _apply_op2(op: str, a: int, b: int, width: int) -> int:
    m = _mask(width)
    if op == "&":    return (a & b) & m
    if op == "|":    return (a | b) & m
    if op == "^":    return (a ^ b) & m
    if op == "nand": return (~(a & b)) & m
    if op == "+":    return (a + b) & m
    if op == "-":    return (a - b) & m
    if op == "*":    return (a * b) & m
    if op == "==":   return 1 if a == b else 0
    if op == "!=":   return 0 if a == b else 1
    if op == "<":    return 1 if a <  b else 0
    if op == "<=":   return 1 if a <= b else 0
    if op == ">":    return 1 if a >  b else 0
    if op == ">=":   return 1 if a >= b else 0
    if op == "<<":   return (a << b) & m
    if op == ">>":   return (a >> b) & m
    raise NotImplementedError(f"Op2 op={op!r} not supported by evaluator")


class _Eval(ExprVisitor[int]):
    """Concrete evaluator; one instance per (expr, env) call.

    The visitor caches per-node so shared sub-Exprs are computed only once, which matters for deeply-nested mux
    trees built from `switch_/case_`.
    """

    def __init__(self, bindings: Bindings) -> None:
        super().__init__()
        # Keyed by id(signal) so we don't trigger Signal.__eq__ (which builds an Op2 Expr rather than returning a
        # bool).
        self._env: dict[int, int] = {id(s): int(v) for s, v in bindings}

    def visit_const(self, e: Const) -> int:
        return e.value & _mask(e.typ.width) if e.typ.width > 0 else 0

    def visit_signal(self, e: Signal) -> int:
        sid = id(e)
        if sid in self._env:
            return self._env[sid]
        if e._driver is not None:
            return self.visit(e._driver)
        raise ValueError(f"evaluator: signal {e.name!r} has no value in env and no driver")

    def visit_op1(self, e: Op1) -> int:
        return _apply_op1(e.op, self.visit(e.a), e.typ.width)

    def visit_op2(self, e: Op2) -> int:
        return _apply_op2(e.op, self.visit(e.a), self.visit(e.b), e.typ.width)

    def visit_ternary(self, e: Ternary) -> int:
        return self.visit(e.a) if self.visit(e.sel) else self.visit(e.b)

    def visit_slice(self, e: Slice) -> int:
        return (self.visit(e.a) >> e.start) & _mask(e.typ.width)

    def visit_resize(self, e: Resize) -> int:
        return self.visit(e.a) & _mask(e.to_width)

    def visit_concat(self, e: Concat) -> int:
        # Concat is LSB-first in spirehdl: parts[0] occupies the low bits.
        out = 0
        shift = 0
        for p in e.parts:
            out |= (self.visit(p) & _mask(p.typ.width)) << shift
            shift += p.typ.width
        return out


def eval_with(expr: Expr, bindings: Bindings) -> int:
    """Top-level entry point: evaluate ``expr`` under ``bindings``.

    Bindings is an iterable of ``(Signal, int)`` pairs (a list of tuples, not a dict — see the module-level
    ``Bindings`` type alias for the rationale). Raises ``ValueError`` if any referenced signal is unbound and has
    no driver to fall back to.
    """
    return _Eval(bindings).visit(expr)
