"""Control structures for Spire using Python context managers.

This module introduces `if_`/`elif_`/`else_` and `switch_`/`case_` style constructs that wrap signal
assignments with conditional muxes. When a signal assignment occurs inside one of these context
managers, the assignment is guarded by the active condition. If the condition evaluates to false, the
signal retains its previous driver (for combinational signals) or its current value (for registers).

Usage::

    y <<= 0
    with if_(sel):
        y <<= 1
    with else_():
        y <<= 2

    with switch_(sel):
        with case_(0):
            y <<= 3
        with default():
            y <<= 4
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Tuple

from spire.expr import Const, Expr, ExprLike, Signal, as_expr, mux
from spire.hdl_traits import BitSerializable


# Condition stack helpers

class _ConditionState:
    active: List[Expr] = []
    pending_if_chain: Optional["_IfChain"] = None
    switch_stack: List["_SwitchState"] = []


@contextlib.contextmanager
def fresh_condition_scope():
    """Run a block with empty condition/switch/pending-chain state, restoring the caller's state afterwards.

    Component construction wraps elaboration in this scope: a component builds the same structure no matter where
    it is constructed (an enclosing ``if_`` gates *assignments*, not elaboration), and a trailing ``if_`` chain
    inside ``elaborate()`` can neither escape to the caller nor swallow the caller's own pending chain.

    Plain Python functions share the caller's scope by design (conditions apply to what a helper assigns); a helper
    that wants isolation can wrap its body in this context manager itself.
    """
    saved = (_ConditionState.active, _ConditionState.pending_if_chain, _ConditionState.switch_stack)
    _ConditionState.active, _ConditionState.pending_if_chain, _ConditionState.switch_stack = [], None, []
    try:
        yield
    finally:
        _ConditionState.active, _ConditionState.pending_if_chain, _ConditionState.switch_stack = saved


def _current_scope() -> Tuple[tuple, tuple]:
    return (tuple(_ConditionState.active), tuple(_ConditionState.switch_stack))


def _same_scope(a: Tuple[tuple, tuple], b: Tuple[tuple, tuple]) -> bool:
    # Element-wise identity: tuple == would invoke Expr.__eq__ (which builds hardware).
    return (len(a[0]) == len(b[0]) and len(a[1]) == len(b[1])
            and all(x is y for x, y in zip(a[0], b[0]))
            and all(x is y for x, y in zip(a[1], b[1])))


def _bool_const(value: bool) -> Expr:
    return as_expr(bool(value))


def _validate_bool(expr: Expr, *, context: str) -> None:
    if expr.typ.width != 1:
        raise ValueError(f"{context} conditions must be 1-bit expressions, got width {expr.typ.width}")


def _push_condition(cond: Expr) -> None:
    _ConditionState.active.append(cond)


def _pop_condition() -> None:
    if not _ConditionState.active:
        raise RuntimeError("Condition stack underflow")
    _ConditionState.active.pop()


def _combined_condition() -> Optional[Expr]:
    if not _ConditionState.active:
        return None
    cond = _ConditionState.active[0]
    for extra in _ConditionState.active[1:]:
        cond = cond & extra
    return cond


# If/elif/else support

@dataclass
class _IfChain:
    covered: Expr
    closed: bool = False
    # The condition/switch ambience (strong refs) in which the chain was left pending; elif_/else_ may only claim
    # it from the identical ambience, so a trailing chain in one case_/branch can't be continued in another.
    scope: Optional[Tuple[tuple, tuple]] = None

    def branch(self, condition: ExprLike, *, context: str) -> Expr:
        if self.closed:
            raise RuntimeError("Cannot add a branch to a closed if/elif/else chain")
        cond_expr = as_expr(condition)
        _validate_bool(cond_expr, context=context)
        gated = cond_expr & ~self.covered
        self.covered = self.covered | gated
        return gated

    def default(self) -> Expr:
        if self.closed:
            raise RuntimeError("Cannot add an else branch after chain was closed")
        cond_expr = ~self.covered
        self.covered = _bool_const(True)
        self.closed = True
        return cond_expr


class _ConditionalContext:
    def __init__(self, condition: Expr, on_exit: Optional[Callable[[], None]] = None):
        _validate_bool(condition, context="Conditional")
        self._condition = condition
        self._on_exit = on_exit
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Conditional context is already active")
        _push_condition(self._condition)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._entered:
            raise RuntimeError("Conditional context is not active")
        self._entered = False
        _pop_condition()
        if self._on_exit is not None:
            self._on_exit()
        return False


def _set_pending_chain(chain: Optional[_IfChain]) -> None:
    _ConditionState.pending_if_chain = chain


def _clear_pending_chain_if_needed() -> None:
    if _ConditionState.pending_if_chain is not None:
        _set_pending_chain(None)


def _claim_pending_chain(context: str) -> _IfChain:
    chain = _ConditionState.pending_if_chain
    if chain is None:
        raise RuntimeError(f"{context} must follow an if_ or elif_ block")
    if chain.scope is None or not _same_scope(chain.scope, _current_scope()):
        raise RuntimeError(f"{context} does not follow an if_/elif_ in the same scope "
                           f"(the pending chain was left in a different branch or switch case)")
    return chain


def if_(condition: ExprLike) -> _ConditionalContext:
    """Context manager representing an `if` branch."""

    _clear_pending_chain_if_needed()
    chain = _IfChain(covered=_bool_const(False))
    cond = chain.branch(condition, context="if")

    def _on_exit():
        chain.scope = _current_scope()
        _set_pending_chain(chain)

    return _ConditionalContext(cond, on_exit=_on_exit)


def elif_(condition: ExprLike) -> _ConditionalContext:
    """Context manager representing an `elif` branch."""

    chain = _claim_pending_chain("elif_")
    cond = chain.branch(condition, context="elif")

    def _on_exit():
        chain.scope = _current_scope()
        _set_pending_chain(chain)

    return _ConditionalContext(cond, on_exit=_on_exit)


def else_() -> _ConditionalContext:
    """Context manager representing an `else` branch."""

    chain = _claim_pending_chain("else_")
    cond = chain.default()

    def _on_exit():
        _set_pending_chain(None)

    return _ConditionalContext(cond, on_exit=_on_exit)


# Switch_/case_ support


class _SwitchState:
    def __init__(self, selector: ExprLike):
        self._selector = as_expr(selector)
        self._covered = _bool_const(False)
        self._closed = False

    def _claim_cases(self, cases: Iterable[ExprLike]) -> Expr:
        if self._closed:
            raise RuntimeError("No further case_ or default branches allowed after default()")

        merged: Optional[Expr] = None
        for value in cases:
            value_expr = as_expr(value)
            if isinstance(value_expr, Const):
                try:
                    Const(value_expr.value, self._selector.typ)  # representability check only
                except ValueError:
                    raise ValueError(f"case value {value_expr.value} can never match the "
                                     f"{self._selector.typ.width}-bit selector") from None
            cmp = self._selector == value_expr
            _validate_bool(cmp, context="case comparison")
            merged = cmp if merged is None else (merged | cmp)

        if merged is None:
            raise ValueError("case_() requires at least one value")

        cond = merged & ~self._covered
        self._covered = self._covered | cond
        return cond

    def case_condition(self, *values: ExprLike) -> Expr:
        return self._claim_cases(values)

    def default_condition(self) -> Expr:
        if self._closed:
            raise RuntimeError("default() has already been used for this switch")
        cond = ~self._covered
        self._covered = _bool_const(True)
        self._closed = True
        return cond

    def reset(self) -> None:
        self._covered = _bool_const(False)
        self._closed = False


class switch_:
    """Context manager modeling a Verilog-style `switch_`/`case_` statement."""

    def __init__(self, selector: ExprLike):
        self._state = _SwitchState(selector)
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("switch_ context cannot be re-entered while active")
        _ConditionState.switch_stack.append(self._state)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._entered:
            raise RuntimeError("switch_ context was not active")
        if not _ConditionState.switch_stack or _ConditionState.switch_stack[-1] is not self._state:
            raise RuntimeError("switch_ stack corruption detected")
        _ConditionState.switch_stack.pop()
        self._state.reset()
        self._entered = False
        return False

    def _ensure_active(self, context: str) -> None:
        if not self._entered or not _ConditionState.switch_stack or _ConditionState.switch_stack[-1] is not self._state:
            raise RuntimeError(f"{context} must be used within an active switch_ context")

    def case_(self, *values: ExprLike) -> _ConditionalContext:
        self._ensure_active("case_")
        cond = self._state.case_condition(*values)
        return _ConditionalContext(cond)

    def default(self) -> _ConditionalContext:
        self._ensure_active("default")
        cond = self._state.default_condition()
        return _ConditionalContext(cond)


def _current_switch_state(context: str) -> _SwitchState:
    if not _ConditionState.switch_stack:
        raise RuntimeError(f"{context} must be used inside a switch_ block")
    return _ConditionState.switch_stack[-1]


def case_(*values: ExprLike) -> _ConditionalContext:
    """Context manager representing a case branch of the innermost switch_."""

    state = _current_switch_state("case_")
    cond = state.case_condition(*values)
    return _ConditionalContext(cond)


def default() -> _ConditionalContext:
    """Context manager representing the default branch of the innermost switch_."""

    state = _current_switch_state("default")
    cond = state.default_condition()
    return _ConditionalContext(cond)


# Assignment patching

_PATCHED = False


def _apply_active_conditions_to_expr(signal: Signal, rhs: ExprLike) -> ExprLike:
    cond = _combined_condition()
    if cond is None:
        return rhs

    rhs_expr = as_expr(rhs)

    if signal._driver is None:
        if signal.kind == "reg":
            # For registers without a prior driver, fall back to the register's current value
            fallback: ExprLike = signal
        else:
            raise RuntimeError(
                f"Conditional assignment to signal '{signal.name}' requires a prior driver to fall back to"
            )
    else:
        fallback = signal._driver

    return mux(cond, rhs_expr, fallback)


def _patch_signal_assignments() -> None:
    """Wrap `Signal.assign` — the one assignment primitive (`<<=` routes through it) — so every driver update,
    including direct `.assign()` calls, respects the active conditions."""
    global _PATCHED
    if _PATCHED:
        return

    original_assign = Signal.assign

    def conditional_assign(self: Signal, rhs):
        if isinstance(rhs, BitSerializable) and not isinstance(rhs, Expr):
            rhs = rhs.to_bits()  # pack composites before gating, as the unconditional path does
        wrapped_rhs = _apply_active_conditions_to_expr(self, rhs)
        return original_assign(self, wrapped_rhs)

    Signal.assign = conditional_assign  # type: ignore[assignment]

    _PATCHED = True


_patch_signal_assignments()


__all__ = ["if_", "elif_", "else_", "switch_", "case_", "default", "fresh_condition_scope"]
