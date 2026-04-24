"""Control structures for SpireHDL using Python context managers.

This module introduces `if_`/`elif_`/`else_` and `switch_`/`case_` style
constructs that wrap signal assignments with conditional muxes.  When a
signal assignment occurs inside one of the provided context managers, the
assignment is guarded by the active condition.  If the condition evaluates to
false, the signal retains its previous driver (for combinational signals) or
its current value (for registers).

Usage example::

    from spirehdl.spirehdl import Bool, UInt
    from spirehdl.spirehdl_module import Module
    from spirehdl.spirehdl_control_structures import case_, default, if_, elif_, else_, switch_

    m = Module("Example", with_clock=False, with_reset=False)
    sel = m.input(Bool(), "sel")
    y = m.output(UInt(8), "y")

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

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

from spirehdl.spirehdl import Expr, ExprLike, Signal, as_expr, mux


# ---------------------------------------------------------------------------
# Condition stack helpers
# ---------------------------------------------------------------------------

class _ConditionState:
    active: List[Expr] = []
    pending_if_chain: Optional["_IfChain"] = None
    switch_stack: List["_SwitchState"] = []


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


# ---------------------------------------------------------------------------
# If/elif/else support
# ---------------------------------------------------------------------------

@dataclass
class _IfChain:
    covered: Expr
    closed: bool = False

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

    def __enter__(self):
        _push_condition(self._condition)
        return self

    def __exit__(self, exc_type, exc, tb):
        _pop_condition()
        if self._on_exit is not None:
            self._on_exit()
        return False


def _set_pending_chain(chain: Optional[_IfChain]) -> None:
    _ConditionState.pending_if_chain = chain


def _clear_pending_chain_if_needed() -> None:
    if _ConditionState.pending_if_chain is not None:
        _set_pending_chain(None)


def if_(condition: ExprLike) -> _ConditionalContext:
    """Context manager representing an `if` branch."""

    _clear_pending_chain_if_needed()
    chain = _IfChain(covered=_bool_const(False))
    cond = chain.branch(condition, context="if")

    def _on_exit():
        _set_pending_chain(chain)

    return _ConditionalContext(cond, on_exit=_on_exit)


def elif_(condition: ExprLike) -> _ConditionalContext:
    """Context manager representing an `elif` branch."""

    if _ConditionState.pending_if_chain is None:
        raise RuntimeError("elif_ must follow an if_ or another elif_ block")
    chain = _ConditionState.pending_if_chain
    cond = chain.branch(condition, context="elif")

    def _on_exit():
        _set_pending_chain(chain)

    return _ConditionalContext(cond, on_exit=_on_exit)


def else_() -> _ConditionalContext:
    """Context manager representing an `else` branch."""

    if _ConditionState.pending_if_chain is None:
        raise RuntimeError("else_ must follow an if_ or elif_ block")
    chain = _ConditionState.pending_if_chain
    cond = chain.default()

    def _on_exit():
        _set_pending_chain(None)

    return _ConditionalContext(cond, on_exit=_on_exit)


# ---------------------------------------------------------------------------
# Switch_/case_ support
# ---------------------------------------------------------------------------


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
            cmp = self._selector == as_expr(value)
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


# ---------------------------------------------------------------------------
# Assignment patching
# ---------------------------------------------------------------------------

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
    global _PATCHED
    if _PATCHED:
        return

    original_ilshift = Signal.__ilshift__

    def conditional_ilshift(self: Signal, rhs: ExprLike):
        wrapped_rhs = _apply_active_conditions_to_expr(self, rhs)
        return original_ilshift(self, wrapped_rhs)

    Signal.__ilshift__ = conditional_ilshift  # type: ignore[assignment]

    _PATCHED = True


_patch_signal_assignments()


__all__ = ["if_", "elif_", "else_", "switch_", "case_", "default"]
