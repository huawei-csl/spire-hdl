"""Registered delay lines — ``pipeline(value, cycles)``.

Retiming a datapath by hand means declaring a register, repeating its type, and wiring
next-state for every signal — and once per leaf for an aggregate. ``pipeline`` states the
intent instead and returns a value of the *same type* as its input, so adding or removing a
stage is a one-token edit::

    y = pipeline(x)                    # 1 cycle  (default)
    y = pipeline(x, 3)                 # 3 cycles, 3 chained registers
    y = pipeline(x, 2, enable=fire)    # holds while `enable` is low
    y = pipeline(x, 2, init=0)         # every stage resets to 0

It accepts any leaf ``Expr``/``Signal`` and any :class:`~spire.composite.base.HDLComposite`
(``Array``, records/interfaces, ``FixedPoint``, ``FloatingPoint``). Each composite stage is a
:class:`CompositeRegister` built with ``CompositeRegister.like(value)``, and the result is the
last stage's ``.value`` view, so ``.bits`` / field access / arithmetic all keep working on the
delayed value. A ``CompositeRegister`` input is delayed as the value it holds and comes back as
its ``.value`` type.

Stages are named ``<base>_d1 … <base>_dN``, with ``<base>`` taken from ``name=``, else the
input signal's name, else the variable the result is bound to. A composite's returned view is
named by ``CompositeRegister`` (``<base>_dN_<field path>``, or ``<base>_dN_q`` for one leaf).
"""
from __future__ import annotations

from typing import Optional, TypeVar, Union

from spire.composite.base import HDLComposite
from spire.composite.register import CompositeRegister
from spire.control_structures import if_
from spire.expr import Expr, ExprLike, Register, Signal, as_expr
from spire.signal_name_inference import (infer_expression_binding_name,
                                         sanitize_signal_name)

T = TypeVar("T", bound=Union[Expr, HDLComposite])

__all__ = ["pipeline"]


def pipeline(
    value: Union[T, int, bool],
    cycles: int = 1,
    *,
    init: Optional[ExprLike] = None,
    enable: Optional[ExprLike] = None,
    name: Optional[str] = None,
) -> T:
    """Delay ``value`` by ``cycles`` clock cycles, returning the same type.

    Args:
        value: what to delay — an ``Expr``/``Signal``, an int/bool literal, or any
            ``HDLComposite``. Composites come back as a new same-shaped instance; a
            ``CompositeRegister`` comes back as its structured ``.value`` type.
        cycles: number of register stages (default 1). ``0`` returns ``value`` undelayed.
        init: reset value applied to **every** stage; must be a constant (an ``int`` for a
            composite, interpreted as the packed reset pattern).
        enable: 1-bit stall control. While low, every stage holds its value.
        name: base name for the stages, overriding the inferred one.

    Raises:
        ValueError: ``cycles`` is negative.
    """
    if isinstance(cycles, bool) or not isinstance(cycles, int):
        raise TypeError(f"pipeline(): cycles must be an int, got {type(cycles).__name__}")
    if cycles < 0:
        raise ValueError(f"pipeline(): cycles must be >= 0, got {cycles}")

    if isinstance(value, CompositeRegister):
        # Delay what the register holds. The result is a value, so it comes back as the
        # register's structured `.value` type, not as another register. Stages are named
        # after the register unless `name` says otherwise.
        name = name or value.name
        value = value.value

    base = _base_name(value, name)

    if isinstance(value, HDLComposite):
        if cycles == 0:
            return value
        return _delay(value, cycles, base, init, enable).value

    e = as_expr(value)
    if cycles == 0:
        return e
    return _delay(e, cycles, base, init, enable)


def _delay(value, cycles: int, base: str, init: Optional[ExprLike], enable: Optional[ExprLike]):
    """Chain of ``cycles`` registers holding ``value``; returns the last one.

    A composite gets ``CompositeRegister.like`` stages and stays a composite throughout: every
    stage is assigned the previous one directly (``r <<= prev``), so nothing is packed or sliced
    here. A leaf gets plain ``Register`` stages the same way.

    ``enable`` wraps each stage's assignment in ``if_``, so a low enable holds the whole line —
    the same thing you would write by hand.
    """
    composite = isinstance(value, HDLComposite)
    en = None if enable is None else as_expr(enable)
    prev = value
    for stage in range(1, cycles + 1):
        stage_name = f"{base}_d{stage}"
        r = (CompositeRegister.like(value, name=stage_name, init=init) if composite
             else Register(value.typ, init=init, name=stage_name))
        if en is None:
            r <<= prev
        else:
            with if_(en):
                r <<= prev
        prev = r
    return r


def _base_name(value, name: Optional[str]) -> str:
    """Prefix for the stage names: ``name=``, else the input's own name, else the variable the
    result is assigned to (``y = pipeline(x)`` -> ``y``), else ``"pipe"``."""
    if name:
        return sanitize_signal_name(name)
    if isinstance(value, Signal) and not getattr(value, "_anonymous_name", False):
        own = value.name  # an anonymous CSE wire (`sig_7`) is renumbered at emission: no basis
    else:
        own = getattr(value, "_suggested_name", None)  # `s = a + b` tags the Expr with "s"
    return sanitize_signal_name(own or infer_expression_binding_name(__file__) or "pipe")
