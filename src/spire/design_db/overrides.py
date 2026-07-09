"""Temporary selection overrides — force ``select_design`` to specific design_ids.

Two front doors over one registry, consulted by ``select_design`` whenever no explicit
``pin=`` is given:

- ``$SPIREHDL_DB_PINS`` — a JSON object ``{slot: design_id}`` in the environment. Crosses
  process boundaries: set it around a compile subprocess and every ``@from_design_db`` splice
  in that process resolves to the pinned designs (rtlscout's composition-space tool sweeps
  splice combinations this way).
- ``selection_overrides({...})`` — the same thing as a scoped, in-process context manager.

``slot`` keys are exact spec_keys or manifest names; values are exact ``design_id``s (the
``pin=`` rules apply downstream: unknown ⇒ error, no prefixes). Overrides are what-if
compiles, not library state: overridden selections are **never recorded** in the manifest,
and an explicit source-level ``pin=`` always wins over an override.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from typing import Dict, Iterator, List, Mapping

from spire.design_db.store import DesignDBError

PINS_ENV = "SPIREHDL_DB_PINS"

_local = threading.local()


def _stack() -> List[Dict[str, str]]:
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    return stack


def _from_env() -> Dict[str, str]:
    raw = os.environ.get(PINS_ENV, "").strip()
    if not raw:
        return {}
    try:
        pins = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DesignDBError(f"${PINS_ENV} is not valid JSON: {exc}") from None
    if (not isinstance(pins, dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in pins.items())):
        raise DesignDBError(f"${PINS_ENV} must be a JSON object of slot -> design_id strings")
    return pins


def current_overrides() -> Dict[str, str]:
    """The active override map: env pins overlaid by ``selection_overrides`` scopes
    (innermost wins). Empty dict when nothing is active — the common case, checked cheaply."""
    pins = _from_env()
    for scope in _stack():
        pins.update(scope)
    return pins


@contextmanager
def selection_overrides(pins: Mapping[str, str]) -> Iterator[None]:
    """Scope-force selections: every ``select_design`` (and therefore every
    ``@from_design_db`` splice) inside the block resolves the listed slots to the given
    design_ids::

        with selection_overrides({"adder8": "verilog:e218599799"}):
            Top().to_verilog_file("design.v")
    """
    _stack().append(dict(pins))
    try:
        yield
    finally:
        _stack().pop()
