"""``apply_encoding`` — in-place rewrite of State Consts.

Because every ``StateCls.NAME`` is a single shared ``Const`` object instance, mutating ``cls.NAME.value`` propagates
to every reference in the user's expression DAG automatically. We don't need to walk the DAG to substitute — we only
need to walk it (via ``_walker``) for verification / input discovery.

This module also exposes ``snapshot_encoding`` / ``restore_encoding`` so the encoding-search loop can try a
candidate, measure cost, then revert before trying the next.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from spirehdl.spirehdl_state import State


def snapshot_encoding(state_cls: "type[State]") -> dict[str, int]:
    """Return a copy of the current ``{name: value}`` mapping for ``state_cls``.

    Pass this to ``restore_encoding`` to undo a tentative apply.
    """
    return dict(state_cls._values)


def apply_encoding(
    state_cls: "type[State]",
    assignment: Mapping[str, int],
    *,
    width: int | None = None,
) -> None:
    """Mutate each ``state_cls.NAME.value`` to ``assignment[name]``.

    Because the Const objects are shared by reference, the change propagates to every Expr that holds one of them.

    Parameters
    ----------
    state_cls : subclass of State
        The state class whose Consts to re-encode.
    assignment : mapping name -> new value
        Must cover every name in ``state_cls.names``; extra keys are ignored with a warning-free pass (callers may
        pass a superset from a class with optional aliases).
    width : optional int
        If given, must match ``state_cls._width``. Width-changing re-encoding (e.g. switching a BINARY 2-bit class
        to ONEHOT 4-bit) is not yet supported and raises ``NotImplementedError``.
    """
    if width is not None and width != state_cls._width:
        raise NotImplementedError(f"width change ({state_cls._width} → {width}) not yet supported")

    for name in state_cls.names:
        if name not in assignment:
            raise ValueError(f"apply_encoding: assignment missing state {name!r}")
        new_value = int(assignment[name])
        c = getattr(state_cls, name)
        c.value = new_value
        state_cls._values[name] = new_value


def restore_encoding(
    state_cls: "type[State]",
    snapshot: Mapping[str, int],
) -> None:
    """Convenience alias for ``apply_encoding(state_cls, snapshot)``."""
    apply_encoding(state_cls, snapshot)
