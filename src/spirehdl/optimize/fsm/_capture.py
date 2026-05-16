"""Capture via `_SharedCache` snapshot/diff.

Records the wires the framework's CSE machinery (`_maybe_share` in `spirehdl.spirehdl`) appends to
`_SharedCache.wires` during a ``with`` block, without resetting the cache (unlike
`optimize.optimize._push_shared_state`, which discards inner wires). The auto-wrapped wires are the "roots" the
encoding-search walker traverses to find State Consts.

The snapshot does *not* attempt to capture the user-named registers / outputs that ``Signal.__ilshift__``-d during
the block — those are passed explicitly to the wrappers (`optimized_fsm(reg, ...)`,
`optimized_encoding(state_cls, ..., signals=[...])`) so a bare `reg <<= S.X` outside any conditional context (which
bypasses `_maybe_share`) is still covered.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spirehdl.spirehdl import Signal


class SharedCacheSnapshot:
    """Records the slice of `_SharedCache.wires` added during a ``with`` block.

    Usage::

        snap = SharedCacheSnapshot()
        with snap:
            # ... user body builds Exprs; every non-trivial Expr lands in _SharedCache.wires via _maybe_share().
            ...
        # snap.new_wires now lists the wires created inside the block.

    Multiple snapshots may be nested; each records its own ``_start_idx`` so inner / outer scopes report disjoint or
    overlapping ranges as appropriate.
    """

    def __init__(self) -> None:
        self._start_idx: int | None = None
        self._end_idx: int | None = None

    def __enter__(self) -> "SharedCacheSnapshot":
        from spirehdl.spirehdl import _SharedCache
        self._start_idx = len(_SharedCache.wires)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        from spirehdl.spirehdl import _SharedCache
        self._end_idx = len(_SharedCache.wires)
        return False

    @property
    def new_wires(self) -> list["Signal"]:
        """Wires added to `_SharedCache.wires` while the snapshot was active.

        Available after exiting the ``with`` block. If queried *during* the block, returns the wires added so far
        (live slice end).
        """
        from spirehdl.spirehdl import _SharedCache
        if self._start_idx is None:
            raise RuntimeError("SharedCacheSnapshot.new_wires accessed before __enter__")
        end = self._end_idx if self._end_idx is not None else len(_SharedCache.wires)
        return _SharedCache.wires[self._start_idx:end]
