"""Thread-safe build context for Sprout-HDL.

All mutable state that was previously stored in module-level globals
(_active_conditions, _pending_if_chain, _switch_stack, CSE state) now lives
inside a ``BuildContext`` instance.  A thread-local default context is
created lazily so that existing single-threaded code keeps working without
any changes.  Multi-threaded or concurrent builds can create isolated
contexts via ``BuildContext()`` and either use them as a context manager
or pass them explicitly.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sprouthdl.sprouthdl import Expr, Signal


# ---------------------------------------------------------------------------
# BuildContext
# ---------------------------------------------------------------------------

class BuildContext:
    """Holds all mutable compilation state for one HDL build."""

    def __init__(self) -> None:
        # --- CSE (shared sub-expression) state ---
        self.cse_counts: dict[int, int] = {}
        self.cse_expr2sig: dict[int, "Signal"] = {}
        self.cse_wires: list["Signal"] = []
        self.cse_index: int = 0

        # --- Control-structure state ---
        self.active_conditions: List["Expr"] = []
        self.pending_if_chain: Optional[object] = None  # Optional[_IfChain]
        self.switch_stack: List[object] = []             # List[_SwitchState]

    # -- CSE helpers ----------------------------------------------------------

    def reset_cse(self) -> None:
        """Clear CSE state (call before emitting each Verilog module)."""
        self.cse_counts.clear()
        self.cse_expr2sig.clear()
        self.cse_wires.clear()
        self.cse_index = 0

    # -- context-manager interface -------------------------------------------

    def __enter__(self) -> "BuildContext":
        _ctx_stack.push(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ctx_stack.pop(self)
        return False


# ---------------------------------------------------------------------------
# Thread-local context stack
# ---------------------------------------------------------------------------

class _ContextStack(threading.local):
    """Per-thread stack of active BuildContext instances."""

    def __init__(self) -> None:
        super().__init__()
        self._stack: list[BuildContext] = []

    @property
    def current(self) -> BuildContext:
        """Return the innermost active context, creating a default if needed."""
        if not self._stack:
            self._stack.append(BuildContext())
        return self._stack[-1]

    def push(self, ctx: BuildContext) -> None:
        self._stack.append(ctx)

    def pop(self, ctx: BuildContext) -> None:
        if not self._stack or self._stack[-1] is not ctx:
            raise RuntimeError("BuildContext stack corruption")
        self._stack.pop()


_ctx_stack = _ContextStack()


def current_build_context() -> BuildContext:
    """Return the active BuildContext for the calling thread."""
    return _ctx_stack.current
