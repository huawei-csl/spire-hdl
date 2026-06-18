"""Walker — find every State Const reachable from a set of root Exprs.

The walker is used for two purposes:

1. **Inventory.** Confirm that a `with` block actually contains references to the state class — if it doesn't, the
   wrapper has nothing to do and exits silently.
2. **Input discovery.** While walking, collect every non-state-Const ``Signal`` leaf. These are the FSM's free inputs
   whose values the transition-table extractor will enumerate.

We do **not** need substitution sites for the encoding-rewrite step: ``State`` Consts are shared object instances
(`StateCls.NAME` is one object referenced everywhere), so mutating ``cls.NAME.value`` in place propagates to every
reference automatically. ``apply_encoding`` in ``_emit.py`` relies on that property.

The walker reuses the existing ``ExprVisitor`` base class (identity-cached DFS) so each shared sub-tree is visited
only once even when the same wire appears in multiple places.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from spire.expr import (
    Concat, Const, Expr, Op1, Op2, Resize, Signal, Slice, Ternary,
)
from spire.visitor import ExprVisitor

if TYPE_CHECKING:
    from spire.state import State


def is_state_const(expr: Expr, state_cls: "type[State]") -> bool:
    """The canonical detection predicate — matches by provenance sentinel,
    not by (value, width), so literal zeros / mask constants are rejected."""
    return isinstance(expr, Const) and getattr(expr, "_state_class", None) is state_cls


class _StateConstFinder(ExprVisitor[None]):
    """Walks an Expr DAG, recording every State Const matching `state_cls`
    and every non-Const, non-state Signal leaf it sees.
    """

    def __init__(self, state_cls: "type[State]") -> None:
        super().__init__()
        self.state_cls = state_cls
        self.found_consts: list[Const] = []
        self.input_signals: list[Signal] = []
        self._seen_signal_ids: set[int] = set()

    # Leaves --------------------------------------------------------------

    def visit_const(self, e: Const) -> None:
        if is_state_const(e, self.state_cls):
            self.found_consts.append(e)

    def visit_signal(self, e: Signal) -> None:
        if e._driver is not None:
            # Auto-shared wires (sig_N) wrap inner Exprs — recurse through. User-named signals (inputs, regs, outputs)
            # usually also have drivers, but their drivers were built in user code and may legitimately contain State
            # Consts the wrapper cares about.
            if id(e) not in self._seen_signal_ids:
                self._seen_signal_ids.add(id(e))
                self.visit(e._driver)
        else:
            # Leaf signal — treat as a free input candidate (drivers attached outside the with-block are still
            # considered inputs from the wrapper's perspective).
            if id(e) not in self._seen_signal_ids:
                self._seen_signal_ids.add(id(e))
                self.input_signals.append(e)

    # Interior nodes -------------------------------------------------------

    def visit_op1(self, e: Op1) -> None:
        self.visit(e.a)

    def visit_op2(self, e: Op2) -> None:
        self.visit(e.a)
        self.visit(e.b)

    def visit_ternary(self, e: Ternary) -> None:
        self.visit(e.sel)
        self.visit(e.a)
        self.visit(e.b)

    def visit_concat(self, e: Concat) -> None:
        for p in e.parts:
            self.visit(p)

    def visit_slice(self, e: Slice) -> None:
        self.visit(e.a)

    def visit_resize(self, e: Resize) -> None:
        self.visit(e.a)


def walk(roots: Iterable[Expr], state_cls: "type[State]") -> _StateConstFinder:
    """Walk every Expr in `roots`, returning the finder with populated ``found_consts`` and ``input_signals`` lists.

    Both lists may contain duplicates by *object identity*; callers that need deduplication should run
    ``{id(x): x for x in lst}.values()`` themselves.
    """
    finder = _StateConstFinder(state_cls)
    for root in roots:
        if root is None:
            continue
        # Treat a bare Const root as if it were a child of a virtual parent — the visitor's visit_const handles the
        # detection.
        finder.visit(root)
    return finder


def find_state_consts(roots: Iterable[Expr], state_cls: "type[State]") -> list[Const]:
    """Shortcut: just the State Consts (no input-signal discovery).
    Returns a deduplicated list (one entry per unique object identity).
    """
    finder = walk(roots, state_cls)
    seen: set[int] = set()
    out: list[Const] = []
    for c in finder.found_consts:
        if id(c) not in seen:
            seen.add(id(c))
            out.append(c)
    return out


def find_input_signals(
    roots: Iterable[Expr],
    state_cls: "type[State]",
    *,
    exclude: Iterable[Signal] = (),
) -> list[Signal]:
    """Return all non-state Signal leaves reachable from ``roots``, minus any listed in ``exclude``. Used by
    ``extract_transition_table`` to discover the FSM's free inputs.

    A ``Signal`` is treated as a leaf when:
      * it has no driver, or
      * it's a user-named ``Signal`` with `kind == "input"` (truly external).

    Auto-shared CSE wires (``_auto_generated=True``) are recursed *through* rather than reported, because their
    drivers are the actual logic.
    """
    finder = walk(roots, state_cls)
    excluded_ids = {id(s) for s in exclude}
    seen: set[int] = set()
    out: list[Signal] = []
    for s in finder.input_signals:
        if id(s) in excluded_ids or id(s) in seen:
            continue
        # Skip auto-shared CSE wires — they are not "free inputs".
        if getattr(s, "_auto_generated", False):
            continue
        seen.add(id(s))
        out.append(s)
    return out
