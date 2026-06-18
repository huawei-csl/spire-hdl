"""``optimized_fsm`` — Hopcroft state minimisation wrapper.

Usage::

    with optimized_fsm(reg, module=m, minimize=True, outputs=[out]):
        with switch_(reg):
            with case_(S.S0): ...
            ...

On ``__exit__``:

1. Extract the transition table from ``reg._driver``.
2. Run Hopcroft → ``{state_value -> canonical_state_value}`` mapping.
3. Apply the canonical map via ``apply_encoding`` — every merged state's Const is mutated to its representative's
   value, propagating through the already-built Expr DAG.
4. Run ``apply_simplify(module)`` to fold the now-redundant mux branches.

If ``minimize=False`` the wrapper is a no-op.

The wrapper does *not* search encodings — that's ``optimized_encoding``'s job (Proposal E). Compose by nesting if
both are wanted.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from spire.optimize.fsm._capture import SharedCacheSnapshot
from spire.optimize.fsm._emit import apply_encoding
from spire.optimize.fsm._hopcroft import minimize_fsm
from spire.optimize.fsm._table import TooLargeForExhaustiveExtraction, extract_transition_table
from spire.optimize.fsm._walker import find_state_consts

if TYPE_CHECKING:
    from spire.expr import Signal
    from spire.component import Module
    from spire.state import State


def _infer_state_cls(reg: "Signal") -> "type[State] | None":
    """Find the State subclass driving ``reg`` by scanning its Const operands.

    Returns ``None`` if no tagged Const is found — caller may either pass
    ``state_cls`` explicitly or proceed as a no-op.
    """
    if reg._driver is None:
        return None
    # Walk the driver tree looking for any Const tagged with _state_class.
    # We can't use find_state_consts (it needs state_cls) — do a bare DFS.
    from spire.expr import Const, Expr
    seen: set[int] = set()
    def visit(e):
        if id(e) in seen: return None
        seen.add(id(e))
        if isinstance(e, Const):
            cls = getattr(e, "_state_class", None)
            if cls is not None:
                return cls
            return None
        # Recurse via attribute slots we know about.
        for attr in ("a", "b", "sel"):
            child = getattr(e, attr, None)
            if isinstance(child, Expr):
                result = visit(child)
                if result is not None: return result
        if hasattr(e, "parts"):
            for p in e.parts:
                if isinstance(p, Expr):
                    result = visit(p)
                    if result is not None: return result
        # Auto-shared wires (Signals with drivers) — recurse through driver.
        from spire.expr import Signal
        if isinstance(e, Signal) and e._driver is not None:
            return visit(e._driver)
        return None
    return visit(reg._driver)


class optimized_fsm:
    """Hopcroft-minimisation wrapper. Reduces equivalence classes in the FSM whose state register is ``reg``,
    mutating Consts of the corresponding State subclass in place so the existing Expr DAG keeps working.

    Parameters
    ----------
    reg : Signal
        The FSM state register (a `Module.reg(...)` instance).
    module : Module
        The Module containing ``reg``. Required so ``apply_simplify`` can run afterwards to collapse the mux
        branches that become redundant once equivalent states share a value.
    minimize : bool
        Master switch. When False the wrapper is a no-op (just a marker).
    outputs : sequence of Signals
        Moore output signals whose drivers are part of the FSM's observable behaviour. Used in the initial
        Hopcroft partition.
    state_cls : optional State subclass
        Override the auto-inferred state class. Useful when ``reg._driver`` hasn't been populated yet on
        ``__enter__`` (the wrapper auto-defers inference to ``__exit__`` if needed).
    """

    def __init__(
        self,
        reg: "Signal",
        module: "Module",
        *,
        minimize: bool = True,
        outputs: Sequence["Signal"] = (),
        state_cls: "type[State] | None" = None,
    ) -> None:
        self.reg = reg
        self.module = module
        self.minimize = minimize
        self.outputs = list(outputs)
        self._state_cls_override = state_cls
        self._snap: SharedCacheSnapshot | None = None

    def __enter__(self) -> "optimized_fsm":
        self._snap = SharedCacheSnapshot()
        self._snap.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._snap is not None
        self._snap.__exit__(exc_type, exc, tb)
        if exc_type is not None or not self.minimize:
            return False

        state_cls = self._state_cls_override or _infer_state_cls(self.reg)
        if state_cls is None:
            # No state Consts found — user must have driven `reg` from something
            # other than state-typed muxes; nothing for the minimiser to do.
            return False

        try:
            table = extract_transition_table(self.reg, state_cls, outputs=self.outputs)
        except TooLargeForExhaustiveExtraction:
            # Skip minimisation when the input domain is too large to enumerate
            # safely; user can still benefit from encoding search separately.
            return False

        canon = minimize_fsm(table)

        # Only act if any state actually merges with another.
        if all(canon[v] == v for v in canon):
            return False

        # Build a name->canonical-value assignment by mapping each state name to the canonical representative of its
        # current value.
        assignment = {name: canon[state_cls._values[name]] for name in state_cls.names}
        apply_encoding(state_cls, assignment)

        # Fold the now-redundant mux branches. apply_simplify is in-place on the Module's signals + their drivers.
        from spire.simplify import apply_simplify
        apply_simplify(self.module)
        return False
