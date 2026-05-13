"""Step 9: cost oracle for encoding-search candidates.

Given a Module, a State subclass, and an objective metric, returns a callable
``cost_fn(assignment)`` that:

1. snapshots the current State encoding,
2. applies ``assignment`` via ``apply_encoding``,
3. measures the chosen objective on the in-memory ``Module`` via in-process
   pyosys (cells/wires/transistors) or aigverse (aig_gates/aig_depth),
4. restores the original encoding,
5. returns the cost as a ``float`` (``inf`` on any failure so the search rejects).

Both pyosys and aigverse are pip dependencies (pinned in ``requirements.txt``),
so the cost oracle has **no runtime dependency on a `yosys` binary** — tests
that exercise it run unconditionally.

Callers who want a richer cost (Sky130 ADP, ABC depth via a custom recipe,
total power, …) can supply their own ``cost_fn`` to ``optimized_encoding``
instead of using ``make_cost_fn`` here.
"""
from __future__ import annotations

from typing import Callable, Literal, TYPE_CHECKING

from spirehdl.fsm._emit import apply_encoding, restore_encoding, snapshot_encoding
from spirehdl.helpers import get_aig_stats, get_yosys_metrics

if TYPE_CHECKING:
    from spirehdl.spirehdl_module import Module
    from spirehdl.spirehdl_state import State


Objective = Literal[
    "cells",         # post-synth Yosys cell count
    "wires",         # post-synth Yosys wire count
    "transistors",   # Yosys CMOS-tech estimated transistor count
    "aig_gates",     # AIG AND-gate count (aigverse `Aig.gates()` size)
    "aig_depth",     # AIG critical-path depth (aigverse DepthAig num_levels)
]

_AIG_OBJECTIVES = {"aig_gates", "aig_depth"}


def _measure(module: "Module", objective: Objective) -> float:
    """Compute one synthesis metric on `module` in-process.

    ``via_aig=False`` keeps the yosys flow on the direct read-Verilog path —
    no aigverse-side rewriting before the synth — so the cell / wire /
    transistor counts match what `yosys; clean -purge; stat` reports
    standalone (the same recipe the rtl_rewriter benchmark uses).
    """
    if objective in _AIG_OBJECTIVES:
        s = get_aig_stats(module)
        if objective == "aig_gates":
            return float(s["num_gates"])
        return float(s["depth"])

    s = get_yosys_metrics(module, via_aig=False)
    if objective == "cells":       return float(s["num_cells"])
    if objective == "wires":       return float(s["num_wires"])
    if objective == "transistors": return float(s["estimated_num_transistors"])
    raise ValueError(f"unknown objective {objective!r}")


def make_cost_fn(
    module: "Module",
    state_cls: "type[State]",
    objective: Objective = "cells",
) -> Callable[[dict[str, int]], float]:
    """Build a cost function for ``search_encoding``.

    The returned callable mutates the State class's Const values to the
    assignment, measures the chosen ``objective`` on ``module`` in-process,
    then restores the original encoding before returning. Even on exceptions
    the original encoding is restored (try/finally), so the State class is
    always left in its declared state after the search.
    """
    base_snapshot = snapshot_encoding(state_cls)

    def cost_fn(assignment: dict[str, int]) -> float:
        try:
            apply_encoding(state_cls, assignment)
            return _measure(module, objective)
        except Exception:
            return float("inf")
        finally:
            restore_encoding(state_cls, base_snapshot)

    return cost_fn


# Backward-compatibility alias for callers that still import the old name.
make_yosys_cost_fn = make_cost_fn
