"""Cost oracle for encoding-search candidates.

Given a Module, a State subclass, and an objective metric, returns a callable ``cost_fn(assignment)`` that:

1. snapshots the current State encoding,
2. applies ``assignment`` via ``apply_encoding``,
3. measures the chosen objective on the in-memory ``Module`` via in-process pyosys (cells/wires/transistors) or
   aigverse (aig_gates/aig_depth),
4. restores the original encoding,
5. returns the cost as a ``float`` (``inf`` on any failure so the search rejects).

Both pyosys and aigverse are pip dependencies (pinned in ``requirements.txt``), so the cost oracle has **no runtime
dependency on a `yosys` binary** — tests that exercise it run unconditionally.

Callers who want a richer cost (Sky130 ADP, ABC depth via a custom recipe, total power, …) can supply their own
``cost_fn`` to ``optimized_encoding`` instead of using ``make_cost_fn`` here.
"""
from __future__ import annotations

from typing import Callable, Literal, TYPE_CHECKING

from spire.optimize.fsm._emit import apply_encoding, restore_encoding, snapshot_encoding
from spire.helpers import get_aig_stats, get_yosys_metrics

if TYPE_CHECKING:
    from spire.component import Module
    from spire.state import State


Objective = Literal[
    "cells",         # post-synth Yosys cell count
    "wires",         # post-synth Yosys wire count
    "transistors",   # Yosys CMOS-tech estimated transistor count
    "aig_gates",     # AIG AND-gate count (aigverse `Aig.gates()` size)
    "aig_depth",     # AIG critical-path depth (aigverse DepthAig num_levels)
    "adp_proxy",     # aig_gates * aig_depth — a PDK-free area×delay proxy
]

_AIG_OBJECTIVES = {"aig_gates", "aig_depth", "adp_proxy"}


def _measure(module: "Module", objective: Objective) -> float:
    """Compute one synthesis metric on `module` in-process.

    ``via_aig=False`` keeps the yosys flow on the direct read-Verilog path — no aigverse-side rewriting before the
    synth — so the cell / wire / transistor counts match what `yosys; clean -purge; stat` reports standalone (the
    same recipe the rtl_rewriter benchmark uses).

    For AIG objectives (incl. ``adp_proxy``) ``module`` must be **combinational**
    — aigverse rejects latches, so the caller passes the combinational cone
    (see ``make_cost_fn(bit_level_emit=True)``), never the sequential FSM.
    """
    if objective in _AIG_OBJECTIVES:
        s = get_aig_stats(module)
        if objective == "aig_gates":
            return float(s["num_gates"])
        if objective == "aig_depth":
            return float(s["depth"])
        # adp_proxy: area proxy (gates) × delay proxy (depth)
        return float(s["num_gates"]) * float(s["depth"])

    s = get_yosys_metrics(module, via_aig=False)
    if objective == "cells":       return float(s["num_cells"])
    if objective == "wires":       return float(s["num_wires"])
    if objective == "transistors": return float(s["estimated_num_transistors"])
    raise ValueError(f"unknown objective {objective!r}")


def make_cost_fn(
    module: "Module",
    state_cls: "type[State]",
    objective: Objective = "cells",
    *,
    bit_level_emit: bool = False,
    dont_cares: bool = False,
) -> Callable[[dict[str, int]], float]:
    """Build a cost function for ``search_encoding``.

    The returned callable mutates the State class's Const values to the assignment, measures the chosen
    ``objective`` in-process, then restores the original encoding before returning. Even on
    exceptions the original encoding is restored (try/finally), so the State class is always left in its declared
    state after the search.

    When ``bit_level_emit`` is True (or the objective is AIG-based, which is only
    meaningful on combinational logic), each candidate is measured on the
    **bit-level-minimised combinational cone** for that encoding — the form that
    will actually be emitted — rather than on the structural mux tree (which
    Yosys would re-encode, flattening the cost landscape). This is what makes
    the encoding choice visible to the search. ``adp_proxy`` always builds the
    cone (it is a sequential-FSM objective); ``cells``/``transistors``/
    ``aig_gates``/``aig_depth`` build it only when ``bit_level_emit`` is set —
    otherwise they measure the module directly, preserving their original
    behaviour (``aig_gates``/``aig_depth`` work on combinational state-select
    modules that have no state register and thus no cone).
    """
    base_snapshot = snapshot_encoding(state_cls)
    use_cone = bit_level_emit or objective == "adp_proxy"

    def cost_fn(assignment: dict[str, int]) -> float:
        try:
            apply_encoding(state_cls, assignment)
            if use_cone:
                from spire.optimize.fsm._minimize_emit import build_comb_cone
                target = build_comb_cone(module, state_cls, dont_cares=dont_cares)
            else:
                target = module
            return _measure(target, objective)
        except Exception:
            return float("inf")
        finally:
            restore_encoding(state_cls, base_snapshot)

    return cost_fn


# Backward-compatibility alias for callers that still import the old name.
make_yosys_cost_fn = make_cost_fn
