"""Step 9: cost oracle for encoding-search candidates.

Given a Module, a State subclass, and an objective metric, returns a
callable ``cost_fn(assignment)`` that:

1. snapshots the current State encoding,
2. applies ``assignment`` via ``apply_encoding``,
3. writes Verilog to a tempdir and runs Yosys ``synth; clean -purge; stat``,
4. parses the stat output for the objective metric,
5. restores the original encoding,
6. returns the cost as a ``float`` (``inf`` on synth failure so the search rejects).

The yosys invocation is intentionally minimal and self-contained — spire-hdl
keeps no hard dependency on rtl_scout's ``core.cost`` module. Callers who
want a richer cost (transistors / ABC depth / Sky130 ADP) can supply their
own ``cost_fn`` to ``optimized_encoding`` instead of using this default.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Literal, TYPE_CHECKING

from spirehdl.fsm._emit import apply_encoding, restore_encoding, snapshot_encoding

if TYPE_CHECKING:
    from spirehdl.spirehdl_module import Module
    from spirehdl.spirehdl_state import State


Objective = Literal["cells", "wires", "transistors"]

_WIRES_RE = re.compile(r"Number of wires:\s+(\d+)")
_CELLS_RE = re.compile(r"Number of cells:\s+(\d+)")


def _yosys_stat(verilog_path: Path, top_module: str, timeout: int = 30) -> dict[str, float]:
    """Run ``yosys`` synth+stat on ``verilog_path`` and return parsed stats."""
    script = (
        "design -reset; "
        f"read_verilog -sv {verilog_path}; "
        f"hierarchy -top {top_module}; "
        "synth; clean -purge; "
        f"stat -top {top_module}"
    )
    try:
        result = subprocess.run(
            ["yosys", "-p", script],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"yosys binary not found: {exc}")
    except subprocess.TimeoutExpired:
        return {"cells": float("inf"), "wires": float("inf"), "transistors": float("inf")}

    combined = result.stdout + "\n" + result.stderr
    wires = _WIRES_RE.findall(combined)
    cells = _CELLS_RE.findall(combined)
    if not wires or not cells:
        return {"cells": float("inf"), "wires": float("inf"), "transistors": float("inf")}
    # Use the last block — corresponds to the top-level stat summary.
    return {
        "cells": float(cells[-1]),
        "wires": float(wires[-1]),
        # `stat -tech cmos` would give transistors, but spire-hdl doesn't ship
        # that as part of the simple path. Callers needing transistors should
        # pass a custom cost_fn.
        "transistors": float("nan"),
    }


def make_yosys_cost_fn(
    module: "Module",
    state_cls: "type[State]",
    objective: Objective = "cells",
    *,
    timeout: int = 30,
    simplify_emit: bool = True,
) -> Callable[[dict[str, int]], float]:
    """Build a cost function for ``search_encoding``.

    The returned callable mutates the State class's Const values to the
    assignment, runs yosys, then restores the original encoding before
    returning. Even on exceptions the original encoding is restored
    (try/finally), so the State class is always left in its declared state
    after the search.

    ``objective`` selects which scalar from the stat output to return.
    """
    base_snapshot = snapshot_encoding(state_cls)

    def cost_fn(assignment: dict[str, int]) -> float:
        try:
            apply_encoding(state_cls, assignment)
            with tempfile.TemporaryDirectory(prefix="fsm_cost_") as tmp:
                v_path = Path(tmp) / f"{module.name}.v"
                module.to_verilog_file(str(v_path), simplify=simplify_emit)
                stats = _yosys_stat(v_path, module.name, timeout=timeout)
            return stats.get(objective, float("inf"))
        except Exception:
            return float("inf")
        finally:
            restore_encoding(state_cls, base_snapshot)

    return cost_fn
