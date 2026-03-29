"""Auto-config lookup: load best_configs.json and resolve queries.

Provides ``lookup_best_config`` which returns the empirically best
``ArithmeticConfig`` for a given (op, width, signed, objective) tuple.
For widths not in the database the nearest evaluated bitwidth is chosen
using a logarithmic distance metric.

Supported objectives:
    - ``"area"``:  minimize transistor_count  (tiebreak: aig_depth)
    - ``"delay"``: minimize aig_depth         (tiebreak: transistor_count)
    - ``"adp"``:   minimize area-delay product (transistor_count * aig_depth)
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Literal

from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
)

Objective = Literal["area", "delay", "adp"]

OBJECTIVES: list[Objective] = ["area", "delay", "adp"]

_DB_PATH = Path(__file__).parent / "best_configs.json"


@lru_cache(maxsize=1)
def _load_db() -> dict:
    with open(_DB_PATH) as f:
        return json.load(f)


def _nearest_width_log(target: int, available: list[int]) -> int:
    """Find the nearest bitwidth on a logarithmic scale.

    Uses ``|log2(target) - log2(candidate)|``.  Ties are broken by
    preferring the larger width (conservative — slightly overestimates
    complexity rather than underestimating).
    """
    log_target = math.log2(target)
    best = None
    best_dist = float("inf")
    for w in available:
        dist = abs(math.log2(w) - log_target)
        if dist < best_dist or (dist == best_dist and (best is None or w > best)):
            best = w
            best_dist = dist
    return best


def _select_best(rows: list[dict], objective: Objective) -> dict | None:
    """Pick the single best row from *rows* for the given *objective*."""
    if not rows:
        return None

    if objective == "area":
        return min(rows, key=lambda r: (r["transistor_count"], r["aig_depth"]))
    elif objective == "delay":
        return min(rows, key=lambda r: (r["aig_depth"], r["transistor_count"]))
    elif objective == "adp":
        return min(rows, key=lambda r: (
            r["transistor_count"] * r["aig_depth"],
            r["transistor_count"],
        ))
    else:
        raise ValueError(f"Unknown objective: {objective!r}. Use one of {OBJECTIVES}")


def lookup_best_config(
    op: Literal["+", "-", "*"],
    width: int,
    signed: bool,
    objective: Objective = "area",
) -> dict:
    """Look up the empirically best config for the given operation.

    Returns a dict with the config keys (``fsa_opt``, and for multipliers
    also ``ppg_opt`` and ``ppa_opt``), plus metric fields.
    """
    db = _load_db()
    op_key = {"+": "add", "-": "sub", "*": "mul"}[op]
    sign_key = "signed" if signed else "unsigned"

    op_data = db["configs"][op_key]
    available_widths = sorted(int(k) for k in op_data.keys())
    nearest = _nearest_width_log(width, available_widths)

    rows = op_data[str(nearest)][sign_key]
    return _select_best(rows, objective)


def lookup_best_arithmetic_config(
    op: Literal["+", "-", "*"],
    width: int,
    signed: bool,
    objective: Objective = "area",
    full_output_bit: bool = True,
):
    """Return an ``ArithmeticConfig`` for the empirically best configuration."""
    # Import here to avoid circular dependency
    from sprouthdl.arithmetic.int_arithmetic_config import ArithmeticConfig
    from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding

    entry = lookup_best_config(op, width, signed, objective)
    encoding = Encoding.twos_complement if signed else Encoding.unsigned

    # For multipliers, pick optim_type from the winning entry (area vs speed
    # full-adder variant).  For adders it's a no-op but we pass it through.
    optim_type = entry.get("optim_type", "area")

    if op == "*":
        return ArithmeticConfig(
            encoding=encoding,
            optim_type=optim_type,
            fsa_opt=FSAOption[entry["fsa_opt"]],
            full_output_bit=full_output_bit,
            multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
            ppg_opt=PPGOption[entry["ppg_opt"]],
            ppa_opt=PPAOption[entry["ppa_opt"]],
        )
    else:
        return ArithmeticConfig(
            encoding=encoding,
            optim_type=optim_type,
            fsa_opt=FSAOption[entry["fsa_opt"]],
            full_output_bit=full_output_bit,
        )
