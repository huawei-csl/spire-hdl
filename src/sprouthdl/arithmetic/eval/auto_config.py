"""Auto-config lookup: load best_configs.json and resolve queries.

Provides ``lookup_best_config`` which returns the empirically best
``ArithmeticConfig`` for a given (op, a_w, b_w, signed, objective) tuple.

For widths not in the database each width is snapped independently to
the nearest evaluated width using a logarithmic distance metric.

For commutative ops (``+``, ``*``), both orientations are checked and
the better one is returned together with a ``swap`` flag.

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
    """Find the nearest bitwidth on a logarithmic scale."""
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


def _snap_width_pair(a_w: int, b_w: int, op_data: dict) -> tuple[int, int]:
    """Snap each width independently to the nearest available width in the DB."""
    # Collect all individual widths present in the DB keys
    available = set()
    for key in op_data:
        parts = key.split("x")
        available.add(int(parts[0]))
        available.add(int(parts[1]))
    available = sorted(available)
    if not available:
        return a_w, b_w
    return _nearest_width_log(a_w, available), _nearest_width_log(b_w, available)


def _lookup_rows(op_data: dict, a_w: int, b_w: int, sign_key: str) -> list[dict]:
    """Get rows for a specific width pair, or empty list if not found."""
    w_key = f"{a_w}x{b_w}"
    return op_data.get(w_key, {}).get(sign_key, [])


def lookup_best_config(
    op: Literal["+", "-", "*"],
    a_w: int,
    b_w: int,
    signed: bool,
    objective: Objective = "area",
) -> tuple[dict, bool]:
    """Look up the empirically best config for the given operation.

    Returns ``(config_dict, swap)`` where *swap* is True if operands
    should be swapped (i.e. the ``b_w x a_w`` orientation was better).
    For subtraction, swap is always False.
    """
    db = _load_db()
    op_key = {"+": "add", "-": "sub", "*": "mul"}[op]
    sign_key = "signed" if signed else "unsigned"

    op_data = db["configs"][op_key]
    snapped_a, snapped_b = _snap_width_pair(a_w, b_w, op_data)

    rows_ab = _lookup_rows(op_data, snapped_a, snapped_b, sign_key)

    if op in ("+", "*") and snapped_a != snapped_b:
        # Commutative: try both orientations
        rows_ba = _lookup_rows(op_data, snapped_b, snapped_a, sign_key)
        best_ab = _select_best(rows_ab, objective)
        best_ba = _select_best(rows_ba, objective)

        if best_ab is None and best_ba is None:
            return None, False
        if best_ba is None:
            return best_ab, False
        if best_ab is None:
            return best_ba, True

        # Compare the two orientations
        combined = _select_best([best_ab, best_ba], objective)
        swap = combined is best_ba
        return combined, swap
    else:
        # Non-commutative (subtraction) or symmetric widths
        best = _select_best(rows_ab, objective)
        return best, False


def lookup_best_arithmetic_config(
    op: Literal["+", "-", "*"],
    a_w: int,
    b_w: int,
    signed: bool,
    objective: Objective = "area",
    full_output_bit: bool = True,
):
    """Return ``(ArithmeticConfig, swap)`` for the empirically best configuration.

    *swap* is True if the caller should swap operand a and b before
    feeding them into the replacement component.
    """
    from sprouthdl.arithmetic.int_arithmetic_config import ArithmeticConfig
    from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding

    entry, swap = lookup_best_config(op, a_w, b_w, signed, objective)
    if entry is None:
        # Fallback to default config
        encoding = Encoding.twos_complement if signed else Encoding.unsigned
        return ArithmeticConfig(encoding=encoding, full_output_bit=full_output_bit), False

    encoding = Encoding.twos_complement if signed else Encoding.unsigned
    optim_type = entry.get("optim_type", "area")

    if op == "*":
        cfg = ArithmeticConfig(
            encoding=encoding,
            optim_type=optim_type,
            fsa_opt=FSAOption[entry["fsa_opt"]],
            full_output_bit=full_output_bit,
            multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
            ppg_opt=PPGOption[entry["ppg_opt"]],
            ppa_opt=PPAOption[entry["ppa_opt"]],
        )
    else:
        cfg = ArithmeticConfig(
            encoding=encoding,
            optim_type=optim_type,
            fsa_opt=FSAOption[entry["fsa_opt"]],
            full_output_bit=full_output_bit,
        )
    return cfg, swap
