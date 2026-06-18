"""Auto-config lookup: load best_configs.csv (or .json) and resolve queries.

Provides ``lookup_best_config`` which returns the empirically best
``ArithmeticConfig`` for a given (op, a_w, b_w, signed, objective, metric) tuple.

For commutative ops (``+``, ``*``), both orientations are checked and
the better one is returned together with a ``swap`` flag.

``metric`` is the final metric being optimized for (default ``DEFAULT_LOOKUP_METRIC``, currently ``"transistors_heavy"``):
    - ``"transistors_heavy"``: yosys transistor count under the full ``synth; clean -purge`` pipeline (column ``transistor_count_heavy``).
    - ``"transistors"``: yosys transistor count under the lite ``abc -fast`` pipeline (column ``transistor_count``).
    - ``"aig_count"``: AIG gate count, post-synth via aigverse (column ``num_aig_gates``).

The ``objective`` parameter then says how to combine the metric with depth:
    - ``"area"``:  minimize the chosen metric (tiebreak: aig_depth)
    - ``"delay"``: minimize aig_depth         (tiebreak: chosen metric)
    - ``"adp"``:   minimize metric × aig_depth (tiebreak: metric)
"""

from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Literal

from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
)

Objective = Literal["area", "delay", "adp"]
Metric = Literal["transistors", "transistors_heavy", "aig_count"]

OBJECTIVES: list[Objective] = ["area", "delay", "adp"]
METRICS: list[Metric] = ["transistors", "transistors_heavy", "aig_count"]

# Default size metric used everywhere the caller doesn't specify one.
DEFAULT_LOOKUP_METRIC: Metric = "transistors_heavy"

# Map a Metric name to the CSV column that holds its value.
_METRIC_COLUMN: dict[Metric, str] = {
    "transistors":       "transistor_count",
    "transistors_heavy": "transistor_count_heavy",
    "aig_count":         "num_aig_gates",
}

_DB_DIR = Path(__file__).parent
_INT_FIELDS = {"transistor_count", "transistor_count_heavy", "aig_depth",
               "num_aig_gates", "a_w", "b_w"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> dict[tuple, list[dict]]:
    """Load CSV into grouped dict keyed by (op, a_w_str, b_w_str, signed)."""
    grouped: dict[tuple, list[dict]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["op"], row["a_w"], row["b_w"], row["signed"])
            clean = {}
            for k, v in row.items():
                if k in ("op", "a_w", "b_w", "signed"):
                    continue
                if v == "":
                    continue
                clean[k] = int(v) if k in _INT_FIELDS else v
            grouped.setdefault(key, []).append(clean)
    return grouped


def _load_json(path: Path) -> dict[tuple, list[dict]]:
    """Load JSON (v3 nested format) into same grouped dict."""
    with open(path) as f:
        db = json.load(f)
    grouped: dict[tuple, list[dict]] = {}
    for op, widths in db["configs"].items():
        for w_key, signs in widths.items():
            a_w, b_w = w_key.split("x")
            for sign_key, rows in signs.items():
                key = (op, a_w, b_w, sign_key)
                grouped[key] = []
                for r in rows:
                    clean = {}
                    for k, v in r.items():
                        if v == "" or v is None:
                            continue
                        clean[k] = int(v) if k in _INT_FIELDS else v
                    grouped[key].append(clean)
    return grouped


@lru_cache(maxsize=1)
def _load_db() -> dict[tuple, list[dict]]:
    """Auto-detect CSV or JSON and load."""
    csv_path = _DB_DIR / "best_configs.csv"
    json_path = _DB_DIR / "best_configs.json"
    if csv_path.exists():
        return _load_csv(csv_path)
    elif json_path.exists():
        return _load_json(json_path)
    else:
        raise FileNotFoundError(f"No config database found in {_DB_DIR}")


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

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


def _metric_col(metric: Metric) -> str:
    if metric not in _METRIC_COLUMN:
        raise ValueError(f"Unknown metric: {metric!r}. Use one of {METRICS}")
    return _METRIC_COLUMN[metric]


def _row_metric(row: dict, metric: Metric) -> int | None:
    """Return the row's value for *metric*, or None if the column is missing
    (e.g. old DB without `transistor_count_heavy`)."""
    col = _metric_col(metric)
    return row.get(col)


def _select_best(rows: list[dict], objective: Objective,
                 metric: Metric = DEFAULT_LOOKUP_METRIC) -> dict | None:
    """Pick the single best row from *rows* for the given *(objective, metric)*.

    Rows that don't carry the requested *metric* column are filtered out.
    """
    if not rows:
        return None
    # Filter rows that have the requested metric column populated.
    valid = [r for r in rows if _row_metric(r, metric) is not None]
    if not valid:
        return None
    col = _metric_col(metric)
    if objective == "area":
        return min(valid, key=lambda r: (r[col], r["aig_depth"]))
    elif objective == "delay":
        return min(valid, key=lambda r: (r["aig_depth"], r[col]))
    elif objective == "adp":
        return min(valid, key=lambda r: (r[col] * r["aig_depth"], r[col]))
    else:
        raise ValueError(f"Unknown objective: {objective!r}. Use one of {OBJECTIVES}")


def _snap_width_pair(a_w: int, b_w: int, db: dict[tuple, list]) -> tuple[int, int]:
    """Snap each width to the nearest available in the DB."""
    available = set()
    for key in db:
        available.add(int(key[1]))
        available.add(int(key[2]))
    available_sorted = sorted(available)
    if not available_sorted:
        return a_w, b_w
    return _nearest_width_log(a_w, available_sorted), _nearest_width_log(b_w, available_sorted)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_best_config(
    op: Literal["+", "-", "*"],
    a_w: int,
    b_w: int,
    signed: bool,
    objective: Objective = "area",
    metric: Metric = DEFAULT_LOOKUP_METRIC,
) -> tuple[dict | None, bool]:
    """Look up the empirically best config for the given operation.

    Returns ``(config_dict, swap)`` where *swap* is True if operands
    should be swapped (i.e. the ``b_w x a_w`` orientation was better).
    """
    db = _load_db()
    op_key = {"+": "add", "-": "sub", "*": "mul"}[op]
    sign_key = "signed" if signed else "unsigned"
    snapped_a, snapped_b = _snap_width_pair(a_w, b_w, db)

    rows_ab = db.get((op_key, str(snapped_a), str(snapped_b), sign_key), [])

    if op in ("+", "*") and snapped_a != snapped_b:
        rows_ba = db.get((op_key, str(snapped_b), str(snapped_a), sign_key), [])
        best_ab = _select_best(rows_ab, objective, metric)
        best_ba = _select_best(rows_ba, objective, metric)

        if best_ab is None and best_ba is None:
            return None, False
        if best_ba is None:
            return best_ab, False
        if best_ab is None:
            return best_ba, True

        combined = _select_best([best_ab, best_ba], objective, metric)
        swap = combined is best_ba
        return combined, swap
    else:
        return _select_best(rows_ab, objective, metric), False


def lookup_best_mia_config(
    n_inputs: int,
    n_bits: int,
    signed: bool,
    objective: Objective = "area",
    metric: Metric = DEFAULT_LOOKUP_METRIC,
) -> dict | None:
    """Look up the best multi-input add config for ``y = sum(operands)``.

    Snaps both ``n_inputs`` (to the nearest available chain length in the
    DB) and ``n_bits`` (to the nearest available width). Returns the row
    with the best (objective, metric) score, or None if the DB carries no
    MIA rows (e.g. eval was run without the `mia` sweep).
    """
    db = _load_db()
    sign_key = "signed" if signed else "unsigned"
    # Collect all chain lengths present (op keys of the form "miaN").
    available_n = sorted({int(k[0][3:]) for k in db
                          if k[0].startswith("mia") and k[0][3:].isdigit()})
    if not available_n:
        return None
    snapped_n = _nearest_width_log(n_inputs, available_n)
    op_key = f"mia{snapped_n}"
    available_w = sorted({int(k[1]) for k in db if k[0] == op_key})
    if not available_w:
        return None
    snapped_w = _nearest_width_log(n_bits, available_w)
    rows = db.get((op_key, str(snapped_w), str(snapped_w), sign_key), [])
    return _select_best(rows, objective, metric)


def lookup_best_mac_config(
    n_bits: int,
    signed: bool,
    objective: Objective = "area",
    metric: Metric = DEFAULT_LOOKUP_METRIC,
) -> dict | None:
    """Look up the best fused MAC config for y = a*b + c (symmetric widths)."""
    db = _load_db()
    sign_key = "signed" if signed else "unsigned"
    snapped = _nearest_width_log(n_bits, sorted({
        int(k[1]) for k in db if k[0] == "mac"
    } or {n_bits}))
    rows = db.get(("mac", str(snapped), str(snapped), sign_key), [])
    return _select_best(rows, objective, metric)


def lookup_best_dot_config(
    n_terms: int,
    n_bits: int,
    signed: bool,
    objective: Objective = "area",
    metric: Metric = DEFAULT_LOOKUP_METRIC,
) -> dict | None:
    """Look up the best fused inner product config for y = sum(ai*bi)."""
    db = _load_db()
    sign_key = "signed" if signed else "unsigned"
    op_key = f"dot{n_terms}"
    available = sorted({int(k[1]) for k in db if k[0] == op_key} or {n_bits})
    snapped = _nearest_width_log(n_bits, available)
    rows = db.get((op_key, str(snapped), str(snapped), sign_key), [])
    return _select_best(rows, objective, metric)


def _objective_metric(entry: dict, objective: Objective,
                       metric: Metric = DEFAULT_LOOKUP_METRIC) -> tuple:
    """Return the sort key for comparing entries by *(objective, metric)*."""
    col = _metric_col(metric)
    m = entry[col]
    d = entry["aig_depth"]
    if objective == "area":
        return (m, d)
    elif objective == "delay":
        return (d, m)
    elif objective == "adp":
        return (m * d, m)
    return (m, d)


def estimate_sep_mul_add_cost(
    a_w: int, b_w: int, c_w: int, signed: bool, objective: Objective,
    metric: Metric = DEFAULT_LOOKUP_METRIC,
) -> dict | None:
    """Estimate cost of separate mul + add from DB entries."""
    mul_entry, _ = lookup_best_config("*", a_w, b_w, signed, objective, metric)
    if mul_entry is None:
        return None
    prod_w = a_w + b_w
    add_entry, _ = lookup_best_config("+", prod_w, c_w, signed, objective, metric)
    if add_entry is None:
        return None
    col = _metric_col(metric)
    return {
        col:        mul_entry[col] + add_entry[col],
        "aig_depth": mul_entry["aig_depth"] + add_entry["aig_depth"],
    }


def pick_best_mac_strategy(
    n_bits: int, c_w: int, signed: bool, objective: Objective,
    metric: Metric = DEFAULT_LOOKUP_METRIC,
) -> tuple[str, dict | None]:
    """Compare fused MAC vs separate mul+add. Returns (strategy, config).

    strategy is "mac" or "sep". config is the fused MAC config dict (if mac),
    or None (if sep — caller uses normal mul+add replacement).
    """
    mac_cfg = lookup_best_mac_config(n_bits, signed, objective, metric)
    sep_cost = estimate_sep_mul_add_cost(n_bits, n_bits, c_w, signed, objective, metric)

    if mac_cfg is None:
        return "sep", None
    if sep_cost is None:
        return "mac", mac_cfg

    mac_metric = _objective_metric(mac_cfg, objective, metric)
    sep_metric = _objective_metric(sep_cost, objective, metric)
    if mac_metric <= sep_metric:
        return "mac", mac_cfg
    return "sep", None


def pick_best_dot_strategy(
    n_terms: int, n_bits: int, signed: bool, objective: Objective,
    metric: Metric = DEFAULT_LOOKUP_METRIC,
) -> tuple[str, dict | None]:
    """Compare fused inner product vs MAC tree vs separate muls+adds.

    Returns (strategy, config) where strategy is "dot", "mac", or "sep".
    """
    dot_cfg = lookup_best_dot_config(n_terms, n_bits, signed, objective, metric)
    mac_cfg = lookup_best_mac_config(n_bits, signed, objective, metric)
    sep_mul, _ = lookup_best_config("*", n_bits, n_bits, signed, objective, metric)
    prod_w = 2 * n_bits
    sep_add, _ = lookup_best_config("+", prod_w, prod_w, signed, objective, metric)

    col = _metric_col(metric)
    candidates = []

    if dot_cfg is not None:
        candidates.append(("dot", dot_cfg, _objective_metric(dot_cfg, objective, metric)))

    # MAC tree: n_terms MACs, depth = mac_depth + log2(n_terms) adder tree levels
    if mac_cfg is not None and sep_add is not None:
        import math
        tree_levels = math.ceil(math.log2(max(n_terms, 2)))
        mac_tree_cost = {
            col:         n_terms * mac_cfg[col] + tree_levels * sep_add[col],
            "aig_depth": mac_cfg["aig_depth"] + tree_levels * sep_add["aig_depth"],
        }
        candidates.append(("mac", mac_cfg, _objective_metric(mac_tree_cost, objective, metric)))

    # Separate: n_terms muls + (n_terms-1) adds
    if sep_mul is not None and sep_add is not None:
        import math
        tree_levels = math.ceil(math.log2(max(n_terms, 2)))
        sep_cost = {
            col:         n_terms * sep_mul[col] + (n_terms - 1) * sep_add[col],
            "aig_depth": sep_mul["aig_depth"] + tree_levels * sep_add["aig_depth"],
        }
        candidates.append(("sep", None, _objective_metric(sep_cost, objective, metric)))

    if not candidates:
        return "sep", None

    best = min(candidates, key=lambda x: x[2])
    return best[0], best[1]
