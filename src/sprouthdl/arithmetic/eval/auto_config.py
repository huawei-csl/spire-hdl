"""Auto-config lookup: load best_configs.csv (or .json) and resolve queries.

Provides ``lookup_best_config`` which returns the empirically best
``ArithmeticConfig`` for a given (op, a_w, b_w, signed, objective) tuple.

For commutative ops (``+``, ``*``), both orientations are checked and
the better one is returned together with a ``swap`` flag.

Supported objectives:
    - ``"area"``:  minimize transistor_count  (tiebreak: aig_depth)
    - ``"delay"``: minimize aig_depth         (tiebreak: transistor_count)
    - ``"adp"``:   minimize area-delay product (transistor_count * aig_depth)
"""

from __future__ import annotations

import csv
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

_DB_DIR = Path(__file__).parent
_INT_FIELDS = {"transistor_count", "aig_depth", "num_aig_gates", "a_w", "b_w"}


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
        best_ab = _select_best(rows_ab, objective)
        best_ba = _select_best(rows_ba, objective)

        if best_ab is None and best_ba is None:
            return None, False
        if best_ba is None:
            return best_ab, False
        if best_ab is None:
            return best_ba, True

        combined = _select_best([best_ab, best_ba], objective)
        swap = combined is best_ba
        return combined, swap
    else:
        return _select_best(rows_ab, objective), False


def lookup_best_mac_config(
    n_bits: int,
    signed: bool,
    objective: Objective = "area",
) -> dict | None:
    """Look up the best fused MAC config for y = a*b + c (symmetric widths)."""
    db = _load_db()
    sign_key = "signed" if signed else "unsigned"
    snapped = _nearest_width_log(n_bits, sorted({
        int(k[1]) for k in db if k[0] == "mac"
    } or {n_bits}))
    rows = db.get(("mac", str(snapped), str(snapped), sign_key), [])
    return _select_best(rows, objective)


def lookup_best_dot_config(
    n_terms: int,
    n_bits: int,
    signed: bool,
    objective: Objective = "area",
) -> dict | None:
    """Look up the best fused inner product config for y = sum(ai*bi)."""
    db = _load_db()
    sign_key = "signed" if signed else "unsigned"
    op_key = f"dot{n_terms}"
    available = sorted({int(k[1]) for k in db if k[0] == op_key} or {n_bits})
    snapped = _nearest_width_log(n_bits, available)
    rows = db.get((op_key, str(snapped), str(snapped), sign_key), [])
    return _select_best(rows, objective)


def _objective_metric(entry: dict, objective: Objective) -> tuple:
    """Return the sort key for comparing entries by objective."""
    tc = entry["transistor_count"]
    d = entry["aig_depth"]
    if objective == "area":
        return (tc, d)
    elif objective == "delay":
        return (d, tc)
    elif objective == "adp":
        return (tc * d, tc)
    return (tc, d)


def estimate_sep_mul_add_cost(
    a_w: int, b_w: int, c_w: int, signed: bool, objective: Objective,
) -> dict | None:
    """Estimate cost of separate mul + add from DB entries."""
    mul_entry, _ = lookup_best_config("*", a_w, b_w, signed, objective)
    if mul_entry is None:
        return None
    prod_w = a_w + b_w
    add_entry, _ = lookup_best_config("+", prod_w, c_w, signed, objective)
    if add_entry is None:
        return None
    return {
        "transistor_count": mul_entry["transistor_count"] + add_entry["transistor_count"],
        "aig_depth": mul_entry["aig_depth"] + add_entry["aig_depth"],
    }


def pick_best_mac_strategy(
    n_bits: int, c_w: int, signed: bool, objective: Objective,
) -> tuple[str, dict | None]:
    """Compare fused MAC vs separate mul+add. Returns (strategy, config).

    strategy is "mac" or "sep". config is the fused MAC config dict (if mac),
    or None (if sep — caller uses normal mul+add replacement).
    """
    mac_cfg = lookup_best_mac_config(n_bits, signed, objective)
    sep_cost = estimate_sep_mul_add_cost(n_bits, n_bits, c_w, signed, objective)

    if mac_cfg is None:
        return "sep", None
    if sep_cost is None:
        return "mac", mac_cfg

    mac_metric = _objective_metric(mac_cfg, objective)
    sep_metric = _objective_metric(sep_cost, objective)
    if mac_metric <= sep_metric:
        return "mac", mac_cfg
    return "sep", None


def pick_best_dot_strategy(
    n_terms: int, n_bits: int, signed: bool, objective: Objective,
) -> tuple[str, dict | None]:
    """Compare fused inner product vs MAC tree vs separate muls+adds.

    Returns (strategy, config) where strategy is "dot", "mac", or "sep".
    """
    dot_cfg = lookup_best_dot_config(n_terms, n_bits, signed, objective)
    mac_cfg = lookup_best_mac_config(n_bits, signed, objective)
    sep_mul, _ = lookup_best_config("*", n_bits, n_bits, signed, objective)
    prod_w = 2 * n_bits
    sep_add, _ = lookup_best_config("+", prod_w, prod_w, signed, objective)

    candidates = []

    if dot_cfg is not None:
        candidates.append(("dot", dot_cfg, _objective_metric(dot_cfg, objective)))

    # MAC tree: n_terms MACs, depth = mac_depth + log2(n_terms) adder tree levels
    if mac_cfg is not None and sep_add is not None:
        import math
        tree_levels = math.ceil(math.log2(max(n_terms, 2)))
        mac_tree_cost = {
            "transistor_count": n_terms * mac_cfg["transistor_count"] + tree_levels * sep_add["transistor_count"],
            "aig_depth": mac_cfg["aig_depth"] + tree_levels * sep_add["aig_depth"],
        }
        candidates.append(("mac", mac_cfg, _objective_metric(mac_tree_cost, objective)))

    # Separate: n_terms muls + (n_terms-1) adds
    if sep_mul is not None and sep_add is not None:
        import math
        tree_levels = math.ceil(math.log2(max(n_terms, 2)))
        sep_cost = {
            "transistor_count": n_terms * sep_mul["transistor_count"] + (n_terms - 1) * sep_add["transistor_count"],
            "aig_depth": sep_mul["aig_depth"] + tree_levels * sep_add["aig_depth"],
        }
        candidates.append(("sep", None, _objective_metric(sep_cost, objective)))

    if not candidates:
        return "sep", None

    best = min(candidates, key=lambda x: x[2])
    return best[0], best[1]


def lookup_best_arithmetic_config(
    op: Literal["+", "-", "*"],
    a_w: int,
    b_w: int,
    signed: bool,
    objective: Objective = "area",
    full_output_bit: bool = True,
):
    """Return ``(ArithmeticConfig, swap)`` for the empirically best configuration."""
    from sprouthdl.arithmetic.int_arithmetic_config import ArithmeticConfig
    from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding

    entry, swap = lookup_best_config(op, a_w, b_w, signed, objective)
    if entry is None:
        encoding = Encoding.twos_complement if signed else Encoding.unsigned
        return ArithmeticConfig(encoding=encoding, full_output_bit=full_output_bit), False

    encoding = Encoding.twos_complement if signed else Encoding.unsigned
    optim_type = entry.get("optim_type", "area") or "area"

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
