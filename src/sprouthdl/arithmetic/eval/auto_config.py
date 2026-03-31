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
    # MAC entries are keyed as op="mac", with a_w=b_w=n_bits
    snapped = _nearest_width_log(n_bits, sorted({
        int(k[1]) for k in db if k[0] == "mac"
    } or {n_bits}))
    rows = db.get(("mac", str(snapped), str(snapped), sign_key), [])
    return _select_best(rows, objective)


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
