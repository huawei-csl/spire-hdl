"""Evaluate arithmetic configurations and build a config database.

Sweeps adder/subtractor FSA options and multiplier PPG x PPA x FSA combos
across width pairs (including asymmetric and both orientations), collects
yosys transistor count and AIG depth, and stores results as CSV or JSON.

Usage:
    python -m sprouthdl.arithmetic.eval.run_arithmetic_eval
    python -m sprouthdl.arithmetic.eval.run_arithmetic_eval --format json --no-pareto
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    get_list_from_enum,
)
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from sprouthdl.arithmetic.prefix_adders.adders import StageBasedPrefixAdder, StageBasedSubtractor
from sprouthdl.helpers import get_aig_stats, get_yosys_metrics
from sprouthdl.sprouthdl import reset_shared_cache

_OUT_DIR = Path(__file__).parent

_FSA_SKIP = {FSAOption.NONE, FSAOption.PLUS_OPERATOR}
_PPG_SKIP = {PPGOption.NONE}
_PPA_SKIP = {PPAOption.NONE}

_CSV_COLUMNS = [
    "op", "a_w", "b_w", "signed",
    "fsa_opt", "ppg_opt", "ppa_opt", "optim_type",
    "transistor_count", "aig_depth", "num_aig_gates",
]


def _width_pairs(bitwidths: list[int]) -> list[tuple[int, int]]:
    """Generate all ordered (a_w, b_w) pairs including asymmetric."""
    pairs = set()
    for a_w in bitwidths:
        for b_w in bitwidths:
            if b_w <= a_w:
                pairs.add((a_w, b_w))
                if a_w != b_w:
                    pairs.add((b_w, a_w))
    return sorted(pairs)


def _pareto_filter(rows: list[dict]) -> list[dict]:
    """Keep only Pareto-optimal rows (not dominated on both TC and depth)."""
    result = []
    for r in rows:
        dominated = any(
            o["transistor_count"] <= r["transistor_count"]
            and o["aig_depth"] <= r["aig_depth"]
            and (o["transistor_count"] < r["transistor_count"] or o["aig_depth"] < r["aig_depth"])
            for o in rows
        )
        if not dominated:
            result.append(r)
    return result


# ---------------------------------------------------------------------------
# Single-config evaluators (run in worker processes)
# ---------------------------------------------------------------------------

def eval_adder(fsa_opt: FSAOption, a_w: int, b_w: int, signed: bool) -> dict:
    reset_shared_cache()
    adder = StageBasedPrefixAdder(
        a_w=a_w, b_w=b_w, signed_a=signed, signed_b=signed,
        optim_type="area", fsa_cls=fsa_opt.value, full_output_bit=True,
    )
    module = adder.to_module(f"adder_{fsa_opt.name}_{a_w}x{b_w}")
    ym = get_yosys_metrics(module)
    aig = get_aig_stats(module)
    return {
        "op": "add", "a_w": a_w, "b_w": b_w, "signed": "signed" if signed else "unsigned",
        "fsa_opt": fsa_opt.name, "ppg_opt": "", "ppa_opt": "", "optim_type": "",
        "transistor_count": int(ym["estimated_num_transistors"]),
        "aig_depth": int(aig["depth"]), "num_aig_gates": int(aig["num_gates"]),
    }


def eval_subtractor(fsa_opt: FSAOption, a_w: int, b_w: int, signed: bool) -> dict:
    reset_shared_cache()
    sub = StageBasedSubtractor(
        a_w=a_w, b_w=b_w, signed_a=signed, signed_b=signed,
        optim_type="area", fsa_cls=fsa_opt.value, full_output_bit=True,
    )
    module = sub.to_module(f"sub_{fsa_opt.name}_{a_w}x{b_w}")
    ym = get_yosys_metrics(module)
    aig = get_aig_stats(module)
    return {
        "op": "sub", "a_w": a_w, "b_w": b_w, "signed": "signed" if signed else "unsigned",
        "fsa_opt": fsa_opt.name, "ppg_opt": "", "ppa_opt": "", "optim_type": "",
        "transistor_count": int(ym["estimated_num_transistors"]),
        "aig_depth": int(aig["depth"]), "num_aig_gates": int(aig["num_gates"]),
    }


def eval_multiplier(
    ppg_opt: PPGOption, ppa_opt: PPAOption, fsa_opt: FSAOption,
    a_w: int, b_w: int, encoding: Encoding, optim_type: Literal["area", "speed"],
) -> dict:
    reset_shared_cache()
    multiplier = MultiplierOption.STAGE_BASED_MULTIPLIER.value(
        a_w=a_w, b_w=b_w, a_encoding=encoding, b_encoding=encoding,
        ppg_cls=ppg_opt.value, ppa_cls=ppa_opt.value,
        fsa_cls=fsa_opt.value, optim_type=optim_type,
    )
    module = multiplier.to_module(
        f"mul_{ppg_opt.name}_{ppa_opt.name}_{fsa_opt.name}_{a_w}x{b_w}_{encoding.name}_{optim_type}"
    )
    ym = get_yosys_metrics(module)
    aig = get_aig_stats(module)
    return {
        "op": "mul", "a_w": a_w, "b_w": b_w,
        "signed": "signed" if is_signed(encoding) else "unsigned",
        "fsa_opt": fsa_opt.name, "ppg_opt": ppg_opt.name, "ppa_opt": ppa_opt.name,
        "optim_type": optim_type,
        "transistor_count": int(ym["estimated_num_transistors"]),
        "aig_depth": int(aig["depth"]), "num_aig_gates": int(aig["num_gates"]),
    }


# ---------------------------------------------------------------------------
# Sweep functions
# ---------------------------------------------------------------------------

def _run_parallel(desc: str, tasks: list, eval_fn, max_workers: int) -> list[dict]:
    """Run tasks in parallel, return collected rows."""
    print(f"{desc}: {len(tasks)} configurations")
    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(eval_fn, *args): args for args in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append(str(e))
    if errors:
        print(f"  {len(errors)} errors out of {len(tasks)}")
    return results


def sweep_adders(bitwidths: list[int], max_workers: int = 16) -> list[dict]:
    fsa_options = [f for f in FSAOption if f not in _FSA_SKIP]
    pairs = _width_pairs(bitwidths)
    tasks = [(fsa, a_w, b_w, signed)
             for (a_w, b_w), fsa, signed in product(pairs, fsa_options, [False, True])]
    return _run_parallel("Adders", tasks, eval_adder, max_workers)


def sweep_subtractors(bitwidths: list[int], max_workers: int = 16) -> list[dict]:
    fsa_options = [f for f in FSAOption if f not in _FSA_SKIP]
    pairs = _width_pairs(bitwidths)
    tasks = [(fsa, a_w, b_w, signed)
             for (a_w, b_w), fsa, signed in product(pairs, fsa_options, [False, True])]
    return _run_parallel("Subtractors", tasks, eval_subtractor, max_workers)


def sweep_multipliers(bitwidths: list[int], max_workers: int = 16) -> list[dict]:
    ppg_options = [p for p in PPGOption if p not in _PPG_SKIP]
    ppa_options = [p for p in PPAOption if p not in _PPA_SKIP]
    fsa_options = [f for f in FSAOption if f not in _FSA_SKIP]
    pairs = _width_pairs(bitwidths)

    tasks = []
    for (a_w, b_w), ppg, ppa, fsa, optim_type in product(
        pairs, ppg_options, ppa_options, fsa_options, ["area", "speed"]
    ):
        if ppg == PPGOption.BOOTH_UNOPTIMISED and a_w <= 2:
            continue
        for encoding in [Encoding.unsigned, Encoding.twos_complement]:
            sig = (is_signed(encoding), is_signed(encoding))
            if sig not in ppg.value.supported_signatures:
                continue
            tasks.append((ppg, ppa, fsa, a_w, b_w, encoding, optim_type))

    return _run_parallel("Multipliers", tasks, eval_multiplier, max_workers)


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], path: Path, pareto: bool = True) -> None:
    """Save rows as flat CSV, optionally Pareto-filtered per group."""
    if pareto:
        grouped: dict[tuple, list] = {}
        for r in rows:
            key = (r["op"], r["a_w"], r["b_w"], r["signed"])
            grouped.setdefault(key, []).append(r)
        filtered = []
        for group_rows in grouped.values():
            filtered.extend(_pareto_filter(group_rows))
        rows = filtered

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_json(rows: list[dict], path: Path, pareto: bool = True) -> None:
    """Save rows as nested JSON (v3 format), optionally Pareto-filtered."""
    db: dict = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "version": 3,
        },
        "configs": {},
    }

    grouped: dict[tuple, list] = {}
    for r in rows:
        key = (r["op"], str(r["a_w"]), str(r["b_w"]), r["signed"])
        grouped.setdefault(key, []).append(r)

    for (op, a_w_s, b_w_s, sign_key), group_rows in grouped.items():
        w_key = f"{a_w_s}x{b_w_s}"
        filtered = _pareto_filter(group_rows) if pareto else group_rows
        # Strip op/a_w/b_w/signed from individual rows (redundant with key)
        clean = [{k: v for k, v in r.items() if k not in ("op", "a_w", "b_w", "signed")} for r in filtered]
        db["configs"].setdefault(op, {}).setdefault(w_key, {})[sign_key] = clean

    all_pairs = sorted({(r["a_w"], r["b_w"]) for r in rows})
    db["metadata"]["width_pairs"] = all_pairs

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(db, f, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    bitwidths: list[int] | None = None,
    max_workers: int = 16,
    output_dir: Path | None = None,
    fmt: str = "csv",
    pareto: bool = True,
) -> None:
    if bitwidths is None:
        bitwidths = [1, 2, 4, 8, 16]
    if output_dir is None:
        output_dir = _OUT_DIR

    sys.setrecursionlimit(10000)
    print(f"Sweep: bitwidths={bitwidths}, format={fmt}, pareto={pareto}")
    t0 = time.time()

    all_rows = []
    all_rows.extend(sweep_adders(bitwidths, max_workers=max_workers))
    all_rows.extend(sweep_subtractors(bitwidths, max_workers=max_workers))
    all_rows.extend(sweep_multipliers(bitwidths, max_workers=max_workers))

    if fmt == "csv":
        out_path = output_dir / "best_configs.csv"
        save_csv(all_rows, out_path, pareto=pareto)
    elif fmt == "json":
        out_path = output_dir / "best_configs.json"
        save_json(all_rows, out_path, pareto=pareto)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Saved to {out_path}")
    print(f"  Total rows collected: {len(all_rows)}")
    if pareto:
        grouped: dict[tuple, list] = {}
        for r in all_rows:
            key = (r["op"], r["a_w"], r["b_w"], r["signed"])
            grouped.setdefault(key, []).append(r)
        n_pareto = sum(len(_pareto_filter(g)) for g in grouped.values())
        print(f"  After Pareto filter: {n_pareto}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run arithmetic evaluation sweep")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--no-pareto", action="store_true", help="Store all rows, not just Pareto front")
    parser.add_argument("--bitwidths", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    main(
        bitwidths=args.bitwidths,
        fmt=args.format,
        pareto=not args.no_pareto,
        max_workers=args.workers,
    )
