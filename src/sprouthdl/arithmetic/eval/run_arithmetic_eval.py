"""Evaluate arithmetic configurations and build a config database.

Sweeps adder FSA options and multiplier PPG x PPA x FSA combos across bitwidths,
collects yosys transistor count and AIG depth, and stores all results so the
best config can be selected at lookup time for any objective (area, delay, ADP).

Usage:
    python -m sprouthdl.arithmetic.eval.run_arithmetic_eval
"""

from __future__ import annotations

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
from sprouthdl.arithmetic.prefix_adders.adders import StageBasedPrefixAdder
from sprouthdl.helpers import get_aig_stats, get_yosys_metrics
from sprouthdl.sprouthdl import reset_shared_cache

_OUT_PATH = Path(__file__).parent / "best_configs.json"

# FSA options to skip in sweeps
_FSA_SKIP = {FSAOption.NONE, FSAOption.PLUS_OPERATOR}
_PPG_SKIP = {PPGOption.NONE}
_PPA_SKIP = {PPAOption.NONE}


# ---------------------------------------------------------------------------
# Single-config evaluators (run in worker processes)
# ---------------------------------------------------------------------------

def eval_adder(
    fsa_opt: FSAOption,
    n_bits: int,
    signed: bool,
) -> dict:
    """Evaluate a single adder configuration. Returns metrics dict.

    Note: optim_type is not swept for adders because it only affects PPA
    stages (compressor tree full-adder selection), which standalone adders
    do not use.  The FSA topology fully determines the adder circuit.
    """
    reset_shared_cache()

    adder = StageBasedPrefixAdder(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
        fsa_cls=fsa_opt.value,
        full_output_bit=True,
    )
    module = adder.to_module(
        f"adder_{fsa_opt.name}_{n_bits}b_{'s' if signed else 'u'}"
    )

    ym = get_yosys_metrics(module)
    aig = get_aig_stats(module)

    return {
        "fsa_opt": fsa_opt.name,
        "transistor_count": int(ym["estimated_num_transistors"]),
        "aig_depth": int(aig["depth"]),
        "num_aig_gates": int(aig["num_gates"]),
    }


def eval_multiplier(
    ppg_opt: PPGOption,
    ppa_opt: PPAOption,
    fsa_opt: FSAOption,
    n_bits: int,
    encoding: Encoding,
    optim_type: Literal["area", "speed"],
) -> dict:
    """Evaluate a single multiplier configuration. Returns metrics dict."""
    reset_shared_cache()

    multiplier = MultiplierOption.STAGE_BASED_MULTIPLIER.value(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=encoding,
        b_encoding=encoding,
        ppg_cls=ppg_opt.value,
        ppa_cls=ppa_opt.value,
        fsa_cls=fsa_opt.value,
        optim_type=optim_type,
    )
    module = multiplier.to_module(
        f"mul_{ppg_opt.name}_{ppa_opt.name}_{fsa_opt.name}_{n_bits}b_{encoding.name}_{optim_type}"
    )

    ym = get_yosys_metrics(module)
    aig = get_aig_stats(module)

    return {
        "ppg_opt": ppg_opt.name,
        "ppa_opt": ppa_opt.name,
        "fsa_opt": fsa_opt.name,
        "optim_type": optim_type,
        "transistor_count": int(ym["estimated_num_transistors"]),
        "aig_depth": int(aig["depth"]),
        "num_aig_gates": int(aig["num_gates"]),
    }


# ---------------------------------------------------------------------------
# Sweep functions
# ---------------------------------------------------------------------------

def sweep_adders(
    bitwidths: list[int],
    max_workers: int = 16,
) -> list[dict]:
    """Sweep all adder FSA options across bitwidths and signed/unsigned."""
    fsa_options = [f for f in FSAOption if f not in _FSA_SKIP]

    tasks = []
    for n_bits, fsa, signed in product(bitwidths, fsa_options, [False, True]):
        tasks.append((fsa, n_bits, signed))

    print(f"Adder sweep: {len(tasks)} configurations")
    results: dict[tuple, list] = {}  # (n_bits, signed) -> [row, ...]
    errors = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(eval_adder, *args): args for args in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Adders"):
            args = futures[fut]
            try:
                row = fut.result()
                key = (args[1], args[2])  # (n_bits, signed)
                results.setdefault(key, []).append(row)
            except Exception as e:
                errors.append(f"{args}: {e}")
                print(f"Error: {e}")

    if errors:
        print(f"Adder sweep: {len(errors)} errors out of {len(tasks)}")
    return results


def sweep_multipliers(
    bitwidths: list[int],
    max_workers: int = 16,
) -> dict[tuple, list]:
    """Sweep multiplier PPG x PPA x FSA across bitwidths and encodings."""
    ppg_options = [p for p in PPGOption if p not in _PPG_SKIP]
    ppa_options = [p for p in PPAOption if p not in _PPA_SKIP]
    fsa_options = [f for f in FSAOption if f not in _FSA_SKIP]

    tasks = []
    for n_bits, ppg, ppa, fsa, optim_type in product(
        bitwidths, ppg_options, ppa_options, fsa_options, ["area", "speed"]
    ):
        for encoding in [Encoding.unsigned, Encoding.twos_complement]:
            sig = (is_signed(encoding), is_signed(encoding))
            if sig not in ppg.value.supported_signatures:
                continue
            tasks.append((ppg, ppa, fsa, n_bits, encoding, optim_type))

    print(f"Multiplier sweep: {len(tasks)} configurations")
    results: dict[tuple, list] = {}  # (n_bits, signed) -> [row, ...]
    errors = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(eval_multiplier, *args): args for args in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Multipliers"):
            args = futures[fut]
            try:
                row = fut.result()
                key = (args[3], is_signed(args[4]))  # (n_bits, signed)
                results.setdefault(key, []).append(row)
            except Exception as e:
                errors.append(f"{args}: {e}")
                print(f"Error: {e}")

    if errors:
        print(f"Multiplier sweep: {len(errors)} errors out of {len(tasks)}")
    return results


# ---------------------------------------------------------------------------
# Database builder
# ---------------------------------------------------------------------------

def build_config_db(
    adder_groups: dict[tuple, list],
    multiplier_groups: dict[tuple, list],
) -> dict:
    """Build the nested config database storing all evaluated rows."""
    db = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "version": 2,
        },
        "configs": {"add": {}, "sub": {}, "mul": {}},
    }

    for (n_bits, signed), rows in adder_groups.items():
        w_key = str(n_bits)
        s_key = "signed" if signed else "unsigned"
        for op_key in ("add", "sub"):
            db["configs"][op_key].setdefault(w_key, {})[s_key] = rows

    for (n_bits, signed), rows in multiplier_groups.items():
        w_key = str(n_bits)
        s_key = "signed" if signed else "unsigned"
        db["configs"]["mul"].setdefault(w_key, {})[s_key] = rows

    all_widths = sorted(
        {k[0] for k in adder_groups} | {k[0] for k in multiplier_groups}
    )
    db["metadata"]["bitwidths"] = all_widths

    return db


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    bitwidths: list[int] | None = None,
    max_workers: int = 16,
    output_path: Path | None = None,
) -> None:
    if bitwidths is None:
        bitwidths = [4, 8, 16]
    if output_path is None:
        output_path = _OUT_PATH

    sys.setrecursionlimit(10000)

    print(f"Running arithmetic evaluation sweep for bitwidths={bitwidths}")
    t0 = time.time()

    adder_groups = sweep_adders(bitwidths, max_workers=max_workers)
    multiplier_groups = sweep_multipliers(bitwidths, max_workers=max_workers)

    n_adder = sum(len(v) for v in adder_groups.values())
    n_mul = sum(len(v) for v in multiplier_groups.values())

    db = build_config_db(adder_groups, multiplier_groups)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(db, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Saved config database to {output_path}")
    print(f"  Adder rows:      {n_adder}")
    print(f"  Multiplier rows: {n_mul}")

    # Print summary using auto_config lookup
    from sprouthdl.arithmetic.eval.auto_config import OBJECTIVES, _select_best, _load_db
    _load_db.cache_clear()  # force reload
    db = _load_db()
    for op_key in ("add", "mul"):
        print(f"\n  Best configs for '{op_key}':")
        for w_key in sorted(db["configs"][op_key], key=int):
            for s_key in ("unsigned", "signed"):
                rows = db["configs"][op_key].get(w_key, {}).get(s_key, [])
                if not rows:
                    continue
                for obj_name in OBJECTIVES:
                    best = _select_best(rows, obj_name)
                    if best:
                        label = f"{w_key}b {s_key:>8s} {obj_name:>15s}"
                        tc = best["transistor_count"]
                        d = best["aig_depth"]
                        fsa = best["fsa_opt"]
                        extra = ""
                        if "ppg_opt" in best:
                            extra = f" ppg={best['ppg_opt']} ppa={best['ppa_opt']}"
                        print(f"    {label}: tc={tc:>6d}  depth={d:>3d}  fsa={fsa}{extra}")


if __name__ == "__main__":
    main()
