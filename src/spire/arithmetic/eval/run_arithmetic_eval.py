"""Evaluate arithmetic configurations and build a config database.

Sweeps adder/subtractor FSA options and multiplier PPG x PPA x FSA combos
across width pairs (including asymmetric and both orientations), collects
yosys transistor count and AIG depth, and stores results as CSV or JSON.

Usage:
    python -m spire.arithmetic.eval.run_arithmetic_eval
    python -m spire.arithmetic.eval.run_arithmetic_eval --format json --no-pareto
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
from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import OptimType

from tqdm import tqdm

from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    get_list_from_enum,
)
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from spire.arithmetic.prefix_adders.adders import StageBasedPrefixAdder, StageBasedSubtractor
from spire.arithmetic.int_mac_fused import FusedMacComponent, MacBuildConfig
from spire.cores.matmul_accumulate.matmul_accumulate_core_fused import (
    MultiplierConfig as FusedMultiplierConfig,
    fused_inner_product,
)
from spire.helpers import (
    extract_yosys_metrics_from_verilog,
    extract_yosys_heavy_metrics_from_verilog,
    get_aig_stats,
)
from spire.expr import reset_shared_cache

_OUT_DIR = Path(__file__).parent

# Always-skipped final-stage adders (placeholders and the operator adapter).
# MULTI_SCAN / ZCG have hard-coded prefix graphs only for n in {8,16,24,32} /
# {24,32}; for unsupported widths they now fall back to the next lower
# template and legalize_P extends it, so individual sub-minimum sizes fail
# silently through _run_parallel's error path rather than via this list.
_FSA_SKIP = {
    FSAOption.NONE,
    FSAOption.PLUS_OPERATOR,
}
# Skip Booth-unoptimised: was not competitive in the multiplier pareto front.
_PPG_SKIP = {
    PPGOption.NONE,
    PPGOption.BOOTH_UNOPTIMISED,
}
# Skip the less performant options
_PPA_SKIP = {
    PPAOption.NONE,
    PPAOption.FIVE_TWO_COMPRESSOR,
    PPAOption.FIVE_TWO_COMPRESSOR_PARALLEL,
    PPAOption.FOUR_TWO_COMPRESSOR_PARALLEL,
    PPAOption.EAGER_WALLACE_TREE,
    PPAOption.BDT_WALLACE_TREE,
}

_CSV_COLUMNS = [
    "op", "a_w", "b_w", "signed",
    "fsa_opt", "ppg_opt", "ppa_opt", "optim_type",
    "transistor_count", "transistor_count_heavy", "aig_depth", "num_aig_gates",
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


_PARETO_METRIC_COLS = ("transistor_count", "transistor_count_heavy", "num_aig_gates")


def _pareto_filter(rows: list[dict]) -> list[dict]:
    """Keep rows that are Pareto-optimal under at least one (metric, aig_depth)
    pair, where metric ∈ {transistor_count, transistor_count_heavy,
    num_aig_gates}.

    A row is kept iff, for some metric column, no other row dominates it on
    (metric, aig_depth).
    """
    result = []
    for r in rows:
        kept_under_any = False
        for col in _PARETO_METRIC_COLS:
            r_val = r.get(col)
            if r_val in (None, ""):
                continue
            r_depth = r["aig_depth"]
            dominated = False
            for o in rows:
                if o is r:
                    continue
                o_val = o.get(col)
                if o_val in (None, ""):
                    continue
                o_depth = o["aig_depth"]
                if (o_val <= r_val and o_depth <= r_depth and
                        (o_val < r_val or o_depth < r_depth)):
                    dominated = True
                    break
            if not dominated:
                kept_under_any = True
                break
        if kept_under_any:
            result.append(r)
    return result


def _yosys_metrics(module) -> dict:
    """Run BOTH the lite (`abc -fast`) and heavy (`synth; clean -purge`) yosys
    pipelines on the same emitted Verilog and return both transistor counts.
    """
    verilog_lines = module.to_verilog().splitlines()
    metrics = extract_yosys_metrics_from_verilog(verilog_lines)
    try:
        heavy = extract_yosys_heavy_metrics_from_verilog(verilog_lines)
        metrics["estimated_num_transistors_heavy"] = heavy["estimated_num_transistors"]
    except Exception:
        # Don't fail the whole eval if the heavy pass errors on a degenerate
        # case (e.g. 1-bit ops with empty hierarchy); just leave the column blank.
        metrics["estimated_num_transistors_heavy"] = None
    return metrics


# ---------------------------------------------------------------------------
# Single-config evaluators (run in worker processes)
# ---------------------------------------------------------------------------

def eval_adder(fsa_opt: FSAOption, a_w: int, b_w: int, signed: bool,
                optim_type: OptimType) -> dict:
    reset_shared_cache()
    adder = StageBasedPrefixAdder(
        a_w=a_w, b_w=b_w, signed_a=signed, signed_b=signed,
        optim_type=optim_type, fsa_cls=fsa_opt.value, full_output_bit=True,
    )
    module = adder.to_module(f"adder_{fsa_opt.name}_{a_w}x{b_w}_{optim_type}")
    ym = _yosys_metrics(module)
    aig = get_aig_stats(module)
    return {
        "op": "add", "a_w": a_w, "b_w": b_w, "signed": "signed" if signed else "unsigned",
        "fsa_opt": fsa_opt.name, "ppg_opt": "", "ppa_opt": "", "optim_type": optim_type,
        "transistor_count": int(ym["estimated_num_transistors"]),
        "transistor_count_heavy": int(ym["estimated_num_transistors_heavy"]) if ym.get("estimated_num_transistors_heavy") is not None else "",
        "aig_depth": int(aig["depth"]), "num_aig_gates": int(aig["num_gates"]),
    }


def eval_subtractor(fsa_opt: FSAOption, a_w: int, b_w: int, signed: bool,
                     optim_type: OptimType) -> dict:
    reset_shared_cache()
    sub = StageBasedSubtractor(
        a_w=a_w, b_w=b_w, signed_a=signed, signed_b=signed,
        optim_type=optim_type, fsa_cls=fsa_opt.value, full_output_bit=True,
    )
    module = sub.to_module(f"sub_{fsa_opt.name}_{a_w}x{b_w}_{optim_type}")
    ym = _yosys_metrics(module)
    aig = get_aig_stats(module)
    return {
        "op": "sub", "a_w": a_w, "b_w": b_w, "signed": "signed" if signed else "unsigned",
        "fsa_opt": fsa_opt.name, "ppg_opt": "", "ppa_opt": "", "optim_type": optim_type,
        "transistor_count": int(ym["estimated_num_transistors"]),
        "transistor_count_heavy": int(ym["estimated_num_transistors_heavy"]) if ym.get("estimated_num_transistors_heavy") is not None else "",
        "aig_depth": int(aig["depth"]), "num_aig_gates": int(aig["num_gates"]),
    }


def eval_multiplier(
    ppg_opt: PPGOption, ppa_opt: PPAOption, fsa_opt: FSAOption,
    a_w: int, b_w: int, encoding: Encoding, optim_type: OptimType,
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
    ym = _yosys_metrics(module)
    aig = get_aig_stats(module)
    return {
        "op": "mul", "a_w": a_w, "b_w": b_w,
        "signed": "signed" if is_signed(encoding) else "unsigned",
        "fsa_opt": fsa_opt.name, "ppg_opt": ppg_opt.name, "ppa_opt": ppa_opt.name,
        "optim_type": optim_type,
        "transistor_count": int(ym["estimated_num_transistors"]),
        "transistor_count_heavy": int(ym["estimated_num_transistors_heavy"]) if ym.get("estimated_num_transistors_heavy") is not None else "",
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
    tasks = [(fsa, a_w, b_w, signed, optim_type)
             for (a_w, b_w), fsa, signed, optim_type in product(
                 pairs, fsa_options, [False, True], ["area", "speed"])]
    return _run_parallel("Adders", tasks, eval_adder, max_workers)


def sweep_subtractors(bitwidths: list[int], max_workers: int = 16) -> list[dict]:
    fsa_options = [f for f in FSAOption if f not in _FSA_SKIP]
    pairs = _width_pairs(bitwidths)
    tasks = [(fsa, a_w, b_w, signed, optim_type)
             for (a_w, b_w), fsa, signed, optim_type in product(
                 pairs, fsa_options, [False, True], ["area", "speed"])]
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


def eval_mac(
    ppg_opt: PPGOption, ppa_opt: PPAOption, fsa_opt: FSAOption,
    n_bits: int, c_bits: int, encoding: Encoding,
    optim_type: OptimType,
) -> dict:
    """Evaluate a single fused MAC configuration (y = a*b + c)."""
    reset_shared_cache()
    fused = FusedMacComponent(MacBuildConfig(
        n_bits=n_bits, c_bits=c_bits,
        ppg_opt=ppg_opt, ppa_opt=ppa_opt, fsa_opt=fsa_opt,
        encoding=encoding, optim_type=optim_type, use_operator=False,
    ))
    module = fused.to_module(
        f"mac_{ppg_opt.name}_{ppa_opt.name}_{fsa_opt.name}_{n_bits}b_c{c_bits}_{encoding.name}_{optim_type}"
    )
    ym = _yosys_metrics(module)
    aig = get_aig_stats(module)
    return {
        "op": "mac", "a_w": n_bits, "b_w": n_bits, "signed": "signed" if is_signed(encoding) else "unsigned",
        "fsa_opt": fsa_opt.name, "ppg_opt": ppg_opt.name, "ppa_opt": ppa_opt.name,
        "optim_type": optim_type,
        "transistor_count": int(ym["estimated_num_transistors"]),
        "transistor_count_heavy": int(ym["estimated_num_transistors_heavy"]) if ym.get("estimated_num_transistors_heavy") is not None else "",
        "aig_depth": int(aig["depth"]), "num_aig_gates": int(aig["num_gates"]),
    }


def sweep_macs(bitwidths: list[int], max_workers: int = 16) -> list[dict]:
    """Sweep fused MAC configs. Uses symmetric widths with c_bits = 2*n_bits."""
    ppg_options = [p for p in PPGOption if p not in _PPG_SKIP]
    ppa_options = [p for p in PPAOption if p not in _PPA_SKIP]
    fsa_options = [f for f in FSAOption if f not in _FSA_SKIP]

    # MAC sweep uses symmetric widths only (a_w == b_w), c_bits = 2*n_bits
    tasks = []
    for n_bits, ppg, ppa, fsa, optim_type in product(
        bitwidths, ppg_options, ppa_options, fsa_options, ["area", "speed"]
    ):
        if n_bits < 2:  # MAC with 1-bit inputs is trivial
            continue
        if ppg == PPGOption.BOOTH_UNOPTIMISED and n_bits <= 2:
            continue
        c_bits = 2 * n_bits
        for encoding in [Encoding.unsigned, Encoding.twos_complement]:
            sig = (is_signed(encoding), is_signed(encoding))
            if sig not in ppg.value.supported_signatures:
                continue
            tasks.append((ppg, ppa, fsa, n_bits, c_bits, encoding, optim_type))

    return _run_parallel("MACs", tasks, eval_mac, max_workers)


def eval_mia(
    ppa_opt: PPAOption, fsa_opt: FSAOption,
    n_inputs: int, n_bits: int, encoding: Encoding,
    optim_type: OptimType,
) -> dict:
    """Evaluate a multi-input add (MIA): y = sum(operands), N≥3 symmetric width.

    Builds a single CSA tree (`ppa_cls`) + final 2-input adder (`fsa_cls`) via
    the shared `compressor_sum` helper. Captures both yosys-pipeline transistor
    counts and AIG stats. Picks `op="mia<N>"` to keep the DB schema flat —
    `lookup_best_mia_config(N, width, signed, …)` snaps to the nearest `N`.
    """
    reset_shared_cache()
    from spire.component import Netlist
    from spire.expr import UInt, SInt
    from spire.arithmetic.int_multipliers.stages.ppa_fsa_util import (
        OutputConfig, compressor_sum,
    )

    signed = is_signed(encoding)
    io_type = SInt if signed else UInt

    # Output width: max input width + ceil(log2(N)) extra bits for carries.
    import math
    extra = max(1, (n_inputs - 1).bit_length())
    out_w = n_bits + extra

    m = Netlist(
        f"mia{n_inputs}_{ppa_opt.name}_{fsa_opt.name}_{n_bits}b_{encoding.name}_{optim_type}",
        with_clock=False, with_reset=False,
    )
    operands = [m.input(io_type(n_bits), f"x{i}") for i in range(n_inputs)]
    y = m.output(io_type(out_w), "y")

    config = OutputConfig(out_width=out_w, optim_type=optim_type)
    sum_expr = compressor_sum(config, list(operands), ppa_opt.value, fsa_opt.value)
    if sum_expr.typ.width > out_w:
        sum_expr = sum_expr[0:out_w]
    y <<= sum_expr

    ym = _yosys_metrics(m)
    aig = get_aig_stats(m)
    return {
        "op": f"mia{n_inputs}", "a_w": n_bits, "b_w": n_bits,
        "signed": "signed" if signed else "unsigned",
        "fsa_opt": fsa_opt.name, "ppg_opt": "", "ppa_opt": ppa_opt.name,
        "optim_type": optim_type,
        "transistor_count": int(ym["estimated_num_transistors"]),
        "transistor_count_heavy": int(ym["estimated_num_transistors_heavy"]) if ym.get("estimated_num_transistors_heavy") is not None else "",
        "aig_depth": int(aig["depth"]), "num_aig_gates": int(aig["num_gates"]),
    }


def sweep_mia(bitwidths: list[int], n_inputs_list: list[int],
              max_workers: int = 16) -> list[dict]:
    """Sweep multi-input add configs. Uses symmetric widths only."""
    ppa_options = [p for p in PPAOption if p not in _PPA_SKIP]
    fsa_options = [f for f in FSAOption if f not in _FSA_SKIP]

    tasks = []
    for n_bits, n_inputs, ppa, fsa, optim_type in product(
        bitwidths, n_inputs_list, ppa_options, fsa_options, ["area", "speed"]
    ):
        if n_bits < 2:  # Trivial / degenerate
            continue
        if n_inputs < 3:
            continue
        for encoding in [Encoding.unsigned, Encoding.twos_complement]:
            tasks.append((ppa, fsa, n_inputs, n_bits, encoding, optim_type))

    return _run_parallel("MIA", tasks, eval_mia, max_workers)


def eval_inner_product(
    ppg_opt: PPGOption, ppa_opt: PPAOption, fsa_opt: FSAOption,
    n_terms: int, n_bits: int, encoding: Encoding,
    optim_type: OptimType,
) -> dict:
    """Evaluate a fused inner product: y = a0*b0 + a1*b1 + ... (n_terms pairs, no c)."""
    reset_shared_cache()
    from spire.expr import Const, UInt as _UInt, SInt as _SInt

    signed = is_signed(encoding)
    io_type = _SInt if signed else _UInt

    fused_cfg = FusedMultiplierConfig(
        ppg_opt=ppg_opt, ppa_opt=ppa_opt, fsa_opt=fsa_opt,
        optim_type=optim_type,
    )

    from spire.component import Netlist
    m = Netlist(f"dot{n_terms}_{ppg_opt.name}_{ppa_opt.name}_{fsa_opt.name}_{n_bits}b_{encoding.name}_{optim_type}",
               with_clock=False, with_reset=False)
    vec_a = [m.input(io_type(n_bits), f"a{i}") for i in range(n_terms)]
    vec_b = [m.input(io_type(n_bits), f"b{i}") for i in range(n_terms)]
    c_term = Const(0, _UInt(1))
    result = fused_inner_product(vec_a, vec_b, c_term, fused_cfg, encoding)
    y = m.output(io_type(result.typ.width), "y")
    y <<= result

    ym = _yosys_metrics(m)
    aig = get_aig_stats(m)
    return {
        "op": f"dot{n_terms}", "a_w": n_bits, "b_w": n_bits,
        "signed": "signed" if signed else "unsigned",
        "fsa_opt": fsa_opt.name, "ppg_opt": ppg_opt.name, "ppa_opt": ppa_opt.name,
        "optim_type": optim_type,
        "transistor_count": int(ym["estimated_num_transistors"]),
        "transistor_count_heavy": int(ym["estimated_num_transistors_heavy"]) if ym.get("estimated_num_transistors_heavy") is not None else "",
        "aig_depth": int(aig["depth"]), "num_aig_gates": int(aig["num_gates"]),
    }


def sweep_inner_products(bitwidths: list[int], max_workers: int = 16) -> list[dict]:
    """Sweep fused inner product configs for 2-term and 4-term dot products."""
    ppg_options = [p for p in PPGOption if p not in _PPG_SKIP]
    ppa_options = [p for p in PPAOption if p not in _PPA_SKIP]
    fsa_options = [f for f in FSAOption if f not in _FSA_SKIP]

    tasks = []
    for n_terms, n_bits, ppg, ppa, fsa, optim_type in product(
        [2, 4], bitwidths, ppg_options, ppa_options, fsa_options, ["area", "speed"]
    ):
        if n_bits < 2:
            continue
        if ppg == PPGOption.BOOTH_UNOPTIMISED and n_bits <= 2:
            continue
        for encoding in [Encoding.unsigned, Encoding.twos_complement]:
            sig = (is_signed(encoding), is_signed(encoding))
            if sig not in ppg.value.supported_signatures:
                continue
            tasks.append((ppg, ppa, fsa, n_terms, n_bits, encoding, optim_type))

    return _run_parallel("Inner products", tasks, eval_inner_product, max_workers)


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

    # Sort for deterministic output (stable diffs across re-runs)
    rows.sort(key=lambda r: (r["op"], r["a_w"], r["b_w"], r["signed"], r["transistor_count"], r["aig_depth"]))

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

    for (op, a_w_s, b_w_s, sign_key), group_rows in sorted(grouped.items()):
        w_key = f"{a_w_s}x{b_w_s}"
        filtered = _pareto_filter(group_rows) if pareto else group_rows
        # Sort within group for deterministic output
        filtered.sort(key=lambda r: (r["transistor_count"], r["aig_depth"]))
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
        bitwidths = [1, 2, 4, 8, 16, 32]
    if output_dir is None:
        output_dir = _OUT_DIR

    sys.setrecursionlimit(10000)
    print(f"Sweep: bitwidths={bitwidths}, format={fmt}, pareto={pareto}")
    t0 = time.time()

    all_rows = []
    all_rows.extend(sweep_adders(bitwidths, max_workers=max_workers))
    all_rows.extend(sweep_subtractors(bitwidths, max_workers=max_workers))
    all_rows.extend(sweep_multipliers(bitwidths, max_workers=max_workers))
    all_rows.extend(sweep_macs(bitwidths, max_workers=max_workers))
    # Multi-input adders for the common chain lengths real designs hit; `build_multi_input_add` snaps to the nearest N.
    all_rows.extend(sweep_mia(bitwidths, n_inputs_list=[3, 4, 5, 8],
                               max_workers=max_workers))
    all_rows.extend(sweep_inner_products(bitwidths, max_workers=max_workers))

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
    parser.add_argument("--bitwidths", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()
    main(
        bitwidths=args.bitwidths,
        fmt=args.format,
        pareto=not args.no_pareto,
        max_workers=args.workers,
    )
