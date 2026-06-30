"""
Evaluation script for sweeping dim and width parameters of the matmul accumulate core.
Compares Winograd and Base architectures.
Uses multiprocessing with timeout to run evaluations in parallel.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from spire.helpers import get_yosys_metrics


class Architecture(str, Enum):
    BASE = "base"
    WINOGRAD = "winograd"


@dataclass
class EvalConfig:
    dim_m: int
    dim_n: int
    dim_k: int
    a_width: int
    b_width: int
    architecture: Architecture


def run_single_evaluation(config: EvalConfig) -> Optional[dict[str, Any]]:
    """Run a single evaluation with the given configuration.

    Returns the yosys metrics dict or None if an error occurred.
    """
    try:
        # Import the appropriate module based on architecture
        from spire.arithmetic.int_arithmetic_config import AdderConfig, MultiplierConfig
        if config.architecture == Architecture.WINOGRAD:
            from spire.cores.matmul_accumulate.matmul_accumulate_core_winograd import (
                MMAcCfg,
                MMAcDims,
                MMAcWidths,
                MatmulAccumulateComponent,
                max_y_width_unsigned,
            )
        else:
            from spire.cores.matmul_accumulate.matmul_accumulate_core import (
                MMAcCfg,
                MMAcDims,
                MMAcWidths,
                MatmulAccumulateComponent,
                max_y_width_unsigned,
            )

        c_width = max_y_width_unsigned(config.a_width, config.b_width, config.dim_k, include_carry_from_add=False)
        encoding = Encoding.twos_complement
        signed_io_type = True
        n_iter_optimizations = 0

        mult_cfg = MultiplierConfig(
            use_operator=False,
            multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
            encodings=TwoInputAritEncodings.with_enc(encoding),
            ppg_opt=PPGOption.BAUGH_WOOLEY if is_signed(encoding) else PPGOption.AND,
            ppa_opt=PPAOption.WALLACE_TREE,
            fsa_opt=FSAOption.RIPPLE_CARRY,
        )
        add_cfg = AdderConfig(
            use_operator=False, fsa_opt=FSAOption.RIPPLE_CARRY, full_output_bit=True, encoding=encoding
        )

        core_config = MMAcCfg(
            dims=MMAcDims(dim_m=config.dim_m, dim_n=config.dim_n, dim_k=config.dim_k),
            widths=MMAcWidths(a_width=config.a_width, b_width=config.b_width, c_width=c_width),
            mult_cfg=mult_cfg,
            add_cfg=add_cfg,
        )

        core = MatmulAccumulateComponent(core_config, signed_io_type=signed_io_type)
        module = core.to_module(f"matmul_accumulate_core_{config.architecture.value}")

        yosys_metrics = get_yosys_metrics(module, n_iter_optimizations=n_iter_optimizations)

        return {
            "config": {
                "dim_m": config.dim_m,
                "dim_n": config.dim_n,
                "dim_k": config.dim_k,
                "a_width": config.a_width,
                "b_width": config.b_width,
                "architecture": config.architecture.value,
            },
            "metrics": yosys_metrics,
        }
    except Exception as e:
        print(f"Error for config {config}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _worker_fn(config: EvalConfig, result_queue: mp.Queue) -> None:
    """Worker function that runs evaluation and puts result in queue."""
    result = run_single_evaluation(config)
    result_queue.put(result)


def run_sweep(
    dim_vals: list[int],
    width_vals: list[int],
    architectures: list[Architecture],
    timeout_minutes: float = 5.0,
    n_workers: int = 4,
    output_file: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Run parameter sweep over dim_vals and width_vals for multiple architectures.

    Args:
        dim_vals: List of dimension values (applied to dim_m, dim_n, dim_k)
        width_vals: List of width values (applied to a_width, b_width)
        architectures: List of architectures to evaluate
        timeout_minutes: Maximum time per evaluation in minutes
        n_workers: Number of parallel workers
        output_file: Path to save results (JSON format)

    Returns:
        List of result dictionaries
    """
    configs = []
    for arch in architectures:
        for dim_val in dim_vals:
            for width_val in width_vals:
                configs.append(EvalConfig(
                    dim_m=dim_val,
                    dim_n=dim_val,
                    dim_k=dim_val,
                    a_width=width_val,
                    b_width=width_val,
                    architecture=arch,
                ))

    print(f"Running sweep with {len(configs)} configurations")
    print(f"Architectures: {[a.value for a in architectures]}")
    print(f"Dimensions: {dim_vals}")
    print(f"Widths: {width_vals}")
    print(f"Timeout: {timeout_minutes} minutes per evaluation")
    print(f"Workers: {n_workers}")
    print("-" * 60)

    results = []
    completed_count = 0
    failed_count = 0
    timeout_count = 0

    timeout_seconds = timeout_minutes * 60

    import time

    # Use individual processes with proper timeout handling
    pbar = tqdm(total=len(configs), desc="Evaluating", unit="config",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    # Process in batches equal to n_workers for true parallel execution with timeout
    for batch_idx in range(0, len(configs), n_workers):
        batch_configs = configs[batch_idx:batch_idx + n_workers]
        pending = []  # (config, process, queue, start_time)

        # Start all processes in this batch
        batch_start_time = time.time()
        for config in batch_configs:
            result_queue = mp.Queue()
            p = mp.Process(target=_worker_fn, args=(config, result_queue))
            p.start()
            pending.append((config, p, result_queue))

        # Poll until all processes complete or timeout
        while pending:
            still_pending = []
            elapsed = time.time() - batch_start_time

            for config, p, result_queue in pending:
                if not p.is_alive():
                    # Process finished - collect result
                    pbar.set_postfix_str(f"{config.architecture.value} d={config.dim_m} w={config.a_width}")
                    try:
                        if not result_queue.empty():
                            result = result_queue.get_nowait()
                            if result is not None:
                                results.append(result)
                                completed_count += 1
                                transistors = result['metrics'].get('estimated_num_transistors', 'N/A')
                                tqdm.write(f"  Completed: {config.architecture.value} dim={config.dim_m}, width={config.a_width} -> "
                                           f"transistors={transistors}")
                            else:
                                results.append({
                                    "config": {
                                        "dim_m": config.dim_m,
                                        "dim_n": config.dim_n,
                                        "dim_k": config.dim_k,
                                        "a_width": config.a_width,
                                        "b_width": config.b_width,
                                        "architecture": config.architecture.value,
                                    },
                                    "metrics": None,
                                    "error": "Evaluation returned None",
                                })
                                failed_count += 1
                                tqdm.write(f"  Failed: {config.architecture.value} dim={config.dim_m}, width={config.a_width}")
                        else:
                            results.append({
                                "config": {
                                    "dim_m": config.dim_m,
                                    "dim_n": config.dim_n,
                                    "dim_k": config.dim_k,
                                    "a_width": config.a_width,
                                    "b_width": config.b_width,
                                    "architecture": config.architecture.value,
                                },
                                "metrics": None,
                                "error": "No result returned",
                            })
                            failed_count += 1
                            tqdm.write(f"  Failed: {config.architecture.value} dim={config.dim_m}, width={config.a_width} (no result)")
                    except Exception as e:
                        results.append({
                            "config": {
                                "dim_m": config.dim_m,
                                "dim_n": config.dim_n,
                                "dim_k": config.dim_k,
                                "a_width": config.a_width,
                                "b_width": config.b_width,
                                "architecture": config.architecture.value,
                            },
                            "metrics": None,
                            "error": str(e),
                        })
                        failed_count += 1
                        tqdm.write(f"  Error: {config.architecture.value} dim={config.dim_m}, width={config.a_width} -> {e}")

                    pbar.update(1)
                    pbar.set_description(f"Evaluating (ok={completed_count}, fail={failed_count}, timeout={timeout_count})")

                elif elapsed >= timeout_seconds:
                    # Timeout reached - terminate process
                    pbar.set_postfix_str(f"{config.architecture.value} d={config.dim_m} w={config.a_width}")
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
                        p.join()

                    results.append({
                        "config": {
                            "dim_m": config.dim_m,
                            "dim_n": config.dim_n,
                            "dim_k": config.dim_k,
                            "a_width": config.a_width,
                            "b_width": config.b_width,
                            "architecture": config.architecture.value,
                        },
                        "metrics": None,
                        "error": "Timeout",
                    })
                    timeout_count += 1
                    tqdm.write(f"  Timeout: {config.architecture.value} dim={config.dim_m}, width={config.a_width}")
                    pbar.update(1)
                    pbar.set_description(f"Evaluating (ok={completed_count}, fail={failed_count}, timeout={timeout_count})")
                else:
                    # Still running and within timeout
                    still_pending.append((config, p, result_queue))

            pending = still_pending
            if pending:
                time.sleep(0.5)  # Poll every 0.5 seconds

    pbar.close()

    # Save results
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"mmac_sweep_results_{timestamp}.json"

    output_path = Path(__file__).parent / output_file
    with open(output_path, "w") as f:
        json.dump({
            "sweep_config": {
                "dim_vals": dim_vals,
                "width_vals": width_vals,
                "architectures": [a.value for a in architectures],
                "timeout_minutes": timeout_minutes,
                "n_workers": n_workers,
            },
            "results": results,
        }, f, indent=2)

    print("-" * 60)
    print(f"Results saved to: {output_path}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sweep parameters for matmul accumulate core architectures")
    parser.add_argument("--test", action="store_true", help="Run with reduced parameters for testing")
    parser.add_argument("--timeout", type=float, default=2.0, help="Timeout in minutes per evaluation")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="mmac_sweep_results.json", help="Output file name")
    args = parser.parse_args()

    if args.test:
        # Reduced parameters for quick testing
        dim_vals = [2, 4]
        width_vals = [4, 8]
        timeout_minutes = 1.0
    else:
        # Full sweep
        dim_vals = [2, 4, 8, 16, 32]
        width_vals = [4, 8, 16, 32]
        timeout_minutes = args.timeout

    n_workers = args.workers if args.workers else min(4, mp.cpu_count())

    # Evaluate both architectures
    architectures = [Architecture.BASE, Architecture.WINOGRAD]

    results = run_sweep(
        dim_vals=dim_vals,
        width_vals=width_vals,
        architectures=architectures,
        timeout_minutes=timeout_minutes,
        n_workers=n_workers,
        output_file=args.output,
    )

    completed = len([r for r in results if r.get('metrics') is not None])
    print(f"\nCompleted {completed} / {len(results)} evaluations")


if __name__ == "__main__":
    main()
