"""Extract Pareto-optimal designs from flowy multi-run experiments.

Usage:
    python -m sprouthdl.pareto --experiment exp_mydesign_comb_20260325_102513
    python -m sprouthdl.pareto --experiment exp_... --root-database ./output/db -o pareto.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from flowy.data_structures.database import (
    RunDatabase,
    RunIdentifier,
    StageIdentifier,
)
from flowy.definitions import DatabaseConfig, ExperimentStages
from flowy.flows.reinforce.analysis.experiment_metrics import (
    get_exp_data_list_from_stage_id,
)


# Metrics extracted from each run's final_mockturtle_design record
_METRIC_KEYS = [
    "aig_count", "mockturtle_depth", "max_depth",
    "nb_transistors", "nb_cells", "lut6_count",
    "mockturtle_gates", "lut6_depth",
]


def _load_run_metrics(run_id: RunIdentifier) -> Optional[Dict[str, Any]]:
    """Load final_mockturtle_design metrics and file paths from a single run."""
    db = RunDatabase(run_id)
    try:
        record = db.load("final_mockturtle_design")
    except Exception:
        return None
    if record is None:
        return None

    metrics: Dict[str, Any] = {}
    for key in _METRIC_KEYS:
        entry = record.get(key)
        if entry is not None and hasattr(entry, "value"):
            metrics[key] = entry.value

    # Extract AIGER file paths
    aiger = record.get("aiger_filepath")
    aiger_map = record.get("aiger_map_filepath")
    if aiger is None or not hasattr(aiger, "path"):
        return None

    return {
        "run_id": str(run_id),
        "run_name": run_id.run,
        "aiger_filepath": aiger.path,
        "aiger_map_filepath": aiger_map.path if aiger_map and hasattr(aiger_map, "path") else None,
        "metrics": metrics,
    }


def get_all_run_ids(experiment: str, root_database: str = "output/db") -> List[RunIdentifier]:
    """Enumerate all run identifiers in an experiment's data_collection stage."""
    stage_id = StageIdentifier(
        root_database=root_database,
        experiment=experiment,
        stage=ExperimentStages.data_collection.value,
    )
    # Get run directory names
    stage_path = Path(root_database) / experiment / ExperimentStages.data_collection.value
    if not stage_path.is_dir():
        return []
    run_ids = []
    for d in sorted(stage_path.iterdir()):
        if d.is_dir() and d.name.startswith("run_"):
            run_ids.append(RunIdentifier(
                root_database=root_database,
                experiment=experiment,
                stage=ExperimentStages.data_collection.value,
                run=d.name,
            ))
    return run_ids


def extract_flowy_pareto(
    experiment: str,
    root_database: str = "output/db",
    area_metric: str = "aig_count",
    delay_metric: str = "mockturtle_depth",
) -> List[Dict[str, Any]]:
    """Extract Pareto-optimal designs from a flowy experiment.

    Returns a list of dicts sorted by area ascending, each containing:
        run_id, run_name, area, delay, aiger_filepath, aiger_map_filepath, metrics
    """
    run_ids = get_all_run_ids(experiment, root_database)
    if not run_ids:
        return []

    # Load metrics from all runs
    entries: List[Dict[str, Any]] = []
    for rid in run_ids:
        info = _load_run_metrics(rid)
        if info is None:
            continue
        area = info["metrics"].get(area_metric)
        delay = info["metrics"].get(delay_metric)
        if area is None or delay is None:
            continue
        info["area"] = area
        info["delay"] = delay
        entries.append(info)

    if not entries:
        return []

    # Pareto front: sort by area ascending, keep running min delay
    entries.sort(key=lambda e: (e["area"], e["delay"]))
    front: List[Dict[str, Any]] = []
    best_delay = float("inf")
    for e in entries:
        if e["delay"] < best_delay:
            front.append(e)
            best_delay = e["delay"]

    return front


def print_pareto(front: List[Dict[str, Any]], area_metric: str, delay_metric: str) -> None:
    """Print a formatted pareto front table."""
    if not front:
        print("Empty pareto front.")
        return
    print(f"Pareto front: {len(front)} designs ({area_metric} vs {delay_metric})")
    print(f"{'idx':>4}  {area_metric:>12}  {delay_metric:>16}  {'run':>30}")
    print("-" * 68)
    for i, e in enumerate(front):
        print(f"{i:4d}  {e['area']:12d}  {e['delay']:16d}  {e['run_name']:>30}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Pareto-optimal designs from a flowy experiment.")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--root-database", default="output/db", help="Database root path")
    parser.add_argument("--area-metric", default="aig_count", help="Area metric (default: aig_count)")
    parser.add_argument("--delay-metric", default="mockturtle_depth", help="Delay metric (default: mockturtle_depth)")
    parser.add_argument("-o", "--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    front = extract_flowy_pareto(
        args.experiment,
        root_database=args.root_database,
        area_metric=args.area_metric,
        delay_metric=args.delay_metric,
    )

    print_pareto(front, args.area_metric, args.delay_metric)

    if args.output:
        out_path = Path(args.output)
        # Remove non-serializable fields for JSON output
        json_front = []
        for e in front:
            entry = {k: v for k, v in e.items() if k != "run_id"}
            json_front.append(entry)
        with open(out_path, "w") as f:
            json.dump(json_front, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
