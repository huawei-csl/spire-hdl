"""Generate cache entries for flowy experiment designs.

By default generates caches for Pareto-front designs only.
With --all, generates caches for ALL runs (not just Pareto).

After running, @flowy_optimized(pareto_point=N) gets instant cache hits.

Usage:
    # Pareto points only (default, same as generate_pareto_caches.py)
    python -m sprouthdl.optimize.generate_caches --experiment exp_... --from-cache .sprouthdl_cache

    # All runs
    python -m sprouthdl.optimize.generate_caches --experiment exp_... --from-cache .sprouthdl_cache --all

    # Parallel
    python -m sprouthdl.optimize.generate_caches --experiment exp_... --from-cache .sprouthdl_cache --all --workers 10
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from sprouthdl.optimize import (
    _DISK_CACHE_DIR,
    _DISK_CACHE_VERSION,
    _read_cache_entry,
    _write_cache_entry,
    _spec_from_dict,
    _spec_to_dict,
)
from sprouthdl.sprouthdl import HDLType
from sprouthdl.optimize.pareto import (
    extract_flowy_pareto,
    get_all_run_ids,
    _load_run_metrics,
)


def _find_base_cache_key(cache_dir: Path) -> Optional[str]:
    """Find the base cache key from an existing cache entry's metadata."""
    versioned = cache_dir / _DISK_CACHE_VERSION
    if not versioned.is_dir():
        return None
    for entry_path in versioned.glob("*.json"):
        try:
            with open(entry_path) as f:
                data = json.load(f)
            metadata = data.get("metadata", {})
            base_key = metadata.get("base_cache_key")
            if base_key:
                return base_key
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def _aig_to_aag_via_module(
    aig_path: str, map_path: str, spec: Dict[str, HDLType]
) -> tuple[List[str], Dict[str, HDLType]]:
    """Convert AIG to AAG via Component/Module pipeline (same as flowy_optimize).

    Returns (aag_lines, output_spec).
    """
    from sprouthdl.sprouthdl import Signal
    from sprouthdl.sprouthdl_module import Component, Module
    from sprouthdl.sprouthdl_aiger import AigerExporter
    from dataclasses import make_dataclass

    io_sigs: Dict[str, Signal] = {}
    for name, typ in spec.items():
        io_sigs[name] = Signal(name=name, typ=typ, kind="input")

    IO = make_dataclass("IO", [(name, Signal) for name in io_sigs])
    io = IO(**io_sigs)

    class _Comp(Component):
        def __init__(self, io_obj):
            self.io = io_obj
        def elaborate(self):
            pass

    comp = _Comp(io)
    comp.from_aig_file(aig_path, map_path, make_internal=False)
    module = comp.to_module("optimized_design")

    aag_lines = AigerExporter(module).get_aag()
    out_spec = module.get_spec()
    return aag_lines, out_spec


def _get_spec_and_output_names_from_cache(
    base_cache_key: str, cache_dir: Path
) -> tuple[Dict[str, HDLType], List[str]]:
    """Get spec and output_names from an existing cache entry (base or any pareto)."""
    result = _read_cache_entry(base_cache_key, cache_dir)
    if result is not None:
        _, spec, output_names = result
        return spec, output_names

    versioned = cache_dir / _DISK_CACHE_VERSION
    for entry_path in versioned.glob(f"{base_cache_key}_pareto_*.json"):
        try:
            with open(entry_path) as f:
                data = json.load(f)
            spec = _spec_from_dict(data["spec"])
            output_names = data["output_names"]
            return spec, output_names
        except (json.JSONDecodeError, KeyError):
            continue

    raise RuntimeError(f"Cannot find spec/output_names from base key {base_cache_key[:12]}...")


def _process_one_entry(args_tuple):
    """Worker function for parallel cache generation."""
    i, entry, base_cache_key, spec_dict, output_names, cache_dir_str = args_tuple
    cache_dir = Path(cache_dir_str)
    pareto_key = f"{base_cache_key}_pareto_{i}"

    # Check if already cached
    existing = _read_cache_entry(pareto_key, cache_dir)
    if existing is not None:
        return i, "cached", None

    aig_path = entry["aiger_filepath"]
    map_path = entry["aiger_map_filepath"]

    if not Path(aig_path).exists():
        return i, "skip", f"AIG not found: {aig_path}"

    try:
        spec = _spec_from_dict(spec_dict)
        aag_lines, out_spec = _aig_to_aag_via_module(aig_path, map_path, spec)
    except Exception as e:
        return i, "skip", str(e)

    metadata = {"base_cache_key": base_cache_key, "pareto_point": i}
    _write_cache_entry(pareto_key, aag_lines, out_spec, output_names, cache_dir, metadata=metadata)
    metrics = entry.get("metrics", {})
    return i, "ok", f"aig_count={metrics.get('aig_count')}, depth={metrics.get('mockturtle_depth')}"


def generate_caches(
    experiment: str,
    base_cache_key: str,
    cache_dir: Path,
    root_database: str = "output/db",
    pareto_only: bool = True,
    workers: int = 1,
) -> int:
    """Generate cache entries for designs from a flowy experiment.

    Args:
        pareto_only: If True (default), only cache Pareto-front designs.
                     If False, cache ALL runs sorted by area metric.
        workers: Number of parallel workers (default: 1 = sequential).

    Returns the number of cache entries written.
    """
    if pareto_only:
        entries = extract_flowy_pareto(experiment, root_database=root_database)
        label = "Pareto-front"
    else:
        # Load ALL runs, sorted by aig_count ascending
        run_ids = get_all_run_ids(experiment, root_database)
        entries = []
        for rid in run_ids:
            info = _load_run_metrics(rid)
            if info is None:
                continue
            area = info["metrics"].get("aig_count")
            delay = info["metrics"].get("mockturtle_depth")
            if area is not None and delay is not None:
                info["area"] = area
                info["delay"] = delay
                entries.append(info)
        entries.sort(key=lambda e: (e["area"], e["delay"]))
        label = "all"

    if not entries:
        print(f"No {label} designs found.")
        return 0

    print(f"Generating caches for {len(entries)} {label} designs")

    # Get spec and output_names from an existing cache entry
    spec, output_names = _get_spec_and_output_names_from_cache(base_cache_key, cache_dir)
    spec_dict = _spec_to_dict(spec)

    if workers > 1:
        # Parallel
        tasks = [
            (i, entry, base_cache_key, spec_dict, output_names, str(cache_dir))
            for i, entry in enumerate(entries)
        ]
        count = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one_entry, t): t[0] for t in tasks}
            for future in as_completed(futures):
                i, status, msg = future.result()
                if status == "cached":
                    print(f"  [{i}] Already cached")
                    count += 1
                elif status == "ok":
                    print(f"  [{i}] Cached: {msg}")
                    count += 1
                else:
                    print(f"  [{i}] SKIP: {msg}")
    else:
        # Sequential
        count = 0
        for i, entry in enumerate(entries):
            pareto_key = f"{base_cache_key}_pareto_{i}"

            existing = _read_cache_entry(pareto_key, cache_dir)
            if existing is not None:
                print(f"  [{i}] Already cached")
                count += 1
                continue

            aig_path = entry["aiger_filepath"]
            map_path = entry["aiger_map_filepath"]

            if not Path(aig_path).exists():
                print(f"  [{i}] SKIP: AIG file not found: {aig_path}")
                continue

            try:
                aag_lines, out_spec = _aig_to_aag_via_module(aig_path, map_path, spec)
            except Exception as e:
                print(f"  [{i}] SKIP: {e}")
                continue

            metadata = {"base_cache_key": base_cache_key, "pareto_point": i}
            _write_cache_entry(pareto_key, aag_lines, out_spec, output_names, cache_dir, metadata=metadata)
            metrics = entry.get("metrics", {})
            print(f"  [{i}] Cached: aig_count={metrics.get('aig_count')}, "
                  f"depth={metrics.get('mockturtle_depth')}")
            count += 1

    return count


# Keep backward compatibility
generate_pareto_caches = generate_caches


def main():
    parser = argparse.ArgumentParser(
        description="Generate disk cache entries for flowy experiment designs.")
    parser.add_argument("--experiment", required=True, help="Flowy experiment name")
    parser.add_argument("--root-database", default="output/db", help="Database root")
    parser.add_argument("--base-cache-key", default=None,
                        help="Base cache key hex (auto-detected from --from-cache if omitted)")
    parser.add_argument("--from-cache", default=None,
                        help="Path to existing .sprouthdl_cache dir to auto-detect base key")
    parser.add_argument("--cache-dir", default=None,
                        help="Output cache directory (default: .sprouthdl_cache)")
    parser.add_argument("--all", action="store_true",
                        help="Generate caches for ALL runs, not just Pareto-front")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for cache generation (default: 1)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(".sprouthdl_cache")

    base_key = args.base_cache_key
    if base_key is None:
        search_dir = Path(args.from_cache) if args.from_cache else cache_dir
        base_key = _find_base_cache_key(search_dir)
        if base_key is None:
            print(f"Could not auto-detect base cache key from {search_dir}")
            print("Provide --base-cache-key explicitly or ensure a cached entry exists.")
            return

    mode = "all runs" if args.all else "Pareto-front only"
    print(f"Experiment:      {args.experiment}")
    print(f"Base cache key:  {base_key[:16]}...")
    print(f"Cache dir:       {cache_dir}")
    print(f"Mode:            {mode}")
    print(f"Workers:         {args.workers}")
    print()

    count = generate_caches(
        args.experiment, base_key, cache_dir,
        root_database=args.root_database,
        pareto_only=not args.all,
        workers=args.workers,
    )

    total_label = "all" if args.all else "Pareto"
    print(f"\nDone: {count} cache entries written ({total_label}).")


if __name__ == "__main__":
    main()
