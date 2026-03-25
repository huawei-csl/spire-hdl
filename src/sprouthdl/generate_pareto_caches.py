"""Generate cache entries for all Pareto-front designs from a flowy experiment.

After running a multi-run optimization, this tool pre-populates the disk cache
so that @flowy_optimized(pareto_point=N) gets instant cache hits.

Usage:
    python -m sprouthdl.generate_pareto_caches --experiment exp_... --base-cache-key abc123...
    python -m sprouthdl.generate_pareto_caches --experiment exp_... --from-cache .sprouthdl_cache
"""
from __future__ import annotations

import argparse
import json
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
from sprouthdl.pareto import extract_flowy_pareto


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

    # Build a Component with the right spec
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
    # Try base key first
    result = _read_cache_entry(base_cache_key, cache_dir)
    if result is not None:
        _, spec, output_names = result
        return spec, output_names

    # Try any pareto entry
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


def generate_pareto_caches(
    experiment: str,
    base_cache_key: str,
    cache_dir: Path,
    root_database: str = "output/db",
) -> int:
    """Generate cache entries for all Pareto-front designs.

    Returns the number of cache entries written.
    """
    front = extract_flowy_pareto(experiment, root_database=root_database)
    if not front:
        print("No Pareto-front designs found.")
        return 0

    # Get spec and output_names from an existing cache entry
    spec, output_names = _get_spec_and_output_names_from_cache(base_cache_key, cache_dir)

    count = 0
    for i, entry in enumerate(front):
        pareto_key = f"{base_cache_key}_pareto_{i}"

        # Check if already cached
        existing = _read_cache_entry(pareto_key, cache_dir)
        if existing is not None:
            print(f"  [{i}] Already cached: {pareto_key[:16]}...")
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate disk cache entries for all Pareto-front designs.")
    parser.add_argument("--experiment", required=True, help="Flowy experiment name")
    parser.add_argument("--root-database", default="output/db", help="Database root")
    parser.add_argument("--base-cache-key", default=None,
                        help="Base cache key hex (auto-detected from --from-cache if omitted)")
    parser.add_argument("--from-cache", default=None,
                        help="Path to existing .sprouthdl_cache dir to auto-detect base key")
    parser.add_argument("--cache-dir", default=None,
                        help="Output cache directory (default: .sprouthdl_cache)")
    args = parser.parse_args()

    # Resolve cache directory
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(".sprouthdl_cache")

    # Resolve base cache key
    base_key = args.base_cache_key
    if base_key is None:
        search_dir = Path(args.from_cache) if args.from_cache else cache_dir
        base_key = _find_base_cache_key(search_dir)
        if base_key is None:
            print(f"Could not auto-detect base cache key from {search_dir}")
            print("Provide --base-cache-key explicitly or ensure a cached entry exists.")
            return

    print(f"Experiment:      {args.experiment}")
    print(f"Base cache key:  {base_key[:16]}...")
    print(f"Cache dir:       {cache_dir}")
    print()

    front = extract_flowy_pareto(args.experiment, root_database=args.root_database)
    print(f"Pareto front: {len(front)} designs")
    print()

    count = generate_pareto_caches(
        args.experiment, base_key, cache_dir, root_database=args.root_database
    )
    print(f"\nDone: {count}/{len(front)} cache entries written.")


if __name__ == "__main__":
    main()
