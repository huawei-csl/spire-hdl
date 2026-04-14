"""Flowy circuit optimization — decorator, multi-run, and Pareto utilities."""

from sprouthdl.optimize.optimize import (  # noqa: F401
    # Public API
    abc_optimize,
    abc_optimized,
    arithmetic_optimized,
    flowy_optimize,
    flowy_optimized,
    set_cache_dir,
    get_cache_dir,
    clear_optimization_cache,
    # Internal helpers used by generate_caches
    _cache,
    _DISK_CACHE_DIR,
    _DISK_CACHE_VERSION,
    _DEFAULT_OPTIMIZE_KWARGS,
    _compute_disk_cache_key,
    _resolve_cache_dir,
    _write_cache_entry,
    _read_cache_entry,
    _build_component,
    _hdltype_to_dict,
    _hdltype_from_dict,
    _spec_to_dict,
    _spec_from_dict,
)
