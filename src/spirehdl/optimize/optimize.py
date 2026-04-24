"""
Decorator and utilities for automated Flowy circuit optimization.

Usage:
    from spirehdl.optimize import flowy_optimized

    @flowy_optimized
    def my_adder(a, b):
        return a + b

    result = my_adder(signal_x, signal_y)
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import shutil
import tempfile
import time
from dataclasses import make_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from spirehdl.spirehdl import Expr, HDLType, Signal
from spirehdl.spirehdl_module import Component, Module


# ---------------------------------------------------------------------------
# Cache: maps verilog hash -> (aag_lines, spec, output_names)
# ---------------------------------------------------------------------------
_cache: Dict[str, Tuple[List[str], Dict[str, HDLType], List[str]]] = {}

# Persistent disk cache directory. Set to None to disable disk caching.
_DISK_CACHE_DIR: Path | None = Path(".spirehdl_cache")
_DISK_CACHE_VERSION: str = "v1"

# Default kwargs for flowy_optimize, used by the @flowy_optimized decorator.
# User-supplied kwargs override these.  A flowy_config.json in cwd (or any
# parent) is loaded on top of the hardcoded defaults at import time.
_DEFAULT_OPTIMIZE_KWARGS: Dict[str, Any] = {
    "direct": True,
    "iterations": 1,
    "mockturtle_chains": 10,
    "mockturtle_chain_len": 10,
    "mockturtle_chain_workers": 10,
    "nb_runs": 1,
    "nb_workers": 10,
}


def set_cache_dir(path: str | Path | None) -> None:
    """Configure the persistent cache directory. Pass None to disable disk caching."""
    global _DISK_CACHE_DIR
    _DISK_CACHE_DIR = Path(path) if path is not None else None


def get_cache_dir() -> Path | None:
    """Return the current persistent cache directory, or None if disabled."""
    return _DISK_CACHE_DIR


def _load_config_defaults() -> None:
    """Search cwd and its parents for flowy_config.json and merge into defaults."""
    path = Path.cwd()
    for directory in [path, *path.parents]:
        cfg = directory / "flowy_config.json"
        if cfg.is_file():
            with open(cfg) as f:
                data = json.load(f)
            if "cache_dir" in data:
                set_cache_dir(data.pop("cache_dir"))
            _DEFAULT_OPTIMIZE_KWARGS.update(data)
            print(f"[spirehdl] Loaded flowy config from {cfg}")
            break


_load_config_defaults()


def clear_optimization_cache(*, disk: bool = True) -> None:
    """Clear all cached optimized circuits.

    Parameters
    ----------
    disk : bool
        If True (default), also remove the persistent disk cache directory.
    """
    _cache.clear()
    if disk and _DISK_CACHE_DIR is not None:
        cache_dir = _DISK_CACHE_DIR / _DISK_CACHE_VERSION
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _hdltype_to_dict(t: HDLType) -> dict:
    return {"width": t.width, "signed": t.signed, "is_bool": t.is_bool}


def _hdltype_from_dict(d: dict) -> HDLType:
    return HDLType(width=d["width"], signed=d["signed"], is_bool=d.get("is_bool", False))


def _spec_to_dict(spec: Dict[str, HDLType]) -> Dict[str, dict]:
    return {name: _hdltype_to_dict(t) for name, t in spec.items()}


def _spec_from_dict(d: Dict[str, dict]) -> Dict[str, HDLType]:
    return {name: _hdltype_from_dict(v) for name, v in d.items()}


def _compute_disk_cache_key(
    verilog_content: str,
    other_args: Dict[str, Any],
    optimize_kwargs: Dict[str, Any],
) -> str:
    """Compute a deterministic SHA-256 hash for disk cache lookup."""
    h = hashlib.sha256()
    h.update(verilog_content.encode("utf-8"))
    other_canonical = json.dumps(
        {k: repr(v) for k, v in sorted(other_args.items())}, sort_keys=True
    )
    h.update(other_canonical.encode("utf-8"))
    kwargs_canonical = json.dumps(
        {k: repr(v) for k, v in sorted(optimize_kwargs.items())}, sort_keys=True
    )
    h.update(kwargs_canonical.encode("utf-8"))
    return h.hexdigest()


def _resolve_cache_dir(
    fn: Callable[..., Any] | None = None,
    explicit_dir: str | Path | None = None,
) -> Path | None:
    """Resolve the effective disk cache directory.

    Priority (highest to lowest):
      1. *explicit_dir* — set via ``@flowy_optimized(cache_dir=...)``
      2. Local cache folder next to the script that defines *fn*
         (only if the folder already exists)
      3. Global ``_DISK_CACHE_DIR`` (set via ``set_cache_dir()`` or
         ``flowy_config.json``; defaults to ``".spirehdl_cache"``)
    """
    # 1. Explicit override from decorator
    if explicit_dir is not None:
        return Path(explicit_dir)
    # 2. Cache folder next to the script defining fn
    if fn is not None:
        try:
            script_dir = Path(inspect.getfile(fn)).resolve().parent
            local_cache = script_dir / (_DISK_CACHE_DIR.name if _DISK_CACHE_DIR else ".spirehdl_cache")
            if local_cache.is_dir():
                return local_cache
        except (TypeError, OSError):
            pass
    # 3. Global default
    return _DISK_CACHE_DIR


def _write_cache_entry(
    cache_key_hex: str,
    aag_lines: List[str],
    spec: Dict[str, HDLType],
    output_names: List[str],
    cache_dir: Path | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Write a cache entry to disk as JSON."""
    if cache_dir is None:
        cache_dir = _DISK_CACHE_DIR
    if cache_dir is None:
        return
    versioned = cache_dir / _DISK_CACHE_VERSION
    versioned.mkdir(parents=True, exist_ok=True)
    entry = {
        "version": 1,
        "output_names": output_names,
        "spec": _spec_to_dict(spec),
        "aag_lines": aag_lines,
    }
    if metadata:
        entry["metadata"] = metadata
    target = versioned / f"{cache_key_hex}.json"
    tmp_path = target.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(entry, f, indent=2)
    tmp_path.rename(target)


def _read_cache_entry(
    cache_key_hex: str,
    cache_dir: Path | None = None,
) -> Optional[Tuple[List[str], Dict[str, HDLType], List[str]]]:
    """Read a cache entry from disk. Returns None on miss or corruption."""
    if cache_dir is None:
        cache_dir = _DISK_CACHE_DIR
    if cache_dir is None:
        return None
    cache_file = cache_dir / _DISK_CACHE_VERSION / f"{cache_key_hex}.json"
    if not cache_file.is_file():
        return None
    try:
        with open(cache_file) as f:
            data = json.load(f)
        if data.get("version") != 1:
            return None
        return data["aag_lines"], _spec_from_dict(data["spec"]), data["output_names"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# flowy_optimize  (moved from synthesise_fp2.py)
# ---------------------------------------------------------------------------

def flowy_optimize(m: Module | Component,
                   nb_runs: int = 1,
                   nb_workers: int = 10,
                   iterations: int = 1,
                   mockturtle_chains: int = 1,
                   mockturtle_chain_len: int = 10,
                   mockturtle_chain_workers: int = 1,
                   selection_metric: str | None = None,
                   verbose: bool = False,
                   direct: bool = False,
                   visualize: bool = False,
                   pareto_point: int | None = None) -> Module | Component:
    """Run Flowy optimization on a Module or Component and return the optimized design.

    Parameters
    ----------
    direct : bool
        If True, call the statistical run_flow directly (single process, no Docker).
        If False (default), use the Docker-based run_flows_in_docker launcher.
    visualize : bool
        If True, visualize the optimization results after completion.
    """
    from flowy.flows.reinforce.data_collection.lib.definitions import (
        RecipeSelection, SelectionMetric,
    )
    from flowy.data_structures.database import RunDatabase, RunIdentifier
    from flowy.flows.sim.extract_best_design import extract_and_store_best_design

    if selection_metric is None:
        selection_metric = SelectionMetric.aig_count
    elif isinstance(selection_metric, str):
        selection_metric = SelectionMetric(selection_metric)

    if isinstance(m, Component):
        m = m.to_module("mydesign_comb")
    name_initial: str = m.name
    m.name = "mydesign_comb"
    verilog_code: str = m.to_verilog()

    random_hash = os.urandom(8).hex()
    filename = f"my_logical_design_{random_hash}.v"
    tempdir = tempfile.gettempdir()
    verilog_path: str = os.path.join(tempdir, filename)

    with open(verilog_path, "w") as f:
        f.write(verilog_code)

    datecode = time.strftime("%Y%m%d_%H%M%S")
    experiment: str = f"exp_{name_initial}_{datecode}"

    if direct:
        # Direct mode — call statistical run_flow locally.
        # When nb_runs > 1, runs execute in parallel with isolated working dirs.
        from flowy.flows.reinforce.run.statistical.run_flow import (
            run_flow as statistical_run_flow,
        )

        run_kwargs = dict(
            use_mockturtle=True,
            iterations=iterations,
            chains=mockturtle_chains,
            chain_len=mockturtle_chain_len,
            chain_workers=mockturtle_chain_workers,
            env_option="auto",
            experiment=experiment,
            recipe_selection=RecipeSelection.PERFORMANCE_SAMPLING,
            strategy_name="equal",
            selection_metric=selection_metric,
            verilog_file=verilog_path,
            compression_scripts_per_step=3,
            scripts_per_step=2,
            simulation_tb=False,
            verbose=verbose,
        )

        if nb_runs <= 1:
            statistical_run_flow(**run_kwargs)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from pathlib import Path as _Path

            actual_workers = min(nb_workers, nb_runs)
            print(f"[spirehdl] Running {nb_runs} optimization runs with {actual_workers} parallel workers")

            temp_dirs: list[str] = []
            try:
                for i in range(nb_runs):
                    td = tempfile.mkdtemp(prefix=f"flowy_run_{i}_")
                    temp_dirs.append(td)

                with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                    futures = {
                        executor.submit(
                            statistical_run_flow,
                            **run_kwargs,
                            working_dir=_Path(td),
                        ): idx
                        for idx, td in enumerate(temp_dirs)
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            future.result()
                            print(f"[spirehdl] Run {idx + 1}/{nb_runs} completed")
                        except Exception as exc:
                            print(f"[spirehdl] Run {idx + 1}/{nb_runs} failed: {exc}")
            finally:
                for td in temp_dirs:
                    shutil.rmtree(td, ignore_errors=True)
    else:
        # Docker-based multi-run launcher (original path)
        import flowy.flows.reinforce.run.statistical.run_flows_in_docker as run_flows_in_docker
        args = run_flows_in_docker.build_parser().parse_args([])

        args.experiment = experiment
        args.nb_runs = nb_runs
        args.nb_workers = nb_workers
        args.iterations = iterations
        args.mockturtle_chains = mockturtle_chains
        args.mockturtle_chain_len = mockturtle_chain_len
        args.mockturtle_chain_workers = mockturtle_chain_workers
        args.recipe_selection = RecipeSelection.PERFORMANCE_SAMPLING.value
        args.strategy_name = "equal"
        args.debug = False
        args.selection_metric = selection_metric.value
        args.verilog_file = verilog_path
        args.compression_scripts_per_step = 3
        args.scripts_per_step = 2
        args.simulation_tb = False
        args.extra_files = ""
        args.verbose = verbose

        run_flows_in_docker.run_with_args(args, commit_hash="e30da590ef15a869b1095d3b8baf5058b3e650d5")

    extract_and_store_best_design(
        experiment=experiment, target_metrics=[selection_metric]
    )

    if visualize:
        from flowy.flows.reinforce.analysis.visualize_runs import visualize_main
        from flowy.data_structures.database import ExperimentIdentifier
        from flowy.definitions import DatabaseConfig
        visualize_main(ExperimentIdentifier(
            root_database=DatabaseConfig.default_path, experiment=experiment
        ))

    os.remove(verilog_path)

    if pareto_point is not None:
        # Select a specific design from the Pareto front
        from spirehdl.optimize.pareto import extract_flowy_pareto
        front = extract_flowy_pareto(experiment)
        if pareto_point >= len(front):
            raise IndexError(
                f"pareto_point={pareto_point} but only {len(front)} "
                f"designs on the Pareto front (0..{len(front) - 1})"
            )
        entry = front[pareto_point]
        aig_file_path = entry["aiger_filepath"]
        aiger_map_file_path = entry["aiger_map_filepath"]
        print(f"[spirehdl] Pareto point {pareto_point}: "
              f"aig_count={entry['metrics'].get('aig_count')}, "
              f"mockturtle_depth={entry['metrics'].get('mockturtle_depth')}")
    else:
        best_design = RunIdentifier(
            root_database="output/db", experiment=experiment,
            stage="analysis", run="best_designs",
        )
        best_design_item = RunDatabase(best_design).load(
            f"final_mockturtle_design_best_design_{selection_metric.value}"
        )
        aig_file_path = best_design_item.get("aiger_filepath").path
        aiger_map_file_path = best_design_item.get("aiger_map_filepath").path

    c_out: Component = m.to_component().from_aig_file(
        aig_file_path, aiger_map_file_path, make_internal=False
    )
    module: Module = c_out.to_module("optimized_design")

    if isinstance(m, Module):
        return module
    else:
        return c_out


# ---------------------------------------------------------------------------
# Internal helpers for the decorator
# ---------------------------------------------------------------------------

def _push_shared_state() -> dict:
    """Snapshot and reset the global shared-wire state.

    Returns a snapshot dict to pass to :func:`_pop_shared_state`.
    """
    from spirehdl.spirehdl import _SharedCache
    snapshot = {
        "counts":     dict(_SharedCache.counts),
        "expr2sig":   dict(_SharedCache.expr2sig),
        "wires":      list(_SharedCache.wires),
        "index":      _SharedCache.index,
        "used_names": set(_SharedCache.used_names),
    }
    _SharedCache.reset()
    return snapshot


def _pop_shared_state(snapshot: dict) -> None:
    """Restore the global shared-wire state from a previous snapshot."""
    from spirehdl.spirehdl import _SharedCache
    _SharedCache.reset()
    _SharedCache.counts.update(snapshot["counts"])
    _SharedCache.expr2sig.update(snapshot["expr2sig"])
    _SharedCache.wires.extend(snapshot["wires"])
    _SharedCache.index = snapshot["index"]
    _SharedCache.used_names.update(snapshot["used_names"])


def _build_component(
    fn: Callable[..., Any],
    logic_args: Dict[str, Tuple[int, bool]],
    other_args: Dict[str, Any],
) -> Tuple[Component, List[str]]:
    """Build a Component from the decorated function using placeholder signals.

    Parameters
    ----------
    fn : callable
        The original decorated function.
    logic_args : dict
        {param_name: (width, signed)} for each Expr argument.
    other_args : dict
        {param_name: value} for each non-Expr argument.

    Returns
    -------
    tuple[Component, list[str]]
        The built Component and the list of output names.
    """
    # Push a clean shared-wire context so the sub-circuit gets
    # deterministic signal names (sig_0, sig_1, …) regardless of what
    # signals the caller has already created.  This makes the Verilog
    # content — and therefore the disk-cache key — independent of
    # context, which is critical for cache hits across different
    # multiplier / adder configurations.
    snapshot = _push_shared_state()

    # Create placeholder input signals
    input_sigs: Dict[str, Signal] = {}
    for name, (width, signed) in logic_args.items():
        typ: HDLType = HDLType(width, signed=signed)
        input_sigs[name] = Signal(name=name, typ=typ, kind="input")

    # Build kwargs for the function call: logic args get placeholder signals,
    # non-logic args get their actual values
    call_kwargs: Dict[str, Any] = {}
    call_kwargs.update(input_sigs)
    call_kwargs.update(other_args)

    # Call function with placeholders in the right parameter order
    fn_sig: inspect.Signature = inspect.signature(fn)
    call_args: List[Any] = []
    for param_name, param in fn_sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            # Special Case - Handle *args (VAR_POSITIONAL): Collect all expanded varargs (e.g. pref_vals_a, pref_vals_b, ...)
            i = 0
            while f"{param_name}_{chr(ord('a') + i)}" in call_kwargs:
                call_args.append(call_kwargs[f"{param_name}_{chr(ord('a') + i)}"])
                i += 1
        else:
            call_args.append(call_kwargs[param_name])

    result: Any = fn(*call_args)

    # Normalize result to list of (name, expr)
    # Names must not share a common prefix+digit pattern (e.g. y0,y1 or out_0,out_1)
    # because yosys merges such ports into a single flattened bus.
    # Use letter suffixes (res_a, res_b, ...) to keep them distinct.
    outputs: List[Tuple[str, Expr]]
    if isinstance(result, tuple):
        outputs = [
            (f"res_{chr(ord('a') + i)}", expr)
            for i, expr in enumerate(result)
        ]
    else:
        outputs = [("y", result)]

    # Create output signals and connect
    output_sigs: Dict[str, Signal] = {}
    for out_name, expr in outputs:
        out_sig: Signal = Signal(name=out_name, typ=expr.typ, kind="output")
        out_sig <<= expr
        output_sigs[out_name] = out_sig

    # Build a concrete Component using make_dataclass so iter_values works
    all_sigs: Dict[str, Signal] = {}
    all_sigs.update(input_sigs)
    all_sigs.update(output_sigs)

    IO: type = make_dataclass("IO", [(name, Signal) for name in all_sigs])
    io: Any = IO(**all_sigs)

    class _GeneratedComponent(Component):
        def __init__(self, io_obj: Any) -> None:
            self.io = io_obj

        def elaborate(self) -> None:
            pass

    comp: Component = _GeneratedComponent(io)
    output_names = [name for name, _ in outputs]

    # Pop back to the caller's shared-wire context so the outer circuit
    # is unaffected by the temporary sub-circuit we just built.
    _pop_shared_state(snapshot)

    return comp, output_names


CacheSpec = Literal["none", "mem", "disk", "both"]


def _parse_cache_spec(spec: CacheSpec) -> Tuple[bool, bool]:
    """Return ``(mem, disk)`` enabled flags for a cache spec string."""
    if spec == "none":
        return False, False
    if spec == "mem":
        return True, False
    if spec == "disk":
        return False, True
    if spec == "both":
        return True, True
    raise ValueError(
        f"Invalid cache spec {spec!r}: must be 'none', 'mem', 'disk', or 'both'."
    )


def _resolve_rw_flags(
    cache_read: CacheSpec,
    cache_write: CacheSpec,
) -> Tuple[bool, bool, bool, bool]:
    """Return ``(read_mem, read_disk, write_mem, write_disk)``."""
    read_mem, read_disk = _parse_cache_spec(cache_read)
    write_mem, write_disk = _parse_cache_spec(cache_write)
    return read_mem, read_disk, write_mem, write_disk


def _cache_lookup(
    cache_key_hex: str,
    use_mem_cache: bool = True,
    use_disk_cache: bool = True,
    cache_dir: Path | None = None,
) -> Optional[Tuple[List[str], Dict[str, HDLType], List[str]]]:
    """Check in-memory cache, then disk cache. Returns None on miss."""
    if use_mem_cache and cache_key_hex in _cache:
        return _cache[cache_key_hex]
    if use_disk_cache:
        disk_result = _read_cache_entry(cache_key_hex, cache_dir)
        if disk_result is not None:
            print(f"[spirehdl] Disk cache hit: {cache_key_hex[:12]}...")
            if use_mem_cache:
                _cache[cache_key_hex] = disk_result
            return disk_result
    return None


def _store_cache(
    cache_key_hex: str,
    aag_lines: List[str],
    spec: Dict[str, HDLType],
    output_names: List[str],
    use_mem_cache: bool = True,
    use_disk_cache: bool = True,
    cache_dir: Path | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Store result in in-memory and/or disk caches."""
    if use_mem_cache:
        _cache[cache_key_hex] = (aag_lines, spec, output_names)
    if use_disk_cache:
        _write_cache_entry(cache_key_hex, aag_lines, spec, output_names, cache_dir, metadata=metadata)
        print(f"[spirehdl] Cached optimization result: {cache_key_hex[:12]}...")


def _optimize_and_cache(
    fn: Callable[..., Any],
    logic_args: Dict[str, Tuple[int, bool]],
    other_args: Dict[str, Any],
    optimize_kwargs: Dict[str, Any],
    cache_read: CacheSpec = "both",
    cache_write: CacheSpec = "both",
    explicit_cache_dir: str | Path | None = None,
    nb_runs: int = 1,
    nb_workers: int = 10,
    pareto_point: int | None = None,
) -> Tuple[List[str], Dict[str, HDLType], List[str]]:
    """Build component, check caches, optimize if needed, cache the AAG lines.

    Returns
    -------
    tuple[list[str], dict[str, HDLType], list[str]]
        (aag_lines, spec, output_names)
    """
    from spirehdl.spirehdl_aiger import AigerExporter

    comp, output_names = _build_component(fn, logic_args, other_args)
    module: Module = comp.to_module("mydesign_comb")

    merged_kwargs = {**_DEFAULT_OPTIMIZE_KWARGS, **optimize_kwargs}
    # nb_workers is pure parallelism — doesn't affect the result, exclude from cache key
    merged_kwargs.pop("nb_workers", None)
    merged_kwargs.pop("nb_runs", None) # needs to be deleted, just here for compatibility with some existing caches

    read_mem, read_disk, write_mem, write_disk = _resolve_rw_flags(cache_read, cache_write)

    # Resolve cache directory (see _resolve_cache_dir for priority).
    # Needed if either read OR write touches disk.
    cache_dir: Path | None = (
        _resolve_cache_dir(fn, explicit_cache_dir)
        if (read_disk or write_disk) else None
    )

    # Compute cache key from verilog content
    verilog_content: str = module.to_verilog()
    base_cache_key: str = _compute_disk_cache_key(verilog_content, other_args, merged_kwargs)
    cache_key_hex: str = base_cache_key
    # Append pareto_point suffix so each point gets its own cache entry
    if pareto_point is not None:
        cache_key_hex = f"{base_cache_key}_pareto_{pareto_point}"

    # Check caches before running expensive optimization
    cached = _cache_lookup(cache_key_hex, read_mem, read_disk, cache_dir)
    if cached is not None:
        return cached

    # Cache miss — run optimization
    # Pop nb_runs before passing merged_kwargs since it's passed explicitly
    merged_kwargs.pop("nb_runs", None)
    optimized: Module | Component = flowy_optimize(
        module, nb_runs=nb_runs, nb_workers=nb_workers,
        pareto_point=pareto_point, **merged_kwargs,
    )

    # Get AAG from the optimized module
    optimized_module: Module
    if isinstance(optimized, Component):
        optimized_module = optimized.to_module("optimized_comb")
    else:
        optimized_module = optimized

    aag_lines: List[str] = AigerExporter(optimized_module).get_aag()
    spec: Dict[str, HDLType] = optimized_module.get_spec()

    cache_metadata = {"base_cache_key": base_cache_key}
    _store_cache(cache_key_hex, aag_lines, spec, output_names,
                 write_mem, write_disk, cache_dir, metadata=cache_metadata)
    return aag_lines, spec, output_names


def _instantiate_from_cache(
    aag_lines: List[str],
    spec: Dict[str, HDLType],
    output_names: List[str],
    actual_logic_args: Dict[str, Expr],
) -> Union[Expr, Tuple[Expr, ...]]:
    """Create a fresh Component from cached AAG and wire actual Expr args.

    Parameters
    ----------
    aag_lines : list[str]
        Cached AIGER ASCII lines.
    spec : dict
        Port spec from the optimized module.
    output_names : list[str]
        Names of outputs (e.g. ["y"] or ["y0", "y1"]).
    actual_logic_args : dict
        {param_name: Expr} — the actual runtime Expr arguments.

    Returns
    -------
    Expr or tuple[Expr, ...]
    """
    # Build a Component with the right spec to use from_aag_lines
    io_sigs: Dict[str, Signal] = {}
    for name, typ in spec.items():
        if name in actual_logic_args:
            io_sigs[name] = Signal(name=name, typ=typ, kind="input")
        else:
            io_sigs[name] = Signal(name=name, typ=typ, kind="output")

    IO: type = make_dataclass("IO", [(name, Signal) for name in io_sigs])
    io: Any = IO(**io_sigs)

    class _CachedComponent(Component):
        def __init__(self, io_obj: Any) -> None:
            self.io = io_obj

        def elaborate(self) -> None:
            pass

    comp: Component = _CachedComponent(io)
    comp.from_aag_lines(aag_lines, group=True, make_internal=True)

    # Wire actual Expr args to the (now internal wire) inputs
    for name, expr in actual_logic_args.items():
        sig: Signal = getattr(comp.io, name)
        sig <<= expr

    # Return output(s)
    if len(output_names) == 1:
        return getattr(comp.io, output_names[0])
    else:
        return tuple(getattr(comp.io, oname) for oname in output_names)


# ---------------------------------------------------------------------------
# The decorator
# ---------------------------------------------------------------------------

def flowy_optimized(_fn: Callable[..., Any] | None = None, **kw: Any) -> Callable[..., Any]:
    """Decorator that automatically optimizes a function's logic through Flowy.

    Can be used with or without arguments::

        @flowy_optimized
        def my_adder(a, b):
            return a + b

        @flowy_optimized(nb_runs=100)
        def my_adder(a, b):
            return a + b

    At call time, Expr arguments are detected via ``isinstance(arg, Expr)``.
    The function is converted to a Component, optimized via ``flowy_optimize``,
    and the result is cached. Subsequent calls with the same argument types
    (same widths/signedness for logic args, same values for non-logic args)
    reuse the cached optimized circuit.

    Disk cache directory resolution (highest priority first):
      1. ``cache_dir`` argument to this decorator
      2. A cache folder next to the script defining the decorated function
         (only if it already exists)
      3. Global setting via ``set_cache_dir()`` or ``"cache_dir"`` in
         ``flowy_config.json`` (default: ``".spirehdl_cache"``)

    Cache read/write controls (``cache_read`` / ``cache_write``):
      Each takes one of ``"none"``, ``"mem"``, ``"disk"``, ``"both"``.
      Defaults are ``"both"`` for each — read+write both caches.
    """
    cache_read: CacheSpec = kw.pop("cache_read", "both")
    cache_write: CacheSpec = kw.pop("cache_write", "both")
    cache_dir: str | Path | None = kw.pop("cache_dir", None)
    # Pop multi-run and pareto params so they don't affect the base cache key
    nb_runs: int = kw.pop("nb_runs", 1)
    nb_workers: int = kw.pop("nb_workers", 10)
    pareto_point: int | None = kw.pop("pareto_point", None)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Bind arguments to parameter names
            fn_sig: inspect.Signature = inspect.signature(fn)
            bound: inspect.BoundArguments = fn_sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Separate logic (Expr) vs non-logic args
            logic_args: Dict[str, Tuple[int, bool]] = {}
            other_args: Dict[str, Any] = {}
            actual_logic_args: Dict[str, Expr] = {}

            for param_name, value in bound.arguments.items():
                param = fn_sig.parameters[param_name]
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    # Special Case - Handle *args (VAR_POSITIONAL): expand tuple into individual entries.
                    # Use letter suffixes (e.g. pref_vals_a, pref_vals_b) to avoid
                    # yosys merging numeric-suffixed ports into a single bus.
                    for i, v in enumerate(value):
                        varg_name = f"{param_name}_{chr(ord('a') + i)}"
                        if isinstance(v, Expr):
                            logic_args[varg_name] = (v.typ.width, v.typ.signed)
                            actual_logic_args[varg_name] = v
                        else:
                            other_args[varg_name] = v
                elif isinstance(value, Expr):
                    logic_args[param_name] = (value.typ.width, value.typ.signed)
                    actual_logic_args[param_name] = value
                else:
                    other_args[param_name] = value

            # If no logic args, just call the original function
            if not logic_args:
                return fn(*args, **kwargs)

            # Build component, check caches (mem + disk), optimize on miss
            aag_lines, spec, output_names = _optimize_and_cache(
                fn, logic_args, other_args, kw,
                cache_read=cache_read, cache_write=cache_write,
                explicit_cache_dir=cache_dir,
                nb_runs=nb_runs, nb_workers=nb_workers,
                pareto_point=pareto_point,
            )

            return _instantiate_from_cache(
                aag_lines, spec, output_names, actual_logic_args
            )

        return wrapper

    if _fn is not None:
        # Called as @flowy_optimized without parentheses
        return decorator(_fn)
    else:
        # Called as @flowy_optimized(...) with arguments
        return decorator


# ---------------------------------------------------------------------------
# ABC / DeepSyn optimization
# ---------------------------------------------------------------------------

def abc_optimize(
    m: Module | Component,
    abc_script: str = "strash; &get -n; &deepsyn -T 10; &put",
    suppress_stderr: bool = True,
) -> List[str]:
    """Run yosys synthesis + ABC script optimization and return optimized AAG lines.

    Parameters
    ----------
    m : Module or Component
        The design to optimize.
    abc_script : str
        ABC commands to execute (e.g. ``"strash; &get -n; &deepsyn -T 10; &put"``).
    suppress_stderr : bool
        Suppress yosys/ABC output.

    Returns
    -------
    list[str]
        Optimized AIGER ASCII lines.
    """
    from pyosys import libyosys as ys
    from spirehdl.aig.aig_aigerverse import file_to_lines
    from spirehdl.helpers import _suppress_output

    if isinstance(m, Component):
        m = m.to_module("mydesign_comb")

    verilog_content: str = m.to_verilog()

    fd_v, verilog_tmp = tempfile.mkstemp(suffix=".v")
    os.close(fd_v)
    fd_abc, abc_tmp = tempfile.mkstemp(suffix=".abc")
    os.close(fd_abc)
    fd_aag, aag_out_tmp = tempfile.mkstemp(suffix=".aag")
    os.close(fd_aag)

    try:
        with open(verilog_tmp, "w") as f:
            f.write(verilog_content)
        with open(abc_tmp, "w") as f:
            f.write(abc_script + "\n")

        with _suppress_output(stderr=suppress_stderr):
            ys.run_pass("design -reset")
            ys.run_pass(f"read_verilog -sv {verilog_tmp}")
            ys.run_pass("hierarchy -check -auto-top")
            ys.run_pass("proc; opt; fsm; memory; opt")
            ys.run_pass(f"abc -script {abc_tmp}")
            ys.run_pass("techmap; opt; abc -fast; opt")
            ys.run_pass("aigmap")
            ys.run_pass(f"write_aiger -ascii -symbols -no-startoffset {aag_out_tmp}")

        return file_to_lines(aag_out_tmp)
    finally:
        for p in (verilog_tmp, abc_tmp, aag_out_tmp):
            if os.path.exists(p):
                os.remove(p)


def abc_optimized(
    _fn: Callable[..., Any] | None = None,
    *,
    abc_script: str = "strash; &get -n; &deepsyn -T 10; &put",
    cache_read: CacheSpec = "both",
    cache_write: CacheSpec = "both",
    cache_dir: str | Path | None = None,
) -> Callable[..., Any]:
    """Decorator that optimizes a function's logic through ABC (via yosys/pyosys).

    Usage::

        @abc_optimized(abc_script="strash; &get -n; &deepsyn -T 30; &put")
        def my_mult(a, b):
            return a * b

    At call time, Expr arguments are detected and the function is converted to
    a Component, optimized via ``abc_optimize``, and the result is cached.

    Cache read/write controls (``cache_read`` / ``cache_write``):
      Each takes one of ``"none"``, ``"mem"``, ``"disk"``, ``"both"``.
      Defaults are ``"both"`` for each — read+write both caches.
    """
    from spirehdl.spirehdl_aiger import AigerExporter, AigerImporter
    from spirehdl.spirehdl_module import IOCollector

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            fn_sig: inspect.Signature = inspect.signature(fn)
            bound: inspect.BoundArguments = fn_sig.bind(*args, **kwargs)
            bound.apply_defaults()

            logic_args: Dict[str, Tuple[int, bool]] = {}
            other_args: Dict[str, Any] = {}
            actual_logic_args: Dict[str, Expr] = {}

            for param_name, value in bound.arguments.items():
                param = fn_sig.parameters[param_name]
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    for i, v in enumerate(value):
                        varg_name = f"{param_name}_{chr(ord('a') + i)}"
                        if isinstance(v, Expr):
                            logic_args[varg_name] = (v.typ.width, v.typ.signed)
                            actual_logic_args[varg_name] = v
                        else:
                            other_args[varg_name] = v
                elif isinstance(value, Expr):
                    logic_args[param_name] = (value.typ.width, value.typ.signed)
                    actual_logic_args[param_name] = value
                else:
                    other_args[param_name] = value

            if not logic_args:
                return fn(*args, **kwargs)

            # Build component and module
            comp, output_names = _build_component(fn, logic_args, other_args)
            module: Module = comp.to_module("mydesign_comb")
            original_spec: Dict[str, HDLType] = module.get_spec()

            read_mem, read_disk, write_mem, write_disk = _resolve_rw_flags(
                cache_read, cache_write,
            )

            # Cache key includes the abc script
            optimize_kwargs = {"abc_script": abc_script}
            resolved_cache_dir: Path | None = (
                _resolve_cache_dir(fn, cache_dir) if (read_disk or write_disk) else None
            )
            verilog_content: str = module.to_verilog()
            cache_key_hex: str = _compute_disk_cache_key(
                verilog_content, other_args, optimize_kwargs,
            )

            cached = _cache_lookup(
                cache_key_hex, read_mem, read_disk, resolved_cache_dir,
            )
            if cached is not None:
                return _instantiate_from_cache(
                    cached[0], cached[1], cached[2], actual_logic_args,
                )

            # Cache miss — optimize
            raw_aag: List[str] = abc_optimize(module, abc_script)

            # Regroup bit-blasted ports to match original widths
            opt_module: Module = AigerImporter(raw_aag).get_sprout_module()
            IOCollector().group(opt_module, original_spec)
            aag_lines: List[str] = AigerExporter(opt_module).get_aag()
            spec: Dict[str, HDLType] = opt_module.get_spec()

            _store_cache(
                cache_key_hex, aag_lines, spec, output_names,
                write_mem, write_disk, resolved_cache_dir,
            )

            return _instantiate_from_cache(
                aag_lines, spec, output_names, actual_logic_args,
            )

        return wrapper

    if _fn is not None:
        return decorator(_fn)
    else:
        return decorator


# ---------------------------------------------------------------------------
# Arithmetic replacement optimization
# ---------------------------------------------------------------------------

def arithmetic_optimized(
    _fn: Callable[..., Any] | None = None,
    *,
    objective: Literal["area", "delay", "adp"] = "area",
) -> Callable[..., Any]:
    """Decorator that rewrites a function's ``+``, ``-``, ``*`` operators
    with optimized StageBased hardware via ``replace_arithmetic_ops``.

    Usage::

        @arithmetic_optimized(objective="adp")
        def opt_mac(a, b, c):
            return a * b + c

    At call time, ``Expr`` arguments are detected and the function is
    converted into a ``Component`` with placeholder inputs.  Every ``+``,
    ``-``, ``*`` and equality operator in the resulting expression graph is
    replaced by the empirically best StageBased adder / subtractor /
    multiplier for its specific bit-width and signedness, with MAC /
    inner-product fusion applied where applicable.  The optimized sub-graph
    is then spliced into the caller's design: the function returns ``Expr``
    values that can be used exactly like ``fn``'s original outputs.

    Unlike ``@abc_optimized`` / ``@flowy_optimized``, no AIG flattening and no
    disk cache are involved -- the replacement is a fast database lookup that
    produces structured Verilog directly.

    Parameters
    ----------
    objective : {"area", "delay", "adp"}
        Optimization target:

        - ``"area"``:  minimize Yosys transistor count
        - ``"delay"``: minimize AIG depth (proxy for critical-path delay)
        - ``"adp"``:   minimize area-delay product
    """
    from spirehdl.arithmetic.int_arithmetic_config import (
        ArithmeticAutoConfig,
        replace_arithmetic_ops,
    )

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            fn_sig: inspect.Signature = inspect.signature(fn)
            bound: inspect.BoundArguments = fn_sig.bind(*args, **kwargs)
            bound.apply_defaults()

            logic_args: Dict[str, Tuple[int, bool]] = {}
            other_args: Dict[str, Any] = {}
            actual_logic_args: Dict[str, Expr] = {}

            for param_name, value in bound.arguments.items():
                param = fn_sig.parameters[param_name]
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    for i, v in enumerate(value):
                        varg_name = f"{param_name}_{chr(ord('a') + i)}"
                        if isinstance(v, Expr):
                            logic_args[varg_name] = (v.typ.width, v.typ.signed)
                            actual_logic_args[varg_name] = v
                        else:
                            other_args[varg_name] = v
                elif isinstance(value, Expr):
                    logic_args[param_name] = (value.typ.width, value.typ.signed)
                    actual_logic_args[param_name] = value
                else:
                    other_args[param_name] = value

            if not logic_args:
                return fn(*args, **kwargs)

            # Build a Component wrapping fn()'s graph with placeholder inputs.
            comp, output_names = _build_component(fn, logic_args, other_args)

            # Rewrite +, -, * (and ==, !=) with optimized StageBased components.
            replace_arithmetic_ops(
                comp,
                ArithmeticAutoConfig(objective=objective),
            )

            # Splice into the caller's graph: turn comp.io ports into internal
            # wires, drive the input wires from the caller's actual Expr args,
            # and return the output wires as the optimized expressions.
            comp.make_internal()
            for name, expr in actual_logic_args.items():
                sig: Signal = getattr(comp.io, name)
                sig <<= expr

            if len(output_names) == 1:
                return getattr(comp.io, output_names[0])
            return tuple(getattr(comp.io, oname) for oname in output_names)

        return wrapper

    if _fn is not None:
        return decorator(_fn)
    return decorator
