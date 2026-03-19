"""
Decorator and utilities for automated Flowy circuit optimization.

Usage:
    from sprouthdl.optimize import flowy_optimized

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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from sprouthdl.sprouthdl import Expr, HDLType, Signal
from sprouthdl.sprouthdl_module import Component, Module


# ---------------------------------------------------------------------------
# Cache: maps verilog hash -> (aag_lines, spec, output_names)
# ---------------------------------------------------------------------------
_cache: Dict[str, Tuple[List[str], Dict[str, HDLType], List[str]]] = {}

# Persistent disk cache directory. Set to None to disable disk caching.
_DISK_CACHE_DIR: Path | None = Path(".sprouthdl_cache")
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
            print(f"[sprouthdl] Loaded flowy config from {cfg}")
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
         ``flowy_config.json``; defaults to ``".sprouthdl_cache"``)
    """
    # 1. Explicit override from decorator
    if explicit_dir is not None:
        return Path(explicit_dir)
    # 2. Cache folder next to the script defining fn
    if fn is not None:
        try:
            script_dir = Path(inspect.getfile(fn)).resolve().parent
            local_cache = script_dir / (_DISK_CACHE_DIR.name if _DISK_CACHE_DIR else ".sprouthdl_cache")
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
                   nb_runs: int = 50,
                   nb_workers: int = 10,
                   iterations: int = 1,
                   mockturtle_chains: int = 1,
                   mockturtle_chain_len: int = 10,
                   mockturtle_chain_workers: int = 1,
                   selection_metric: str | None = None,
                   verbose: bool = False,
                   direct: bool = False,
                   visualize: bool = False) -> Module | Component:
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
        # Call the statistical run_flow directly — single process, no Docker
        from flowy.flows.reinforce.run.statistical.run_flow import (
            run_flow as statistical_run_flow,
        )
        statistical_run_flow(
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

    best_design = RunIdentifier(
        root_database="output/db", experiment=experiment,
        stage="analysis", run="best_designs",
    )
    best_design_item = RunDatabase(best_design).load(
        f"final_mockturtle_design_best_design_{selection_metric.value}"
    )
    aig_file_path: str = best_design_item.get("aiger_filepath").path
    aiger_map_file_path: str = best_design_item.get("aiger_map_filepath").path

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
    # Save and reset the shared wire state so the sub-circuit gets
    # deterministic signal names (sig_0, sig_1, …) regardless of what
    # signals the caller has already created.  This makes the Verilog
    # content — and therefore the disk-cache key — independent of
    # context, which is critical for cache hits across different
    # multiplier / adder configurations.
    from sprouthdl.sprouthdl import _SHARED
    saved_counts  = dict(_SHARED.counts)
    saved_e2s     = dict(_SHARED.expr2sig)
    saved_wires   = list(_SHARED.wires)
    saved_index   = _SHARED.index
    _SHARED.counts.clear()
    _SHARED.expr2sig.clear()
    _SHARED.wires.clear()
    _SHARED.index = 0

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

    # Restore the caller's shared wire state so the outer circuit is
    # unaffected by the temporary sub-circuit we just built.
    _SHARED.counts.clear()
    _SHARED.counts.update(saved_counts)
    _SHARED.expr2sig.clear()
    _SHARED.expr2sig.update(saved_e2s)
    _SHARED.wires.clear()
    _SHARED.wires.extend(saved_wires)
    _SHARED.index = saved_index

    return comp, output_names


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
            print(f"[sprouthdl] Disk cache hit: {cache_key_hex[:12]}...")
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
) -> None:
    """Store result in in-memory and/or disk caches."""
    if use_mem_cache:
        _cache[cache_key_hex] = (aag_lines, spec, output_names)
    if use_disk_cache:
        _write_cache_entry(cache_key_hex, aag_lines, spec, output_names, cache_dir)
        print(f"[sprouthdl] Cached optimization result: {cache_key_hex[:12]}...")


def _optimize_and_cache(
    fn: Callable[..., Any],
    logic_args: Dict[str, Tuple[int, bool]],
    other_args: Dict[str, Any],
    optimize_kwargs: Dict[str, Any],
    use_mem_cache: bool = True,
    use_disk_cache: bool = True,
    explicit_cache_dir: str | Path | None = None,
) -> Tuple[List[str], Dict[str, HDLType], List[str]]:
    """Build component, check caches, optimize if needed, cache the AAG lines.

    Returns
    -------
    tuple[list[str], dict[str, HDLType], list[str]]
        (aag_lines, spec, output_names)
    """
    from sprouthdl.sprouthdl_aiger import AigerExporter

    comp, output_names = _build_component(fn, logic_args, other_args)
    module: Module = comp.to_module("mydesign_comb")

    merged_kwargs = {**_DEFAULT_OPTIMIZE_KWARGS, **optimize_kwargs}

    # Resolve cache directory (see _resolve_cache_dir for priority)
    cache_dir: Path | None = _resolve_cache_dir(fn, explicit_cache_dir) if use_disk_cache else None

    # Compute cache key from verilog content
    verilog_content: str = module.to_verilog()
    cache_key_hex: str = _compute_disk_cache_key(verilog_content, other_args, merged_kwargs)

    # Check caches before running expensive optimization
    cached = _cache_lookup(cache_key_hex, use_mem_cache, use_disk_cache, cache_dir)
    if cached is not None:
        return cached

    # Cache miss — run optimization
    optimized: Module | Component = flowy_optimize(module, **merged_kwargs)

    # Get AAG from the optimized module
    optimized_module: Module
    if isinstance(optimized, Component):
        optimized_module = optimized.to_module("optimized_comb")
    else:
        optimized_module = optimized

    aag_lines: List[str] = AigerExporter(optimized_module).get_aag()
    spec: Dict[str, HDLType] = optimized_module.get_spec()

    _store_cache(cache_key_hex, aag_lines, spec, output_names,
                 use_mem_cache, use_disk_cache, cache_dir)
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
         ``flowy_config.json`` (default: ``".sprouthdl_cache"``)
    """
    use_mem_cache: bool = kw.pop("use_mem_cache", True)
    use_disk_cache: bool = kw.pop("use_disk_cache", True)
    cache_dir: str | Path | None = kw.pop("cache_dir", None)

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
                use_mem_cache=use_mem_cache, use_disk_cache=use_disk_cache,
                explicit_cache_dir=cache_dir,
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
