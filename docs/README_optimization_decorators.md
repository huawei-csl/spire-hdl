# Circuit Optimization Decorators

Spire provides two decorators that optimize combinational logic at the AIG level.  Decorate any Python function that builds logic from `Expr` arguments, and the framework automatically converts it to a circuit, runs the chosen optimizer, caches the result, and splices the optimized logic back into your design.

## `@abc_optimized` -- ABC via Yosys

Runs an [ABC](https://github.com/berkeley-abc/abc) script on the gate-level AIG.  The
design is first lowered to gates by a fixed *prep* leg (`techmap; opt; abc -fast; opt`),
then your `abc_script` runs on that AIG via a standalone `abc` subprocess.  This ordering
matters: ABC has no rewriter for coarse cells like `$mul`/`$add`, so the script must run
*after* `techmap` to have any effect.  (Earlier versions ran the script before `techmap`,
where it was essentially inert.)

```python
from spire.optimize import abc_optimized, ABC_RECIPES

@abc_optimized                                    # bare -> ABC_RECIPES["balanced"]
def my_mult(a, b):
    return a * b

# Use like any other function -- Expr in, Expr out:
result = my_mult(signal_a, signal_b)
```

### Recipes

`ABC_RECIPES` is a small dict of curated scripts.  Reference an entry directly on the
`abc_script` argument, or pass any raw ABC string of your own.

```python
@abc_optimized(abc_script=ABC_RECIPES["area"])    # smallest transistor count
def small_mult(a, b):
    return a * b
```

| Recipe | Script | Strength | Wall-clock |
|--------|--------|----------|-----------|
| `"area"` | `strash; &get -n; &deepsyn -T 10; &put` | smallest transistor/gate count; trades depth for area | ~5–10 s |
| **`"balanced"`** (default) | `strash; dch -f; balance` | best all-rounder — strong on depth and area together | ~40 ms |
| `"depth"` | `strash; balance; rewrite; balance; refactor; balance` | gentle structure; favours depth where `dch` over-restructures | ~0.1 s |

### Raw ABC script examples

Any ABC string works on the `abc_script` argument:

| Script | Description |
|--------|-------------|
| `strash; balance; rewrite -l; refactor -l; balance; rewrite -l; rewrite -lz; balance; refactor -lz; rewrite -lz; balance` | Classic `resyn2` |
| `strash; &get -n; &deepsyn -T 60; &put` | DeepSyn with 60 s budget |
| `strash; &get -n; &deepsyn -J 5 -T 30; &put` | DeepSyn, 5 random restarts, 30 s |
| `strash; dc2; scorr; strash; dch -f; if -K 6; mfs2` | Technology-oriented flow |

### Benchmark results

Estimated transistors (`core.cost.YosysTransistorCost`) for the un-optimized design
(no decorator) vs. each recipe, plus raw `resyn2` and `transtoch` scripts (bold = best per row):

| Circuit | Original (no opt) | `balanced` (default) | `area` | `depth` | raw `resyn2` | raw `transtoch`¹ |
|---------|------------------:|---------------------:|-------:|--------:|-------------:|-----------------:|
| 4-bit multiplier  |    508 |    478 |    474 |    498 |    490 | **434** (−14.6%) |
| 8-bit multiplier  |  2,688 |  2,642 | **2,468** (−8.2%) |  2,640 |  2,576 | timeout |
| 16-bit multiplier | 11,966 | 11,830 | **11,170** (−6.7%) | 11,874 | 11,578 | timeout |
| 8×8→17 MAC        |  3,596 |  3,320 | **3,122** (−13.2%) |  3,474 |  3,242 | timeout |
| 8-bit adder       |    360 |    338 |    308 |    350 | **296** (−17.8%) |    308 |

¹ `&get -n; &transtoch -o -N 16 -M 2 -P 16; &put` — transduction (similar to `&deepsyn`).
Strong on small designs (best on the 4-bit multiplier) but scales poorly: multipliers ≥ 8 bits
time out past 120 s, so raise `timeout=` to try them.

`area` wins on the larger multipliers / MAC, `resyn2` on the adder, `transtoch` on the
small multiplier — worth trying a couple of scripts per design. `area`/`transtoch` use
randomized search, so expect small run-to-run variance.

The same designs by **AIG node count** (`core.cost.AigCountCost`):

| Circuit | Original (no opt) | `balanced` | `area` | `depth` | `resyn2` | `transtoch` |
|---------|------------------:|-----------:|-------:|--------:|---------:|------------:|
| 4-bit multiplier  |    91 |    91 |    91 |    91 |    90 | **84** |
| 8-bit multiplier  |   466 |   461 | **446** |   460 |   457 | timeout |
| 16-bit multiplier | 2,051 | 2,048 | **1,978** | 2,011 | 2,013 | timeout |
| 8×8→17 MAC        |   624 |   596 | **573** |   589 |   591 | timeout |
| 8-bit adder       |    69 |    72 |    69 |    71 |    69 |    69 |

And by **AIG logic depth** (`core.cost.AigDepthCost`):

| Circuit | Original (no opt) | `balanced` | `area` | `depth` | `resyn2` | `transtoch` |
|---------|------------------:|-----------:|-------:|--------:|---------:|------------:|
| 4-bit multiplier  | **17** | 17 | 17 | 17 | 17 | 18 |
| 8-bit multiplier  | **32** | 35 | 36 | 39 | 36 | timeout |
| 16-bit multiplier |    53 | **51** | 66 | 71 | 72 | timeout |
| 8×8→17 MAC        | **34** | 42 | 43 | 43 | 43 | timeout |
| 8-bit adder       |    16 | **15** | 16 | 17 | 16 | 16 |

The gate-leaning recipes (`area`/`resyn2`) shrink the AIG at the cost of depth — the
plain-synthesis baseline is already depth-optimized, so it usually has the lowest depth.
`balanced` is the exception that also reduces depth (it beats Original on the 16-bit
multiplier and the adder).

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `abc_script` | `ABC_RECIPES["balanced"]` (`"strash; dch -f; balance"`) | ABC commands to run on the gate-level AIG. Raw string or an `ABC_RECIPES` entry |
| `prep_script` | `"techmap; opt; abc -fast; opt"` | Yosys passes that lower coarse cells to gates before ABC. Rarely changed |
| `timeout` | `120` | Wall-clock seconds for the ABC subprocess. Raise for long scripts (e.g. `&deepsyn`/`&transtoch` with big budgets); `None` waits indefinitely |
| `cache_read` | `"both"` | Which caches to consult: `"none"` / `"mem"` / `"disk"` / `"both"` |
| `cache_write` | `"both"` | Which caches to populate: same values |
| `cache_dir` | `None` | Override cache directory |

> A standalone `abc` binary enables the effective out-of-process optimization. Discovery
> order: `$SPIRE_ABC`, then `abc` on `PATH`, then `yosys-abc`. If none is found,
> `abc_optimize` automatically falls back to an in-process pyosys path (legacy ordering,
> largely inert on coarse cells) and emits a `RuntimeWarning` — so it still runs on a
> pure-pyosys install, just without the gains.

### Lower-level function

`abc_optimize(module, abc_script=..., prep_script=...)` takes a `Module` or `Component`
and returns optimized AAG lines directly, without the decorator/caching machinery.

```python
from spire.optimize import abc_optimize, ABC_RECIPES

aag_lines = abc_optimize(my_module, abc_script=ABC_RECIPES["area"])
```

### Iterative optimization (nested + cached)

`@abc_optimized` can be stacked on top of itself (or another optimization decorator).
Each layer optimizes the output of the layer beneath it, so adding another decorator and
re-running continues from the already-optimized circuit instead of starting over.  Every
layer caches its result (keyed by its input Verilog + script), so on a re-run the inner
layers are cache hits and only the new outer pass does work — letting you push a design
further across successive runs without recomputing what's already done.

This lets you explore a *tree* of optimization sequences — branch off good results with
different follow-up scripts — efficiently, since shared prefixes are computed once
(prefix memoization, i.e. beam/tree search over synthesis flows).

```python
@abc_optimized(abc_script=ABC_RECIPES["area"])                       # 2nd pass: refines the 1st
@abc_optimized(abc_script="strash; &get -n; &deepsyn -T 30; &put")   # 1st pass: cached after first run
def my_mult(a, b):
    return a * b
```

---

## `@flowy_optimized` Decorator

Uses the Flowy with MockTurtle. Supports multi-run optimization and Pareto-front design selection.

```python
from spire.optimize import flowy_optimized

@flowy_optimized(direct=True, iterations=1, mockturtle_chains=1,
                 mockturtle_chain_len=2, mockturtle_chain_workers=1)
def optimized_mult(a, b):
    return a * b
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nb_runs` | `1` | Number of parallel optimization runs |
| `nb_workers` | `10` | Parallel workers for multi-run |
| `iterations` | `1` | MockTurtle iterations per run |
| `mockturtle_chains` | `1` | Number of MockTurtle chains |
| `mockturtle_chain_len` | `10` | Chain length |
| `direct` | `False` | `True` = local execution, `False` = Docker |
| `pareto_point` | `None` | Select a specific Pareto-front design (0 = best area) |
| `cache_read` | `"both"` | Which caches to consult: `"none"` / `"mem"` / `"disk"` / `"both"` |
| `cache_write` | `"both"` | Which caches to populate: same values |

### Lower-level function

```python
from spire.optimize import flowy_optimize

optimized_module = flowy_optimize(my_module, nb_runs=10, direct=True)
```

---

## Caching

Both decorators above share the same two-level cache:

1. **In-memory** -- keyed by a SHA-256 hash of the Verilog content + non-logic arguments + optimizer parameters.  Instant on repeated calls within the same process.
2. **Disk** -- stored in `.spire_cache/v1/` as JSON files containing the optimized AAG lines and port spec.  Survives across runs.

Use `clear_optimization_cache()` to reset the in-memory cache, or delete `.spire_cache/` for the disk cache.

```python
from spire.optimize import set_cache_dir, clear_optimization_cache

set_cache_dir("/my/cache/path")   # override default location
clear_optimization_cache()         # clear in-memory cache
```

Reads and writes are independently gated by `cache_read` and `cache_write`.  Each takes one of `"none"`, `"mem"`, `"disk"`, `"both"`.

| Intent | `cache_read` | `cache_write` |
|---|---|---|
| Read + write both caches (default) | `"both"` | `"both"` |
| Force recompile, but populate both caches for later | `"none"` | `"both"` |
| Mem in-process, never persist | `"mem"` | `"mem"` |
| Use disk cache, don't pollute it | `"both"` | `"mem"` |
| Force recompile, populate disk only | `"none"` | `"disk"` |
| Fully bypass cache | `"none"` | `"none"` |

---

## `@arithmetic_optimized` -- Structural arithmetic replacement

For local reuse of a small arithmetic block, the `@arithmetic_optimized` decorator offers the same one-liner ergonomics as `@abc_optimized` but without going through an external synthesizer — the body of the decorated function is wrapped into a `Component`, `replace_arithmetic_ops` is run on it, and the optimized sub-graph is spliced back into the caller's design:

```python
from spire.expr import UInt
from spire.component import Module
from spire.optimize import arithmetic_optimized

@arithmetic_optimized(objective="adp")
def opt_mac(a, b, c):
    return a * b + c

m = Module("Top", with_clock=False, with_reset=False)
a = m.input(UInt(8), "a")
b = m.input(UInt(8), "b")
c = m.input(UInt(16), "c")
y = m.output(UInt(17), "y")
y <<= opt_mac(a, b, c)    # MAC fusion happens inside the decorator
print(m.to_verilog())
```

MAC / inner-product fusion, bit-width-aware configuration lookup, and `==`/`!=` lowering all work the same as with `replace_arithmetic_ops` on a hand-built component, because the decorator just calls `replace_arithmetic_ops` under the hood.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `objective` | `"area"` | `"area"` / `"delay"` / `"adp"` — see [README_arithmetic_optimization.md](README_arithmetic_optimization.md) for details |

---

## Stacking decorators

The optimizations are complementary and may be nested.  The easiest way is to stack the decorators directly (other combinations — e.g. running `replace_arithmetic_ops` on a hand-built component and then feeding the result through `abc_optimize` — work too):

```python
@abc_optimized(abc_script="strash; balance; rewrite -l; refactor -l; balance")
@arithmetic_optimized(objective="area")
def opt_mac(a, b, c):
    return a * b + c
```

On an 8×8 → 17-bit MAC, `@arithmetic_optimized` alone lands at 682 AIG gates and `@abc_optimized` alone at 700; stacking both drops to 611 — better than either on its own, because the arithmetic rewriter picks good structural blocks and ABC then cleans up the flattened AIG.

**Order matters.** `@abc_optimized` has to be the *outer* decorator: the inner one must run first so that `+`/`-`/`*`/`==`/`!=` operators still exist to be matched against the arithmetic configuration database.  Once ABC has flattened the design to an AIG, there is nothing left for the arithmetic rewriter to recognize.
