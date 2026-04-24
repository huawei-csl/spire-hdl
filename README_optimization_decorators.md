# Circuit Optimization Decorators

Sprout-HDL provides two decorators that optimize combinational logic at the AIG level.  Decorate any Python function that builds logic from `Expr` arguments, and the framework automatically converts it to a circuit, runs the chosen optimizer, caches the result, and splices the optimized logic back into your design.

## `@abc_optimized` -- ABC / DeepSyn via Yosys

Runs an [ABC](https://github.com/berkeley-abc/abc) script through Yosys/PyOsys.  The script is passed as a string argument.

```python
from sprouthdl.optimize import abc_optimized

@abc_optimized(abc_script="strash; &get -n; &deepsyn -T 10; &put")
def my_mult(a, b):
    return a * b

# Use like any other function -- Expr in, Expr out:
result = my_mult(signal_a, signal_b)
```

### ABC script examples

| Script | Description |
|--------|-------------|
| `strash; balance; rewrite -l; refactor -l; balance; rewrite -l; rewrite -lz; balance; refactor -lz; rewrite -lz; balance` | Classic `resyn2` -- fast, no timeout |
| `strash; &get -n; &deepsyn -T 10; &put` | DeepSyn with 10 s budget |
| `strash; &get -n; &deepsyn -T 60; &put` | DeepSyn with 60 s budget |
| `strash; &get -n; &deepsyn -J 5 -T 30; &put` | DeepSyn, 5 random restarts, 30 s |
| `strash; dc2; scorr; strash; dch -f; if -K 6; mfs2` | Technology-oriented flow |

### Benchmark results

Gate counts (AIG AND-gates) before and after optimization:

| Circuit | Original | resyn2 | deepsyn -T 5 | deepsyn -T 30 |
|---------|----------|--------|--------------|---------------|
| 8-bit multiplier | 1,824 | 569 (-69%) | 607 (-67%) | 601 (-67%) |
| 16-bit multiplier | 15,552 | 2,620 (-83%) | 2,621 (-83%) | 2,621 (-83%) |
| 8-bit adder | 67 | 79 (+18%) | 79 (+18%) | 79 (+18%) |

Complex arithmetic (multipliers) sees large reductions because the starting AIG from Sprout's expression tree is deliberately naive.  Simple circuits like adders may not benefit (or even grow slightly) due to overhead in the Yosys synthesis pipeline.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `abc_script` | `"strash; &get -n; &deepsyn -T 10; &put"` | ABC commands to run |
| `cache_read` | `"both"` | Which caches to consult: `"none"` / `"mem"` / `"disk"` / `"both"` |
| `cache_write` | `"both"` | Which caches to populate: same values |
| `cache_dir` | `None` | Override cache directory |

### Lower-level function

`abc_optimize(module, abc_script)` takes a `Module` or `Component` and returns optimized AAG lines directly, without the decorator/caching machinery.

```python
from sprouthdl.optimize import abc_optimize

aag_lines = abc_optimize(my_module, "strash; &get -n; &deepsyn -T 30; &put")
```

---

## `@flowy_optimized` -- RL-based optimization via Flowy

Uses the [Flowy](https://github.com/lsils/flowy) framework for reinforcement-learning-guided circuit optimization with MockTurtle.  Supports multi-run optimization and Pareto-front design selection.

```python
from sprouthdl.optimize import flowy_optimized

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
from sprouthdl.optimize import flowy_optimize

optimized_module = flowy_optimize(my_module, nb_runs=10, direct=True)
```

---

## Caching

Both decorators above share the same two-level cache:

1. **In-memory** -- keyed by a SHA-256 hash of the Verilog content + non-logic arguments + optimizer parameters.  Instant on repeated calls within the same process.
2. **Disk** -- stored in `.sprouthdl_cache/v1/` as JSON files containing the optimized AAG lines and port spec.  Survives across runs.

Use `clear_optimization_cache()` to reset the in-memory cache, or delete `.sprouthdl_cache/` for the disk cache.

```python
from sprouthdl.optimize import set_cache_dir, clear_optimization_cache

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

For local reuse of a small arithmetic block, the `@arithmetic_optimized` decorator offers the same one-liner ergonomics as `@abc_optimized` / `@flowy_optimized` but without going through an external synthesizer — the body of the decorated function is wrapped into a `Component`, `replace_arithmetic_ops` is run on it, and the optimized sub-graph is spliced back into the caller's design:

```python
from sprouthdl.sprouthdl import UInt
from sprouthdl.sprouthdl_module import Module
from sprouthdl.optimize import arithmetic_optimized

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

### Adding `@flowy_optimized` on top

`@flowy_optimized` can be stacked as the *outermost* decorator above `@abc_optimized` + `@arithmetic_optimized` to add stochastic mockturtle search on top of the realatively deterministic pipeline — which can occasionally break below the floor at the cost of higher variance:

```python
@flowy_optimized(
    direct=True, iterations=1,
    mockturtle_chains=10, mockturtle_chain_len=20, mockturtle_chain_workers=10,
    nb_runs=1, selection_metric="aig_count",
    cache_read="none",  # each call explores a fresh point
)
@abc_optimized(
    abc_script="strash; &get -n; &deepsyn -T 30; &put", cache_read="none",
)
@arithmetic_optimized(objective="area")
def opt_mult(a, b):
    return a * b
```
