# Circuit Optimization Decorators

Spire provides decorators that optimize combinational logic.  Decorate any Python function that builds logic from `Expr` arguments, and the framework automatically converts it to a circuit, runs the chosen optimizer, caches the result, and splices the optimized logic back into your design.

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

### What to decorate

This section is about **technology-mapped metrics** -- ASAP7 (or similar) area
and delay from synthesis + STA, as produced by the PPA flow.  Against the AIG
proxies used in the benchmark tables below, scope barely matters and the
decorator nearly always looks like a win; the trade-off only appears once the
design is mapped to real cells under a timing constraint.

ABC has no cell library, no timing model and no target delay: it minimizes the
AIG, trading mapped delay for area wherever you apply it.  What decides whether
that trade pays is **the structure you wrap**, not the metric you are
optimizing.

**Prefer small, heavily-instantiated, register-bounded blocks.**  One and the
same leaf helper, same recipe, measured in two surrounding structures:

| where the block sits | seeds worse than not decorating | median result |
|---|---|---|
| inside a combinational datapath | **85 / 100** | +2.3% (worse) |
| in a pipeline stage, flops either side | **2 / 100** | **-6.0%** (better) |

A register boundary turns a target that usually hurts into one that almost
always helps.  Two properties predict payoff:

* **replication** -- a block instantiated N times has its improvement applied N
  times, and a small block gets far more ABC search per gate than a whole-design
  pass can afford (~200x at 16 seeds).
* **pass-through** -- over a handful of seeds, compare the block's own area swing
  with the full design's.  If the full swing is roughly *block swing x
  instances*, gains transfer.  If it is many times larger, the surrounding logic
  is being re-mapped and the block's own numbers predict nothing.

**Deprioritize -- but never skip -- regions dominated by wide word-level
arithmetic.**  The decorator lowers its region to gates *before* the main
synthesis, so a generic techmap can bake carry chains that later AIG rewriting
never rediscovers as fast adders (on one measured datapath a *no-op* round trip
cost 11% delay from this alone -- premature lowering, not the region being
"frozen").  Start with other candidates -- small self-contained helpers,
replicated blocks, register-bounded stages -- but if arithmetic-heavy blocks are
all a design offers, decorate them anyway with a seed sweep rather than leaving
the decorator untried.

Scopes do not compose predictably, and more is not better: decorating four separate
sub-blocks at once measured **worse on area than the best single one**.  Prefer one
well-chosen, well-seeded block over several.

**Mind the critical path** (STA reports it per evaluation).  Wrapping blocks *off*
the path cannot improve delay by construction -- it only buys area, safely, since
slack absorbs the lowering tax.  To move delay the wrapped region must contain the
path, which also puts the tax on it: expect seed 0 to lose there, and judge only
after a seed sweep.

This split is exactly why the decorator suits **area x delay (ADP)** objectives:
it is the one mechanism that gives different regions different synthesis
objectives -- an area-minimizing sweep off the path (area bought at zero delay
cost, a pure ADP win) while the main timing-driven synthesis keeps the path
fast.  Uniform whole-design synthesis cannot express that trade.

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
| `"area"` | `strash; &get -n; &deepsyn -T 120; &put` | smallest transistor/gate count; trades depth for area | ~120 s |
| **`"balanced"`** (default) | `strash; dch -f; balance; <resyn2>; &get -n; &deepsyn -T 110; &put` | best all-rounder — strong on depth and area together | ~115 s |
| `"depth"` | `strash; <resyn2>; <resyn2>; &get -n; &deepsyn -T 110; &put` | gentle structure; favours depth where `dch` over-restructures | ~115 s |

### Raw ABC script examples

Any ABC string works on the `abc_script` argument:

| Script | Description |
|--------|-------------|
| `strash; balance; rewrite -l; refactor -l; balance; rewrite -l; rewrite -lz; balance; refactor -lz; rewrite -lz; balance` | Classic `resyn2` |
| `strash; &get -n; &deepsyn -T 60; &put` | DeepSyn with 60 s budget |
| `strash; &get -n; &deepsyn -J 5 -T 30; &put` | DeepSyn, 5 random restarts, 30 s |
| `strash; &get -n; &deepsyn -T 120 -S 7; &put` | Same as `ABC_RECIPES["area"]`, seed 7 |
| `strash; dc2; scorr; strash; dch -f; if -K 6; mfs2` | Technology-oriented flow |

### Seeding: one decorator run is a coin flip

`&deepsyn` takes `-S <0..100>`, a random seed (default 0).  None of the
`ABC_RECIPES` sets it, so every unseeded application explores the **same single
trajectory** -- and a single trajectory is not reliably an improvement.  Measured
on one large datapath over 100 sampled seeds, best-of-K against the
*undecorated* design (ADP):

| K seeds | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| expected best | -2.7 % | -6.1 % | -7.7 % | **-9.4 %** | -11.0 % |

**22 of 100 seeds were worse than not decorating at all.**  So the useful move is
to try a few seeds and keep whichever measures best, rather than applying the
decorator once and trusting it:

```python
# same recipe, different trajectories -- evaluate each, keep the winner
@abc_optimized(abc_script="strash; &get -n; &deepsyn -T 120 -S 3; &put")
def backend(a, b, c): ...
```

Two things make this cheap.  The script text is part of the cache key, so **each
seed is its own cache entry** -- re-trying a seed you have already run costs
nothing, and a seed sweep is resumable.  And the payoff is concentrated: going
from one seed to four roughly doubles the gain for three extra evaluations.

### Do not give up after one attempt

Two independent noise sources make a single measurement nearly worthless as
evidence:

* **The search is randomized.**  One unseeded trajectory is a coin flip (22 of
  100 worse than not decorating); best-of-16 delivers roughly 3.5x the expected
  gain of best-of-1.
* **The measurement is noisy.**  On provably identical netlists -- same logic,
  only the internal wire names permuted -- mapped delay moved **6-8%** and area
  **1-3%**.  A lone result showing the decorator "lost" a few percent, especially
  on delay, may be pure measurement noise rather than a real regression.

So a bad first attempt is the point to try **more seeds** and a **second recipe**
(`area` / `balanced` / `depth`, or a raw script), not the point to abandon the
decorator.  Keep the best measured result.

**Budget at least four decorated evaluations per design, whatever you are
optimizing** -- three or four different `-S` seeds, plus a second scope or recipe
if the first disappoints.  Giving up after one application is the most common way
to leave real gains unclaimed, and a metric with no recorded wins is usually a
metric nobody has yet attacked with seeds -- not a dead end.

Seeding pays most where the decorated region is **small and instantiated many
times**: one good solution improves every instance at once, and a small region
gets far more ABC search per gate than a whole-design pass can afford.

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
| `abc_script` | `ABC_RECIPES["balanced"]` (dch + resyn2 + `&deepsyn -T 110`) | ABC commands to run on the gate-level AIG. Raw string or an `ABC_RECIPES` entry |
| `prep_script` | `"techmap; opt; abc -fast; opt"` | Yosys passes that lower coarse cells to gates before ABC. Rarely changed |
| `timeout` | `None` (wait indefinitely) | Wall-clock seconds for the ABC subprocess. Set a number to bound long scripts (e.g. `&deepsyn`/`&transtoch` with big budgets) |
| `cache_read` | `"both"` | Which caches to consult: `"none"` / `"mem"` / `"disk"` / `"both"` |
| `cache_write` | `"both"` | Which caches to populate: same values |
| `cache_dir` | `None` | Override cache directory |

> A standalone `abc` binary enables the effective out-of-process optimization. Discovery
> order: `$SPIRE_ABC`, then `abc` on `PATH`, then `yosys-abc`. If none is found,
> `abc_optimize` automatically falls back to an in-process pyosys path (legacy ordering,
> largely inert on coarse cells) and emits a `RuntimeWarning` — so it still runs on a
> pure-pyosys install, just without the gains.

### Lower-level function

`abc_optimize(design, abc_script=..., prep_script=...)` takes a `Component` or `Netlist`
and returns optimized AAG lines directly, without the decorator/caching machinery.

```python
from spire.optimize import abc_optimize, ABC_RECIPES

aag_lines = abc_optimize(my_component, abc_script=ABC_RECIPES["area"])
```

> **Prefer `@abc_optimized` when the design must keep its interface.** AAG is bit-blasted,
> so re-importing these lines (`AigerImporter(aag).get_spire_netlist()`) yields a netlist
> with *scalar* ports (`a_0_`, `a_1_`, …) instead of the original buses — emitting that
> gives a module a bus-port testbench cannot bind to. The decorator re-attaches the
> optimized logic to the component's own IO, preserving `(a, b, y)`. Use the low-level
> function when you want the AAG itself, not a drop-in replacement module.

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

## Caching

`@abc_optimized` uses a two-level cache:

1. **In-memory** -- keyed by a SHA-256 hash of the Verilog content + non-logic arguments + optimizer parameters.  Instant on repeated calls within the same process.
2. **Disk** -- stored in `.spire_cache/v1/` as JSON files containing the optimized AAG lines and port spec.  Survives across runs.

Use `clear_optimization_cache()` to reset the caches — by default it clears the in-memory cache AND removes the versioned disk cache (`.spire_cache/v1/`); pass `disk=False` for in-memory only.

```python
from spire.optimize import set_cache_dir, clear_optimization_cache

set_cache_dir("/my/cache/path")   # override default location
clear_optimization_cache()         # clears in-memory AND the .spire_cache/v1/ disk cache
clear_optimization_cache(disk=False)  # in-memory only
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
from spire import Component, IORecord, Input, Output, UInt
from spire.optimize import arithmetic_optimized

@arithmetic_optimized(objective="adp")
def opt_mac(a, b, c):
    return a * b + c

class Top(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(8)), b=Input(UInt(8)), c=Input(UInt(16)), y=Output(UInt(17)))
        self.elaborate()

    def elaborate(self):
        io = self.io
        io.y <<= opt_mac(io.a, io.b, io.c)    # MAC fusion happens inside the decorator

print(Top().to_verilog(name="Top"))
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
