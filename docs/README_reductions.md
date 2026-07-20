# Reductions (`spire.reduce`)

Balanced log-depth trees for fold operations: max, min, argmax/argmin, sum,
product, clamp, and a generic `reduce_tree`.

## Why

A reduction written as the natural loop builds an O(N)-deep chain:

```python
acc = xs[0]
for i in range(1, n):
    acc = mux(xs[i] > acc, xs[i], acc)   # running max — 19 muxes deep for n=20
```

Downstream synthesis cannot fix this. There is no netlist cell for word-level
max/min (Verilog has no max operator), so the design reaches yosys/ABC as
compare+mux cones whose associativity is invisible. Measured on a 20-input
8-bit max through full `yosys synth` (`ltp`):

| form | cells | depth |
|---|---|---|
| loop-built chain | 1038 | 171 |
| `max_(xs)` balanced tree | 988 | **45** |

Bitwise chains (`and`/`or`/`xor`) are associative at the AIG level, so a
thorough ABC script *can* rebalance them — but the default `yosys synth` flow
(`abc -fast`) measurably does not (63 levels for a 64-bit parity chain, vs 6
for the tree), and pre-synthesis consumers (the Python simulator, direct AIG
export and its metrics) always see the graph spire built. The tree is cheap
insurance either way. Full numbers in [Measured](#measured) below.

## API

```python
from spire.reduce import (reduce_tree, prefix_scan,
                          max_, min_, sum_, prod_, clamp_, argmax_, argmin_)

y  <<= max_(xs)                              # balanced compare-select tree
y  <<= sum_(xs)                              # balanced adder tree (result widens; slice to fit)
y  <<= clamp_(x, lo, hi)                     # min(max(x, lo), hi), assumes lo <= hi
y  <<= reduce_tree(lambda a, b: a | b, xs)   # generic fold; fn must be associative

val, idx = argmax_(xs)                       # (value, index) — leftmost of equal values wins

y  <<= max_(xs, topology="matrix")           # topology= selects the physical form (below)
running = prefix_scan(lambda a, b: mux(a >= b, a, b), xs)   # all N running maxes, O(log N) deep
```

| helper | returns | notes |
|---|---|---|
| `reduce_tree(fn, xs)` | `Expr` | associativity of `fn` is caller-asserted, not checked |
| `max_` / `min_` | `Expr` | compare-select tree; signedness from the element types |
| `sum_` / `prod_` | `Expr` | widths widen per tree level, so no intermediate overflow; slice the result to your target width. For long chains feeding arithmetic, prefer the [arithmetic generator](README_arithmetic_generator.md)'s compressor trees |
| `clamp_(x, lo, hi)` | `Expr` | two compare-selects, constant depth |
| `argmax_` / `argmin_` | `(value, index)` | index is `ceil(log2(N))` bits; ties resolve to the **leftmost** element, matching a left-to-right linear scan |

All order-preserving forms pair elements left-to-right; the default `"tree"`
has depth `ceil(log2 N)` at N−1 operators — same operator count as the loop,
so the depth win is free.

## Topologies

Every fold helper takes `topology=`; `prefix_scan` has its own set.

| topology | depth | area | notes |
|---|---|---|---|
| `"tree"` (default) | O(log N) | N−1 ops | balanced binary |
| `"chain"` | O(N) | N−1 ops | serial left fold — baseline/reference |
| `"huffman"` | ≤ tree | N−1 ops | arrival-aware: shallow input cones merge first (unit-cost expression-depth heuristic). Reorders operands, so `fn` must be commutative; rejected for `argmax_`/`argmin_` (would break the tie rule) |
| `"matrix"` | ~2 stages | O(N²) compares | `max_`/`min_`/`argmax_`/`argmin_` only: all pairs compared in parallel, one-hot winner select. Shallowest form for small N; index comes out one-hot for free |

`prefix_scan(fn, xs, topology=...)` returns all N inclusive prefixes — use it
when partial results are tapped, instead of duplicating chains:

| scan | depth | area | fanout |
|---|---|---|---|
| `"sklansky"` (default) | log N | (N/2)·log N ops | high |
| `"brentkung"` | 2·log N | ~2N ops | low |
| `"koggestone"` | log N | N·log N ops | low |

Scans preserve operand order, so `fn` only needs associativity (not
commutativity).

## Measured

All numbers from
[`testing/examples/reduction_topology_bench.py`](../testing/examples/reduction_topology_bench.py)
(`python testing/examples/reduction_topology_bench.py`, needs yosys on PATH for
the syn columns). "AIG" = spire's raw AIG export (what the simulator and AIG
metrics see); "syn" = `yosys synth` + `ltp` (cells / gate levels).

**Peak select — max of 20 × 8-bit** (`loop` = the hand-written running max):

| topology | AIG gates | AIG depth | syn cells | syn depth |
|---|---|---|---|---|
| loop | 1748 | 361 | 1038 | 171 |
| `tree` | 1748 | 95 | 988 | **45** |
| `huffman` | 1748 | 95 | 939 | 45 |
| `matrix` | 13592 | 28 | 7892 | **19** |

Same operator count as the loop, 3.8× shallower — free. `matrix` buys another
2.4× depth for 8× area. `huffman` equals `tree` here because all inputs arrive
at depth 0; it pays off only for skewed arrival times.

**Pooling winner — argmax of 16 × 8-bit** (value + index):

| topology | AIG gates | AIG depth | syn cells | syn depth |
|---|---|---|---|---|
| `chain` | 1453 | 285 | 864 | 135 |
| `tree` | 1413 | 76 | 820 | **36** |
| `matrix` | 8654 | 26 | 5054 | **17** |

**Watermark — running max of 16 with all prefixes tapped:**

| topology | AIG gates | AIG depth | syn cells | syn depth |
|---|---|---|---|---|
| loop | 1380 | 285 | 905 | 135 |
| `sklansky` | 2944 | 76 | 1675 | **36** |
| `brentkung` | 2392 | 114 | 1344 | 54 |
| `koggestone` | 4508 | 76 | 2590 | **36** |

The textbook trade: Brent-Kung is the cheapest scan at 1.5× the depth of
Sklansky; Kogge-Stone matches Sklansky's depth with lower fanout but the most
area.

**Parity — xor reduce of 64 bits:**

| topology | AIG gates | AIG depth | syn cells | syn depth |
|---|---|---|---|---|
| `chain` | 189 | 126 | 63 | 63 |
| `tree` | 189 | 12 | 63 | **6** |

Identical cell count — the default `synth` flow keeps whatever shape the
source had, so the chain stays 63 levels deep even *after* synthesis.

## Relation to `selection_topology`

[Selection cascades](README_control_structures.md) are the sibling problem:
choosing among values by independent predicates. Reduction-shaped chains (arms
referencing the chain itself) are deliberately **skipped** by
`selection_topology` — no selection topology can shorten them, since the serial
dependency lives in the arms. These helpers are the intended replacement: state
the reduction, get the tree. A future recognition layer that rebalances
loop-built reductions automatically is sketched in
`metadocs/REDUCTION_TOPOLOGY_PLAN.md`.

Integer/bitwise only — floating-point folds are not associative.

## Tests

See [`testing/test_reduce.py`](../testing/test_reduce.py): random + exhaustive
equivalence per helper, argmax tie-breaking (exhaustive with duplicates), and a
tree-depth structure check.
