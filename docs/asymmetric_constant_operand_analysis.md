# Asymmetric Widths & Constant Operand Analysis

Analysis of potential gains from two optimizations in `replace_arithmetic_ops`:
1. **Asymmetric width sweep**: evaluating actual (a_w, b_w) pairs instead of using `max(a_w, b_w)` from the symmetric database
2. **Constant operand specialization**: exploiting known-constant inputs to reduce circuit size

---

## 1. Asymmetric Width Sweep

### Current behavior (max-proxy)

When operands have different widths (e.g., 4-bit + 8-bit), the auto-config currently looks up `max(a_w, b_w)` in the symmetric database. This means a 4x8 adder gets the config optimized for 8x8.

### Adder results

The proxy uses the 8x8 or 16x16 best config. The asymmetric sweep evaluates all FSA topologies at the actual (a_w, b_w) and also tries the swapped orientation (b_w, a_w).

| a x b | Objective | Proxy FSA | Proxy TC | Proxy Depth | Best Asym FSA | Asym TC | Asym Depth | TC saved | Depth saved |
|------:|-----------|-----------|-------:|------:|---------------|------:|------:|------:|------:|
| 4 x 8 | area | Sparse-KS-2 | 306 | 16 | Ripple Carry | 214 | 13 | **92 (30%)** | 3 |
| 4 x 8 | delay | Sparse-KS-2 | 306 | 16 | Kogge-Stone | 252 | 8 | 54 | **8 (50%)** |
| 4 x 8 | adp | Sparse-KS-2 | 306 | 16 | Kogge-Stone | 252 | 8 | 54 | 8 |
| 4 x 16 | area | Sparse-KS-2 | 626 | 32 | Ripple Carry | 360 | 21 | **266 (43%)** | 11 |
| 4 x 16 | delay | Kogge-Stone | 952 | 22 | Sparse-KS-2 | 478 | 10 | 474 | **12 (55%)** |
| 4 x 16 | adp | Brent-Kung | 746 | 26 | Sparse-KS-2 | 478 | 10 | **268 (36%)** | 16 |
| 8 x 16 | area | Sparse-KS-2 | 626 | 32 | Sparse-KS-2 | 454 | 23 | **172 (27%)** | 9 |
| 8 x 16 | delay | Kogge-Stone | 952 | 22 | Sparse-KS-2 | 454 | 23 | 498 | -1 |
| 8 x 16 | adp | Brent-Kung | 746 | 26 | Sparse-KS-2 | 454 | 23 | **292 (39%)** | 3 |

**Conclusion**: Asymmetric adder sweep saves **27-55%** transistors and depth depending on width ratio and objective. The wider the ratio, the larger the gain. Even for 8x16 (2:1 ratio), there's a consistent 27-39% area reduction.

### Multiplier results

The proxy uses the best 8x8 or 16x16 symmetric config. The asymmetric sweep tests all PPG x PPA x FSA combinations in both orientations (a_w, b_w) and (b_w, a_w).

| a x b | Objective | Proxy Config | Proxy TC | Proxy Depth | Best Asym Config (orient) | Asym TC | Asym Depth | TC saved |
|------:|-----------|-------------|-------:|------:|---------------------------|------:|------:|------:|
| 4 x 8 | area | AND/Dadda/RippleCarry | 2438 | 73 | AND/Dadda/RippleCarry (8x4) | 1092 | 41 | **1346 (55%)** |
| 4 x 8 | delay | AND/Accum/Kogge-Stone | 2942 | 29 | AND/Accum/Sklansky (4x8) | 1186 | 18 | **1756 (60%)** |
| 4 x 8 | adp | AND/CarrySave/Sklansky | 2522 | 30 | AND/Accum/Sklansky (4x8) | 1186 | 18 | **1336 (53%)** |
| 4 x 16 | area | Booth/CarrySave/HanCarlson | 10118 | 62 | AND/Dadda/HanCarlson (16x4) | 2320 | 78 | **7798 (77%)** |
| 4 x 16 | delay | AND/CarrySave/Kogge-Stone | 11438 | 45 | AND/Accum/Brent-Kung (4x16) | 2884 | 21 | **8554 (75%)** |
| 4 x 16 | adp | Booth/CarrySave/Kogge-Stone | 10622 | 47 | AND/Accum/Brent-Kung (4x16) | 2884 | 21 | **7738 (73%)** |
| 8 x 16 | area | Booth/CarrySave/HanCarlson | 10118 | 62 | Booth/Dadda/Sparse-KS-2 (16x8) | 5112 | 88 | **5006 (49%)** |
| 8 x 16 | delay | AND/CarrySave/Kogge-Stone | 11438 | 45 | AND/Accum/Kogge-Stone (8x16) | 6408 | 31 | **5030 (44%)** |
| 8 x 16 | adp | Booth/CarrySave/Kogge-Stone | 10622 | 47 | AND/CarrySave/Kogge-Stone (8x16) | 5930 | 33 | **4692 (44%)** |

**Conclusion**: Asymmetric multiplier sweep saves **49-77%** transistors. This is expected — a 4x8 multiplier has fundamentally fewer partial products than an 8x8 one. The operand orientation matters and differs by objective:
- **Area**: wider operand as `a` tends to win (8x4, 16x4, 16x8)
- **Delay/ADP**: narrower operand as `a` can be better (4x8, 4x16)

The DB should store the winning orientation alongside the config.

### Summary: asymmetric sweep

| Operation | Typical TC savings | Worth implementing? |
|-----------|-------------------:|---------------------|
| Adder | 27-55% | **Yes** — significant, cheap to sweep (FSA only) |
| Multiplier | 53-77% | **Yes** — massive savings, sweep is larger but results are dramatic |

The sweep cost is manageable: adders have ~11 FSA options x 2 orientations = 22 configs per width pair. Multipliers have ~5 PPG x 5 PPA x 11 FSA x 2 orientations = 550 configs per width pair, but can be parallelized.

---

## 2. Constant Operand Specialization

When one operand of `+` or `*` is a compile-time constant (e.g., `a * 42` or `a + 1`), the circuit can be dramatically smaller.

### Methodology

All metrics were collected using the existing pipeline — no special constant-propagation pass:
- **Transistor count**: `get_yosys_metrics(module)` — exports to AAG, runs `optimize_aag()` (aigverse: resubstitution + SOP refactoring + cut rewriting, 3 iterations), then Yosys `synth` + `stat` for transistor estimation
- **AIG depth**: `get_aig_stats(module)` — same AAG export + `optimize_aig_elaborate()` (aigverse, 3 iterations), then `DepthAig().num_levels()`

The constant simply flows through the existing build pipeline. The `AigerExporter` bit-blasts the `Const()` node into fixed 0/1 literals. The aigverse AIG optimizer then propagates these constants (AND with 0 = 0, AND with 1 = identity) and eliminates dead gates. No special handling is needed in `replace_arithmetic_ops`.

### Variable config + constant input (transparent approach)

Using the best variable-input config from the DB, but feeding a constant into one operand. The AIG optimizer handles constant propagation automatically.

**8-bit adder (including +1, +2 — common increment/decrement patterns):**

| Const | Var TC | VarCfg+Const TC | Depth | TC save | Var Depth | Depth save |
|------:|-------:|------:|------:|------:|------:|------:|
| 1 | 306 | 126 | 8 | **59%** | 16 | **50%** |
| 2 | 306 | 110 | 7 | **64%** | 16 | **56%** |
| 3 | 306 | 126 | 8 | **59%** | 16 | **50%** |
| 7 | 306 | 126 | 8 | **59%** | 16 | **50%** |
| 42 | 306 | 114 | 7 | **63%** | 16 | **56%** |
| 127 | 306 | 150 | 5 | **51%** | 16 | **69%** |
| 255 | 306 | 150 | 5 | **51%** | 16 | **69%** |

Constants 127 (0x7F) and 255 (0xFF) achieve the best depth reduction (69%) despite slightly higher transistor count — many carry-chain positions become trivial when adding all-ones patterns. For delay objective, sweeping FSA configs adds no extra gain (the variable-config already wins for all constants).

**8-bit multiplier:**

| Const | Var TC | VarCfg+Const TC | Depth | Total save |
|------:|-------:|------:|------:|------:|
| 1 | 2438 | 0 | 0 | **100%** |
| 2 | 2438 | 0 | 0 | **100%** |
| 3 | 2438 | 500 | 12 | **79%** |
| 7 | 2438 | 788 | 16 | **68%** |
| 15 | 2438 | 1018 | 19 | **58%** |
| 42 | 2438 | 782 | 31 | **68%** |
| 127 | 2438 | 1574 | 56 | **35%** |
| 255 | 2438 | 1782 | 59 | **27%** |

**16-bit multiplier:**

| Const | Var TC | VarCfg+Const TC | Depth | Total save |
|------:|-------:|------:|------:|------:|
| 3 | 10118 | 1218 | 33 | **88%** |
| 42 | 10118 | 2320 | 43 | **77%** |
| 255 | 10118 | 1804 | 22 | **82%** |

Note: `*1` and `*2` optimize to 0 transistors because AIG optimization reduces them to identity/shift — the entire multiplier is eliminated.

### Constant-aware config sweep (additional gain from choosing a different architecture)

The question: does a different PPG/PPA/FSA perform better when one input is constant, compared to the best variable-input config?

**8-bit adder** — sweeping FSA options with constant input (area and delay):

| Const | VarCfg+C TC | VarCfg+C Dep | Best swept (area) | Swept TC | Extra TC save | Best swept (delay) | Swept Dep | Extra Dep save |
|------:|------:|------:|---|------:|------:|---|------:|------:|
| 1 | 126 | 8 | (equal) | 126 | 0% | (equal) | 8 | 0% |
| 2 | 110 | 7 | (equal) | 110 | 0% | (equal) | 7 | 0% |
| 3 | 126 | 8 | (equal) | 126 | 0% | (equal) | 8 | 0% |
| 42 | 114 | 7 | (equal) | 114 | 0% | (equal) | 7 | 0% |
| 127 | 150 | 5 | Ripple Carry | 126 | **16%** | (equal) | 5 | 0% |
| 255 | 150 | 5 | Ripple Carry | 126 | **16%** | (equal) | 5 | 0% |

For adders, the constant-aware sweep adds minimal value — 0-16% extra TC savings for area, and **zero** extra depth savings for delay. The transparent approach already captures all the gain. A constant-specific DB is not worthwhile for adders.

**8-bit multiplier** — sweeping PPG x PPA x FSA with constant input:

| Const | VarCfg+C TC | Best swept config | Swept TC | Extra save |
|------:|------:|---|------:|------:|
| 1 | 0 | (equal) | 0 | 0% |
| 2 | 0 | (equal) | 0 | 0% |
| 3 | 500 | BoothUnopt/Dadda/RippleCarry | 280 | **44%** |
| 7 | 788 | BoothUnopt/Dadda/RippleCarry | 292 | **63%** |
| 15 | 1018 | BoothUnopt/Dadda/RippleCarry | 290 | **72%** |
| 42 | 782 | AND/Accum/RippleCarry | 660 | **16%** |
| 127 | 1574 | BoothUnopt/Dadda/BrentKung | 276 | **83%** |
| 255 | 1782 | BoothUnopt/Dadda/KoggeStone | 168 | **91%** |

**16-bit multiplier:**
\
| Const | VarCfg+C TC | Best swept config | Swept TC | Extra save |
|------:|------:|---|------:|------:|
| 3 | 1218 | BoothOpt/Dadda/SparseKS2 | 612 | **50%** |
| 42 | 2320 | AND/Accum/RippleCarry | 1486 | **36%** |
| 255 | 1804 | BoothUnopt/Dadda/RippleCarry | 602 | **67%** |

**For multipliers, the constant-aware sweep provides massive additional gains (16-91%).** The optimal architecture changes dramatically when one input is constant — Booth Unoptimised with Dadda Tree often wins because its structure maps well to constant patterns after AIG optimization.

### Summary

The savings stack: transparent constant propagation gives 27-100% over variable, and a constant-aware config sweep gives an additional 16-91% on top:

| Const (8-bit mul) | Variable TC | Transparent TC | Swept TC | Total save |
|---:|---:|---:|---:|---:|
| 3 | 2438 | 500 | 280 | **88%** |
| 42 | 2438 | 782 | 660 | **73%** |
| 127 | 2438 | 1574 | 276 | **89%** |
| 255 | 2438 | 1782 | 168 | **93%** |

### Integration complexity

**Transparent (Option A)** — zero effort, already works:
The constant flows through the existing build pipeline → AIG export bit-blasts it to 0/1 literals → aigverse optimizer propagates and simplifies. Gains: 27-100%.

**Constant-aware sweep (Option B)** — additional gain from DB entries:
Detect `Const` nodes in `replace_arithmetic_ops`, look up a constant-specific config from the DB. The DB would need entries keyed by `(op, width, constant_value)` or `(op, width, hamming_weight)`. Gains: additional 16-91% beyond Option A.

**Shift-and-add decomposition (Option C)** — more radical:
For specific constants, decompose multiplication into shifts and adds (e.g., `*3 = (x<<1) + x`). Would bypass the multiplier entirely. Higher implementation effort, potentially optimal for small constants.

---

## 3. Comparison Replacement

Investigated replacing comparison operators (`==`, `!=`, `<=`, `>`) with optimized hardware at the graph level (inside `replace_arithmetic_ops`), before the AIG writer's default chain-based implementations.

### Equality (`==`, `!=`): XOR + NOR-tree

The AIG writer's `bv_eq` uses a linear AND-chain (O(n) depth). Replacing `==`/`!=` at the graph level with XOR per bit + balanced OR-tree + final invert gives O(log n) depth.

Tested on the FP16 multiplier (119 `==` ops, 2 `!=`, all with constant operands):

| Objective | Arith only TC | + NOR tree TC | Delta TC | Arith only Depth | + NOR tree Depth | Delta Depth |
|-----------|---:|---:|---:|---:|---:|---:|
| area | 11668 | 11640 | **-28** | 222 | 217 | **-5** |
| delay | 12580 | 12496 | **-84** | 145 | 143 | **-2** |
| adp | 11900 | 11762 | **-138** | 145 | 146 | +1 |

Small but consistent improvements. The NOR-tree is now integrated into `replace_arithmetic_ops` — applied automatically for all `==` and `!=` ops.

A linear NOR-chain (O(n) depth) was also tested but had **zero effect** — the AIG optimizer already produces the equivalent chain, so explicit chaining adds nothing.

### Inequality (`<=`, `>`, `!=`): subtraction-based

Replacing inequality comparisons with prefix subtractors (carry-out for `<=`/`>`, zero-detect for `!=`):

| Objective | NOR tree only TC | + ineq sub TC | Delta TC | NOR tree Depth | + ineq sub Depth | Delta Depth |
|-----------|---:|---:|---:|---:|---:|---:|
| area | 11640 | 11728 | +88 | 217 | 217 | 0 |
| delay | 12496 | 12596 | +100 | 143 | 139 | **-4** |
| adp | 11762 | 11876 | +114 | 146 | 141 | **-5** |

Inequality subtraction helps depth (-4 to -5) but **hurts area** (+88 to +114 TC). Only 5 inequality ops in the FP16 design — too few to justify the overhead. Not implemented.

### Conclusion

- **`==`/`!=` NOR-tree**: implemented, consistent small gains across all objectives
- **Inequality subtraction**: not worth it for designs with few comparisons, trades area for small depth gain
- **Overall**: comparisons are a minor contributor; the 119 equality ops in FP16 are mostly small-width (5-10 bits) against constants, where the tree vs chain difference is minimal

---

## 4. Fused Multiply-Accumulate (MAC) Detection

Investigated detecting `a * b + c` patterns in the graph and replacing with a fused MAC, which feeds the accumulate operand directly into the multiplier's column reduction (saving one full FSA stage).

SpireHDL already has a `FusedMacComponent` / `fused_inner_product` infrastructure.

### Results: fused MAC vs separate auto-replaced mul+add

| Size | Sep auto (area) TC | Fused (area) TC | Fused better? | Sep auto (adp) TC | Fused (adp) TC | Fused better? |
|------|---:|---:|---|---:|---:|---|
| 4x4 | 776 | 790 | No (+2%) | 788 | 802 | No (+2%) |
| 8x8 | 3072 | 3092 | No (+1%) | 3312 | 3170 | **Yes (-4%)** |
| 16x16 | 11406 | 11354 | **Yes (-0.5%)** | 12288 | 11580 | **Yes (-6%)** |

The fused MAC is **marginally better at larger widths** (8x8+, up to 6% for adp) but **slightly worse at small widths**. The AIG optimizer is effective enough to mostly eliminate the redundant FSA between a separate multiplier and adder.

For asymmetric widths (4x8, 8x16), the fused MAC was significantly worse (up to 95% more transistors) because the current `FusedMacComponent` uses `max(a_w, b_w)` (square multiplier), while separate auto-config handles actual asymmetric widths natively.

### With dedicated fused MAC config sweep

The earlier results used the multiplier's best config, which isn't optimal for a fused MAC (the accumulate operand changes the column structure). A dedicated sweep over PPG x PPA x FSA x optim_type for the fused MAC yields much better results:

**8-bit MAC (200 configs swept):**

| Objective | Sep auto TC | Sep Depth | Fused swept TC | Fused Depth | TC delta | Depth delta |
|-----------|---:|---:|---:|---:|---:|---:|
| area | 3072 | 79 | 3088 | 37 | +0.5% | **-53%** |
| delay | 3620 | 48 | 4288 | 28 | +18% | **-42%** |
| adp | 3312 | 41 | 3170 | 32 | **-4%** | **-22%** |

**16-bit MAC (200 configs swept):**

| Objective | Sep auto TC | Sep Depth | Fused swept TC | Fused Depth | TC delta | Depth delta |
|-----------|---:|---:|---:|---:|---:|---:|
| area | 11406 | 73 | 11354 | 65 | **-0.5%** | **-11%** |
| delay | 13824 | 56 | 12716 | 47 | **-8%** | **-16%** |
| adp | 12288 | 61 | 12716 | 47 | +3% | **-23%** |

The fused MAC **dominates on depth** (11-53% shallower) at roughly equal area. The depth savings come from eliminating the separate FSA stage between multiplier and adder — the accumulate operand is absorbed directly into the multiplier's column reduction.

Best fused MAC configs differ from best multiplier configs: CarrySave PPA wins consistently for MAC (vs Dadda for standalone multiplier), because CarrySave handles the extra accumulate columns more efficiently.

### Conclusion

- **With proper sweep**: fused MAC gives **11-53% depth reduction** at similar area — significant for delay-sensitive designs
- **Without sweep** (using multiplier config): only 0-6% gain — the config mismatch hides the fusion benefit
- A fused MAC sweep in the DB would unlock this potential
- Pattern detection (`Op2<+>` with `Op2<*>` child) is straightforward in the existing graph walker
- **Recommended** as a medium-priority optimization, especially for DSP-heavy designs (filters, neural nets)

---

## 5. Recommendations

| Feature | Gain | Effort | Priority |
|---------|------|--------|----------|
| Asymmetric adder sweep in DB | 27-55% TC | Low (22 configs/pair) | **High** |
| Asymmetric multiplier sweep in DB | 49-77% TC | Medium (550 configs/pair, parallelize) | **High** |
| Operand swap selection from DB | Included in above | Included | **High** |
| Constant operand transparent (A) | 27-100% TC | None (already works) | **Free** |
| Constant-aware config sweep (B) | +16-91% on top of A | Medium (sweep per constant class) | **High** for multipliers |
| `==`/`!=` NOR-tree | -28 to -138 TC, -2 to -5 depth | Low (implemented) | **Done** |
| Inequality subtraction | -4 to -5 depth, +88-114 TC | Low | **Low** (trades area for depth) |
| Fused MAC detection + sweep | 11-53% depth, ~equal area | Medium | **Medium** (big depth win for DSP designs) |
| Shift-and-add decomposition (C) | Potentially optimal for small consts | High | Low (B suffices) |

| Feature | Gain | Effort | Priority |
|---------|------|--------|----------|
| Asymmetric adder sweep in DB | 27-55% TC | Low (22 configs/pair) | **High** |
| Asymmetric multiplier sweep in DB | 49-77% TC | Medium (550 configs/pair, parallelize) | **High** |
| Operand swap selection from DB | Included in above | Included | **High** |
| Constant operand transparent (A) | 27-100% TC | None (already works) | **Free** |
| Constant-aware config sweep (B) | +16-91% on top of A | Medium (sweep per constant class) | **High** for multipliers |
| Shift-and-add decomposition (C) | Potentially optimal for small consts | High | Low (B suffices) |
