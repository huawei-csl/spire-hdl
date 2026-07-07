# Re-fix status — branch `fix/issue-review-v2`

Single tracking table for the plan in REFIX_PLAN.md. ISSUES.md / ISSUES2.md / REVIEW_GUIDE.md stay frozen as the
source record (they live on the old branch / main checkout); status changes happen only here.

Conventions: **Item** = plan phase step; **Source** = issue ids it discharges (I = ISSUES.md, I2 = ISSUES2.md,
R = REAUDIT.md); **Status** = `pending` → `in progress` → `applied` (awaiting review) → `committed <sha>`;
each item's commit message cites the same ids. An item is not `applied` without its failing-then-passing test.

## Phase 0 — scaffolding

| Item | Source | Status | Notes |
|---|---|---|---|
| Semantics reference (`docs/README_semantics.md`) | I2 §1 preamble | applied | process-free reference doc |
| Plan + status docs in `metadocs/`, committed via .gitignore whitelist | — | applied | this commit |
| Differential harness in `testing/differential/` | I2 §1, harness from issues2_repros | applied | ieee_eval (yosys-calibrated port) + harness + leaf battery + nested battery + fuzz; 2.8 s |
| Baseline run recorded; bug shapes as strict xfails | I 0.1, 0.2, 0.3-0.6, 13.2; I2 §1.1-§1.6 | applied | harness: 62 pass / 33 strict-xfail; leaf: mixed ± * and any-signed ordered cmp diverge (= 0.1/0.2 exactly); (s4-s4)<s5 conformant-by-accident — guards the §1.3 trap; AIGER: unsigned fuzz clean, signed xfail (0.3-0.6). Full suite @ baseline: 539 passed / 19 skipped / 45 xfailed / 12 xpassed / 0 failed, 13m28s (12 xpass = the pre-existing non-strict BOOTH_OPTIMISED sweep; skips = 12 BAUGH_WOOLEY-signed-only + 7 missing PPA fingerprints) |
| `simplify=` docstring caveat (experimental until Phase 7) | I 0.7/1.1, I2 §2.1 | applied | ir.py to_verilog_lines docstring |

## Phase 1 — expression signedness & width-correct emission

| Item | Source | Status | Notes |
|---|---|---|---|
| Width-isolation pass (emission-time, always on) | I2 §1.1, §1.2, §1.5, §1.6; I 13.2; I 0.1 (all-signed half) | applied | src/spire/width_isolation.py + ir.py hook, after simplify/balance/cse so pattern matchers see original nesting; flat-safe 1-bit boolean cones stay inline (preserves _minimize_emit's flat-PPA path, flag-free); flipped 13 strict xfails (all §1.1/§1.5/§1.6/13.2 shapes, unsigned fuzz, ss ordered compares + narrow compound compare — alignment Resize wires carry declared signedness); FpMulSN(5,10) structure cost +47 wires (~+8.7%); fast sweep 134 passed; full suite 552/19/32xf/12xp/0 fail = baseline + exactly the 13 flips |
| Signed compares re-fix on isolated operands | I 0.1, I2 §1.3 | applied | mixed compares promoted to max+1 signed at construction (op_cmp) — exact integer compare per charter §2; NO $signed() wrapping (declared signedness carries the island — avoids the §1.3 trap; guard test green). Sim semantics corrected for mixed compares: baseline coerced-to-signed (u8==s-8 was true), now exact (false) — charter authority case |
| Mixed-arith alignment re-fix | I 0.2, I2 §1.4 | applied | `_align_mixed_arith`: mixed operands materialized at result width per own signedness (op_add/sub/mul, covers unary minus); exact as pattern arithmetic mod 2^w; §1.4 poisoning structurally gone (isolated nodes). Bonus: AIGER-signed fuzz conformant — exporter's mixed paths (0.3-0.5) no longer exercised by operator-built IR (Phase-4 audit still due); 0.6 (multi-bit mux selector, 720 mism.) pinned as dedicated strict xfail. Harness 95 pass / 1 xfail (only 0.6) |
| Ternary emission per charter C1 | I2 §1.5 | applied | discharged by the width-isolation pass (mixed-sign muxes get an explicitly-typed boundary wire); mux_operand shape green |
| Const validation + boundary-literal emission | I2 §2.2, §2.7, R P9 | applied | Const.__init__ raises on unrepresentable values (kills P9 + §2.7 at the root; simplify's raw folds now crash instead of silently wrong — acceptable, simplify is opt-in/experimental until its phase); most-negative literals emit `$signed(w'd<pattern>)` (old `-w'sd` form proven divergent: 256/256); Resize const path masks instead of constructing invalid Consts. Also fixed harness evaluator fidelity: `-w'sd` now parsed as unary minus per IEEE (was folded into the literal — would have masked §2.2) |
| Negative shift amounts rejected | I2 §2.6, charter C3 | applied | op_shift ValueError for const amounts < 0 (both directions) |
| Identifier legality | I2 §2.4, charter C6 | applied | keyword set + suffixing in sanitize_signal_name (inferred `reg` → `reg_`); explicit names checked at emission (AIGER importer's transient `y[0]` names stay legal pre-grouping); module names checked at construction. CS6 suite fallout: test_const_width_masked repurposed (out-of-range Const premise now impossible) |
| Register `init=` restricted to direct constants | I2 §2.3 | applied | redesigned per review: instead of collecting init cones, `set_init` accepts only a direct Const (re-typed to the register, range-checked) — the §2.3 failure mode (undeclared/self-captured reset net, reproduced as `r <= r[3:0]`) becomes unconstructible, reset arms emit plain literals, and dynamic reset values (unsynthesizable as async set/clear) are rejected loudly. CompositeRegister init narrowed to packed ints (composite leaves are wires); two composite tests updated to the new contract; semantics doc §3.2 extended |
| `flat_emit` scope decision | I 13.2, I2 §1.6 | pending | |

## Phase 2 — core structures

| Item | Source | Status | Notes |
|---|---|---|---|
| Control flow: chain + condition-state scoping | I 2.3, I2 §4.1–4.3, §4.8; R R6/I 13.1 | applied | redesign: `fresh_condition_scope` (save/clear/restore at component construction) + strong-ref identity fingerprint on pending chains (kills §4.8's id-reuse hazard); 13.1 limitation REMOVED (component between if_/else_ now works); conditional patch moved from `__ilshift__` to `assign` (§4.2) with composite packing (§4.1); 10 new scoping tests |
| Control flow minors | I2 §4.4, §4.7 | applied | case-value representability check; _ConditionalContext entered-guard. (§4.5 arr[i]=x and §4.6 State subclassing → composite/state change sets). CS8 fallout fixed en route: test_array2 reg init packed numerically (was an Array-of-Consts `.to_bits()` pattern — note: in-tree evidence for the declined const-folding variant). Full suite 609/19/13xf/12xp/0 fail |
| Sharing/CSE correctness (1.2/1.3/1.4) | I 1.2, 1.3, 1.4 | applied | adopt-as-is per guide, plus the tests the old branch never wrote (all three fail at baseline): visitor cache pins nodes via (node, result) tuples; CSE redirects the first duplicate unconditionally; force_share honored regardless of prior reference count. Note: width isolation already masks 1.4's emission symptom — fix is mechanism correctness |
| Sharing determinism/name hygiene (1.6, 2.6, §2.5) | I 1.6, 2.6, I2 §2.5 | applied | duplicate-port ValueError on all four creation paths (incl. implicit clk/rst); default netlist name = class name; §2.5 redesigned deeper than the old branch: construction wire names are provisional, the collector assigns canonical per-netlist names (suggested base or sig_N in traversal order) — same-process re-emission byte-identical with NO manual reset, composition-safe. 3 tests, all fail at pre-fix state. 1.5 moved to AIGER phase |
| Component/IO: clk/rst are framework-only | I 2.1 | applied | DESIGN DECISION (review): clocking is implicit-only via with_clock/with_reset — IO leaves named clk/rst are rejected with guidance (2.1's silent drop → loud error; guide's "adopt" overridden; zero in-tree io-clk users; keeps room for future clock-domain contexts). Tests: framework-flag path incl. sim; both leaf names rejected |
| Component/IO remainder | I 2.2, 2.5, 2.7, 4.5 | pending | next |
| Composite port naming (depth-safe) | I 4.8-equiv, I2 §5.1 guard | pending | ships with ≥2-depth tests |
| Composite re-fixes + minors | I 4.1–4.6, I2 §5.2–5.10, charter C5 | pending | |
| Simulator core re-fixes (both reset paths) | I 3.1–3.4, I2 §3.1–3.6 | pending | |

## Phase 3 — memories, primitives, interfaces

| Item | Source | Status | Notes |
|---|---|---|---|
| 5.1 with async-ROM exemption | I 5.1, I2 §6.4 guard | pending | |
| Memory/FIFO re-fixes + via_reg parity | I 5.2–5.8, I2 §6.1–6.3, §6.5–6.7 | pending | |

## Phase 4 — backends (AIGER, decorators)

| Item | Source | Status | Notes |
|---|---|---|---|
| AIGER re-fixes (mostly adopt) | I 0.3–0.6, 3.5, 3.7 | pending | cherry-pick candidates |
| AIGER robustness minors | I2 §12.1–12.5 | pending | |
| Decorator re-fixes | I 6.2–6.11 | pending | |
| Output-name collision + abc output validation | I2 §8.1, §8.2 | pending | |
| Decorator minors (selected) | I2 §8.3–8.13 | pending | |

## Phase 5 — FSM optimization

| Item | Source | Status | Notes |
|---|---|---|---|
| Walker/evaluator re-fixes | I 6.1, 0.8 | pending | |
| Collapse-family design (outputs, observability, guards, post-check) | I2 §7.1–7.5 | pending | short design note first |
| FSM minors + real post-wrapper test | I 6.13, 6.14, I2 §7.6–7.9, §15.3 | pending | |

## Phase 6 — arithmetic, multipliers, cores

| Item | Source | Status | Notes |
|---|---|---|---|
| Arithmetic re-fixes + residues | I 7.1–7.12, I2 §9.4–9.6 | pending | |
| Dispatch: subtractor gate, operator typing, fused typing→gate | I 7.8, I2 §9.1–9.3 | pending | order matters |
| Multiplier re-fixes + SM guards + seeding (all 3 generators) | I 8.1–8.10, I2 §10.1–10.7 | pending | |
| Core fixes | I 9.1–9.4, I2 §11.1–11.8 | pending | |
| Test repairs | I2 §15.1, §15.2, §15.4–15.8 | pending | |

## Phase 7 — simplify (deferred)

| Item | Source | Status | Notes |
|---|---|---|---|
| Const folding + width fixes re-applied | I 0.7, 1.1 | pending | |
| Signedness-preserving rewrites + no in-place mutation | I2 §2.1, §2.8 | pending | |

## Phase 8 — docs, examples, tests-infra, packaging

| Item | Source | Status | Notes |
|---|---|---|---|
| Examples/scripts (complete 10.4 incl. guards) | I 10.3, 10.4, I2 §13.1–13.9 | pending | |
| Docs corrections/regeneration | I 10.1, 10.7, 9.4, I2 §14.1–14.7 | pending | |
| Packaging/metadata/extras | I 10.2, 10.5, 10.6, I2 §16.1–16.8 | pending | |
