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
| Baseline run recorded; bug shapes as strict xfails | I 0.1, 0.2, 0.3-0.6, 13.2; I2 §1.1-§1.6 | applied | 62 pass / 33 strict-xfail; leaf: mixed ± * and any-signed ordered cmp diverge (= 0.1/0.2 exactly); (s4-s4)<s5 conformant-by-accident — guards the §1.3 trap; AIGER: unsigned fuzz clean, signed xfail (0.3-0.6) |
| `simplify=` docstring caveat (experimental until Phase 7) | I 0.7/1.1, I2 §2.1 | pending | one line |

## Phase 1 — expression signedness & width-correct emission

| Item | Source | Status | Notes |
|---|---|---|---|
| Width-isolation of non-leaf operands at op construction | I2 §1.1–1.2, §1.6, I 13.2 | pending | design center; force-share first, narrow later |
| Signed compares re-fix on isolated operands | I 0.1, I2 §1.3 | pending | |
| Mixed-arith alignment re-fix | I 0.2, I2 §1.4 | pending | |
| Ternary emission per charter C1 | I2 §1.5 | pending | |
| Const validation + boundary-literal emission | I2 §2.2, §2.7, R P9, charter C4 | pending | |
| Negative shift amounts rejected | I2 §2.6, charter C3 | pending | |
| Identifier legality | I2 §2.4, charter C6 | pending | |
| Register `init=` cone collection | I2 §2.3 | pending | |
| `flat_emit` scope decision | I 13.2, I2 §1.6 | pending | |

## Phase 2 — core structures

| Item | Source | Status | Notes |
|---|---|---|---|
| Control flow: chain + condition-state scoping | I 2.3, I2 §4.1–4.3 | pending | |
| Control flow minors | I2 §4.4–4.7 | pending | |
| Sharing/CSE re-fixes | I 1.2–1.6, 2.6, I2 §2.5 | pending | |
| Component/IO re-fixes | I 2.1, 2.2, 2.5, 2.7, 4.5 | pending | |
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
