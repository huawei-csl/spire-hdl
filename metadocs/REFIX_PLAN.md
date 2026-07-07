# Clean re-fix plan — branch `fix/issue-review-v2` (from `30cdb2a`)

Goal: re-apply everything worth keeping from `fix/issue-review` (104 commits, rated per-fix in REVIEW_GUIDE.md) plus
the round-2 findings (ISSUES2.md), as a **clean, minimal, supervised** series — starting fresh from `30cdb2a`, the
exact fork point of the old branch from `main`. The old branch is reference material only.

Inputs and how they compose:
- **ISSUES.md** — what was wrong at the baseline (all of it exists again at `30cdb2a`, unfixed).
- **REVIEW_GUIDE.md** — how the old branch fixed each item, its severity/cleanliness rating, and per-fix refix advice
  (`adopt as-is` / `adopt, trim` / `reimplement smaller` / `rethink`). Not up to date: it predates round 2.
- **ISSUES2.md** — 127 new findings. For this plan they split three ways:
  - *pre-existing bugs* → new work items, slotted into phases below;
  - *gaps in old fixes* (0.1/0.2 nesting, 5.3 rst-high branch, 7.5 missed generators, R4 catch, P9 scope, P5
    atomicity, P1 entry points, 10.4 sub-items, 6.16 checks) → folded into the corresponding re-fix's definition of
    done, so the v2 fix covers them from the start;
  - *regressions the old branch introduced* (§5.1 nested-IO names, §6.4 async-ROM guard, §10.1 `_reseed` crash,
    §15.1 assertion-free test, §13.1 broken import, §14.7 dead link) → design-in guards: the v2 version of each fix
    ships with the test that would have caught the regression.
- **issues2_repros/** (gitignored, in the main checkout) — runnable evidence for every ISSUES2 item; source for the
  permanent differential harness (Phase 0) and for per-fix regression tests.

---

## 0. Semantics charter — what is the reference?

Decision (to be committed as `docs/README_semantics.md` in Phase 0 so every fix can cite it):

**A. Combinational / expression value semantics: the spire IR + Python simulator is the specification.**
Every IR node evaluates self-determined at its own `typ.width`/`typ.signed` and wraps; parents extend the wrapped
value. Rationale: (1) the AIGER backend already implements exactly this — two of three backends agree today, the
emitted Verilog is the odd one out; (2) all user code and the entire test suite are written against the simulator;
(3) "each node wraps at its declared width" is a coherent, teachable rule, while IEEE context-determination is the
surprising one. Consequence: **Verilog emission must be changed to implement spire semantics** (width isolation of
compound operands), never the other way around.

**B. Temporal / reset / power-on / memory-array semantics: real hardware (the emitted RTL) is the specification.**
The simulator must model what the RTL does: pre-edge sampling (3.1), registers that hold through reset when their RTL
has no reset arm (ISSUES2 §3.1, §6.1), reset-arm priority, read-under-write modes. Where a 2-state simulator cannot
represent RTL reality (power-on X before the first reset — 3.4/§3.5; X on out-of-range reads — §6.3), we adopt and
*document* a convention (reset-first flows; OOR reads = 0 in both sim and via_reg) rather than pretend equivalence.

**C. Case-by-case list** (each gets an explicit decision recorded in the charter when its fix lands):
1. `Ternary` typing — spire says signed-if-either-branch-signed; IEEE says both. **Recommend: keep spire's rule**
   (changing IR types silently changes existing users' simulations) and make emission implement it (isolate the mux
   into a typed named wire). Revisit only if isolation proves ugly.
2. Signed `>>` — spire is a logical shift on signed values (AIGER agrees). Keep + document; arithmetic shift is a
   feature request, not a bug fix. Only the context-width divergence gets fixed.
3. Negative constant shift amounts (§2.6) — meaningless in both backends → construction-time `ValueError`.
4. Out-of-range `Const(v, typ)` (§2.7, P9) — validate at construction (error), since `as_expr` paths are already
   safe; masking hides user bugs.
5. `composite == composite` (§5.4) — either build a packed hardware compare (consistent with `Expr.__eq__`) or raise
   (consistent with `Expr.__bool__`). Decide when touching composites in Phase 2; leaning "build hardware compare".
6. Verilog identifier legality (§2.4) — sim accepts any Python name; RTL cannot. Spec: emission-time names are
   sanitized (keyword/lexical check, deterministic fallback), never a user-visible sim behavior change.

---

## 1. Working rules (the "cleanly this time" corrective)

1. One issue cluster per commit; minimal diff; message cites the ISSUES / ISSUES2 / REAUDIT ids and the REVIEW_GUIDE
   advice it followed (`adopt` / `trim` / `reimplement`).
2. **Definition of done per fix**: (a) the ISSUES/ISSUES2 repro turned into a small failing-then-passing pytest;
   (b) the ISSUES2 gap-list and regression-list for that fix explicitly checked; (c) full suite green; (d) for
   anything touching emission — the Phase-0 differential gates green.
3. Cherry-pick from the old branch **only** where REVIEW_GUIDE says `adopt as-is` *and* ISSUES2 found no gap in that
   fix (mostly the AIGER 0.3–0.6 family, several small MINOR fixes); everything rated `adopt, trim`,
   `reimplement smaller`, or `rethink` is rewritten by hand at the size REVIEW_GUIDE suggests.
4. Tests proportionate to the fix (REVIEW_GUIDE's cleanliness lesson): prefer one seeded differential/property test
   over sprawling case enumerations; put shared harness code in one place (`testing/differential/`).
5. Never mark an item fixed in the tracking table without the test from rule 2(a).
6. Tracking: the `metadocs/REFIX_STATUS.md` table (issue id → commit → status), replacing scattered status edits in
   ISSUES.md/ISSUES2.md (those two files stay frozen as the source record). This plan and the status table live in
   `metadocs/` and are committed via .gitignore whitelist entries; the rest of `metadocs/` stays ignored. The
   semantics reference (`docs/README_semantics.md`) is a normal committed doc and deliberately carries no
   plan/branch/process references.
7. Interaction protocol: Claude applies each change set in the worktree, updates REFIX_STATUS.md in the same change
   set, and hands over with a suggested one-line commit message (no phase/plan references — cross-referencing lives
   in the status table); Felix reviews in his own window and makes every commit himself. Next change set starts only
   after the review/commit (or feedback).

---

## 2. Phases

Ordered per priority: high-use surfaces first — basic expressions (signedness/extension + wrong emission) and core
structures — then memories, backends, FSM/decorators, arithmetic/cores; `simplify=True` deliberately late; docs and
packaging last. Each phase = one reviewable PR-sized unit with an acceptance gate.

### Phase 0 — Scaffolding (no product-code changes)

- Commit the semantics charter (`docs/README_semantics.md`, §0 above).
- Port the round-2 harness into `testing/differential/` as permanent, seeded, fast tests:
  - the two calibrated IEEE-1364 evaluators (merge into one; keep the leaf-operand calibration battery — ~20k
    exhaustive vectors — as *the* invariant test);
  - the nested-shape battery from ISSUES2 §1 (currently `issues2_repros/emission/nested.py`, `core-ir/test_inline_ctx.py`);
  - a fixed-seed random expression fuzz (sim vs evaluator vs AIGER reload), ~200 circuits, seconds not minutes;
  - optional CI-skipped yosys-elaboration crosscheck (pyosys) for the same battery.
- Baseline run: full suite + harness at `30cdb2a`; record which harness tests are red (they encode the §1 bugs) and
  mark them `xfail(strict=True)` so Phase 1 flips them to green explicitly.
- Interim safety note in `to_verilog(simplify=...)` docstring: simplify is experimental until Phase 7 (one line).

Gate: suite green (modulo recorded baseline xfails); harness red exactly on the known-bug shapes.

### Phase 1 — Expression signedness & width-correct emission  ← the user-priority core

Scope: ISSUES 0.1, 0.2 (redesigned, not re-applied) + ISSUES2 §1.1–1.6, §2.2, §2.4, §2.6, §2.7/P9, §2.3 + old 1.10.

1. **Width-isolation mechanism** (the design center, replacing the old point-fixes): at op construction, any operand
   that is a non-leaf expression gets a width boundary — extend `_maybe_share(force_share=True)` to the self/left
   operand of every `Op2`/`Op1`, `op_cmp` compounds even at equal widths, `Ternary` branches, extension-concat
   payloads, and shift amounts. Start with "share every non-leaf operand" (correct by construction, kills 13.2 as a
   side effect); once the Phase-0 harness pins behavior, optionally narrow with a precise predicate (share only when
   self-determined size or signedness could diverge from context) to reduce wire count. Measure emitted-netlist size
   before/after on the shipped examples; the narrowing step is QoR polish, not correctness.
2. Re-apply **0.1** (signed compares → `$signed` pairs) and **0.2** (mixed-arith alignment) on top — with isolated
   operands both become small and obviously correct; the old fixes' nesting gaps (§1.3, §1.4) are structurally gone.
3. **Ternary**: implement charter decision C1 (typed named wire for mixed-sign muxes).
4. **Const emission**: boundary negatives emitted as signed-base pattern literals (§2.2); out-of-range values
   rejected at construction (C4, subsumes P9's emission masking); bool-path consistency (§2.7).
5. **Identifier legality** (§2.4 + old 1.6 adjacency): keyword/lexical validation in `sanitize_signal_name`, the
   `Netlist` name/port API, and explicit-name paths; deterministic fallback names.
6. **Register `init=` cones** collected by `_SignalCollector` (+ exclude self-colliding inferred names) (§2.3).
7. `flat_emit`: after (1), decide whether it still has a correctness-safe niche; if not, restrict it to width-1
   boolean cones (its only in-tree use) and document (13.2, §1.6).

Gate: Phase-0 xfails all flip green; 100% sim == evaluator == AIGER on leaf battery, nested battery, and fuzz;
yosys crosscheck green; old branch's signedness differential tests (ported) green.

### Phase 2 — Core structures: control flow, sharing/CSE, composites, registers, simulator core

- **Control flow**: 2.3 reimplemented (chain scoping per REVIEW_GUIDE advice) + ISSUES2 §4.3 (scope
  `_ConditionState` at component/decorator boundaries), §4.1 (composite RHS under `if_` via `to_bits()`), §4.2
  (`Signal.assign` under active conditions: route through the conditional wrapper), §4.4–§4.7 minors.
- **Sharing/CSE**: 1.2, 1.3, 1.4 (per guide), 1.5, 1.6, 2.6; per-netlist shared-counter reset (§2.5).
- **Component/IO**: 2.1, 2.2, 2.5, 2.7; dict-IO names (4.5); port naming done once, correctly at depth — the 4.8
  equivalent ships with ≥2-level-nesting and record-in-array tests (kills §5.1 before it exists).
- **Composites**: 4.1–4.4, 4.6 re-fixes + §5.2 (real representability check), §5.3, §5.4 (charter C5), §5.5–§5.10.
- **Simulator core**: 3.1, 3.2, 3.3 re-applied with the §3.1 rst-high branch included this time (one mechanism:
  `_sim_no_reset` respected in *both* `reset()` and `_compute_next_state`); 3.4 + §3.5 documented per charter B;
  §3.2, §3.3, §3.4, §3.6 minors.

Gate: full suite + ported control-flow semantics matrix + composite differential sweeps (FixedPoint vs Fraction,
small formats) green.

### Phase 3 — Memories, primitives, interfaces

- 5.1 re-done with the write-less/unregistered exemption (`needs_clock` counts memories with write ports/reset arm
  or any register — avoids §6.4 while keeping the guard).
- 5.2 (mask forwarding), 5.3 complete (both paths), 5.4–5.8; §6.1 (tag via_reg data cells `_sim_no_reset`), §6.2
  (mask init/reset literals to width), §6.3 (align via_reg OOR default to canonical 0 + charter-B documentation),
  §6.5–§6.7.

Gate: ported FIFO/memory/RAM differential harness (deque/dict golden models, mid-run resets, OOR vectors,
non-pow2 depths) green; emitted templates re-reviewed once.

### Phase 4 — Backends: AIGER + optimization decorators

- **AIGER**: 0.3–0.6, 3.5, 3.7 — mostly `adopt as-is` cherry-picks (round 2 found this layer clean and these fixes
  good); then §12.1 (failure atomicity), §12.2 (named error for memories), §12.3 (raise on undriven), §12.5
  (existence check in `from_aig_file` → close the remaining P1-family entry point), §12.4 reader hygiene (selective).
- **Decorators**: 6.2–6.11 per guide + §8.1 (mangle internal output names / raise on collision — with the `y`
  parameter test), §8.2 (validate abc output before pyosys; longer-term P1-style isolation), then selected minors
  (§8.3–§8.9, §8.11) batched by file.

Gate: 600-circuit AIGER roundtrip property test (seeded, trimmed to ~1 min) green; decorator cache-key +
collision tests green.

### Phase 5 — FSM optimization

- Re-apply 6.1 + 0.8 (walker boundary, evaluator signedness) per guide.
- The §7 design work (new, needs a small design note before coding):
  - default `outputs=`: auto-discover via `find_fsm_outputs` (or hard-require the argument) — kills §7.2;
  - observability rule: an output whose cone hits a non-state register uses that register's driver as the
    observable, else refuse+warn — kills §7.1;
  - multi-register / shared-class guard (reuse `find_state_register`'s detection) — kills §7.3;
  - post-rewrite verification: re-extract the transition table after `apply_encoding` and compare against the
    canon-mapped original — kills §7.4 and acts as a permanent safety net;
  - exhaustive ladder gates on P(2^width, n) — kills §7.5.
- 6.13, 6.14 per guide; §7.6 (catch-all with rollback), §7.7, §7.8, §7.9.
- Replace the vacuous post-wrapper test with a real wrapped-vs-unwrapped differential under toggling stimulus
  (§15.3) and port the seeded 60-machine e2e fuzz as a slow-marked test.

Gate: FSM suite + e2e fuzz green; the three §7 collapse repros (in issues2_repros/fsm/) turned into tests and green.

### Phase 6 — Arithmetic, multipliers, cores

- 7.x per guide (7.3 DAZ, 7.4 chain detection, 7.7 SM −0, …) with §9.5 residues built in (regenerate the packaged
  CSV, docstring counts).
- Dispatch layer: §9.1 (subtractor gate — the 7.8 fix applied to all three siblings at once), §9.2 (operator-path
  result typing), §9.3 in the right order (type `fused_inner_product` results SInt for signed encodings first, then
  narrow the fusion gate to exclude the synthetic constant → restores signed-chain fusion *correctly*).
- 8.x per guide (Karatsuba 8.1/8.2, 8.3–8.9) + §10.2 square-width guards on the five SM wrapper classes, §10.3,
  §10.4; vector generators seeded in ALL THREE places (7.5 done completely — §10.1/§10.6/§15.4) with signed-extreme
  vectors added; §10.5, §10.7.
- Cores: 9.1, 9.2 per guide + §11.1 (Winograd unsigned — proper widening instead of same-width reinterpret), §11.2
  (`_to_signed` in precomputed-B), §11.3 (arbitrary-K correction, shared with fused.py — extract the helper), §11.4
  (constructor validation / encoding forwarding), §11.5 xfail documentation, §11.7, §11.8.
- Test repairs alongside: §15.1 (booth-share asserts), §15.2 (commit the fingerprint capture script or delete the
  dead test), §15.5 (narrow the xfail marker — safe once vectors are seeded), §15.6–§15.8.

Gate: multiplier/core differential sweeps (small dims, signed extremes, all variants) green — ported from
issues2_repros as slow-marked seeded tests.

### Phase 7 — simplify (deliberately deferred)

- 0.7, 1.1 re-applied per guide + §2.1 (signedness-preserving `_fit_result`; identity rules re-typed) and the
  in-place-mutation fix (simplify a clone, or re-canonicalize the module after rewrites so sim/AIGER/emission agree
  post-call), §2.8.
- Until this phase: simplify stays opt-in with the Phase-0 docstring caveat. Safe to defer because the default path
  never runs it.

Gate: simplified-vs-unsimplified differential fuzz (both sim and emitted text) green; the Phase-0 caveat removed.

### Phase 8 — Docs, examples, tests-infra, packaging

- 10.x per guide, done completely this time (main guards actually added — §13.5/§15.9; moved-script imports fixed in
  the same commit — §13.1; the 6.16 check-side fixed with format-matched vectors — §13.2).
- Examples: §13.3 (aig_example call), §13.4 (slice), §13.6–§13.9.
- Docs: §14.1–§14.7 (regenerate benchmark tables from the shipped benches or mark them approximate; fix stale
  defaults/claims; the memories/via_reg and FSM sections rewritten against the new behavior).
- Packaging: §16.1 (ship requirements.txt in the sdist), §16.2/16.4 (extras: `sympy`, `dash`/`plotly`; demote
  eval-only heavyweights), §16.3 (lazy `helpers` import), §16.5–§16.8; 10.2, 10.5, 10.6 per guide.

Gate: docs-block extraction run (issues2_repros/docs/extract_blocks.py) fully green; import sweep + wheel build green.

---

## 3. What deliberately does NOT carry over from the old branch

- The four regression-carrying fix versions (4.8 naming, 5.1 guard, `4b6264a`'s misplaced `_reseed`, `2fe5c47`'s
  assertion-free test) — replaced by the designs above.
- Status-tracking edits inside ISSUES.md/REAUDIT.md — replaced by `REFIX_STATUS.md`.
- Anything REVIEW_GUIDE rated cleanliness ≤ 3 gets reimplemented from the advice line, not trimmed.

## 4. Open decisions before Phase 1 starts

1. Branch name `fix/issue-review-v2` and worktree at `/scratch/farnold/eda_package/spire-hdl-refix` — RESOLVED:
   confirmed; the branch sits directly on origin/main (`30cdb2a`).
2. Charter C1 (Ternary keeps spire's either-signed rule) — confirm or flip to IEEE both-signed.
3. Phase-1 mechanism: start with force-share-all-non-leaf (recommended), narrow later — or require the precise
   predicate from day one (leaner Verilog, more design risk up front).
4. Cherry-pick allowance for `adopt as-is` + no-ISSUES2-gap commits (~2 dozen, mostly AIGER + small MINORs) — ok?
5. Plan/status doc location — RESOLVED: both live in `metadocs/` and are committed (whitelisted in .gitignore);
   the semantics reference is committed at `docs/README_semantics.md` as a process-free reference document.
