# Design DB — a verification-gated library of implementations

> **Status: registration, the verification gate (CEC + sim tiers), the selection decorator, and the
> CLI are in place.** Remaining: per-technology PPA scoring and the agent fillers (rtlscout-side).

`spire.design_db` is a content-addressed store of **correct implementations of a subcircuit**. For
each subcircuit — a *slot*, keyed by its golden specification — the DB holds any number of
implementations, each **proven correct against the slot's golden** before it is admitted, and each
carrying metric vectors and provenance. Producers (optimization tools, agents, humans) *insert*
through the verification gate; consumers *select* by objective. Generation and selection are fully
decoupled: filling a slot once lets every later build pick from it.

## Quick start — the decorator

```python
from spire import Component, IORecord, Input, Output, UInt
from spire.design_db import from_design_db

@from_design_db(objective="area")
def mac(a, b, c):
    return a * b + c

class Top(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(8)), b=Input(UInt(8)), c=Input(UInt(16)),
                           y=Output(UInt(17)))
        self.elaborate()
    def elaborate(self):
        self.io.y <<= mac(self.io.a, self.io.b, self.io.c)   # selected implementation spliced here
```

`@from_design_db` is a **pure reader** — it never generates and never spends budget:

1. The call is traced (exactly like `@abc_optimized`) and the slot is **registered automatically**:
   golden + spec + default Tier-0 CEC verification + the function's source as the starting point.
2. The best admitted implementation (by `objective`/`metric`) is selected and **spliced** in place
   of the function.
3. **Miss ⇒ the original logic** (one-line note; a build never fails because a slot isn't filled).

Parameters: `objective=` (what to minimize — see Selection), `metric=` (measurement system),
`pin=` (an exact `design_id`; missing ⇒ error — a reproducibility lock), `fill=` (opt-in callable
invoked once on miss to generate, then re-select), `db=` (explicit DB root).

### First compile with no DB — the bootstrap path

Decorating costs nothing up front: compiling a design that uses `@from_design_db` when no DB exists
(or the slot is empty) succeeds unchanged.

1. The call is traced and the slot **registers** — resolution finds no DB, so `./design_db` is
   **auto-created** (one printed note; write paths may create — read-only commands never do).
2. The slot gets `spec.json`, `golden.v`, the default Tier-0 CEC `verification.json`,
   `starting_point.py` + `source_ref` (captured best-effort from the defining file), and a manifest
   entry.
3. Selection finds an empty slot → **miss ⇒ original logic**: one note
   (`slot … has no admitted design — using the original logic for f`), and `f`'s own body is used
   inline — the emitted circuit is **structurally identical to the undecorated function**
   (test-asserted via AAG equality). No error, no CEC, no yosys, no budget spent.
4. Later fills (`spire db seed`, `spire db insert`, a `fill=` hook, or an external filler such as
   an RTLScout campaign) populate the slot; the **next compile splices** the selected
   implementation.

The only variants that behave differently on an empty slot: `pin=` errors (a broken
reproducibility lock, by design), and `fill=` fires the generate hook on the miss.

## Quick start — filling a slot

```python
from spire.design_db import insert_design, register_slot

key = register_slot(Adder(), name="adder8")      # or the decorator registers it on first build

candidate = """
module cand_add(input [7:0] a, input [7:0] b, output [8:0] s);
  assign s = {1'b0, a} + {1'b0, b};
endmodule
"""
res = insert_design(key, candidate, source="handwritten")   # verified, then admitted
print(res.design_id, res.metrics["transistors"]["metrics"], res.metrics["aig"]["metrics"])
```

An incorrect candidate raises `VerificationFailed` and leaves no trace in the DB; a candidate whose
ports don't match the slot spec is rejected with a clear message. Inserting a structurally
identical design again returns `deduped=True`. Spire-native producers can pass a
`Component`/`Netlist` directly — `insert_design(key, MyOtherAdder(), source="spire")` lowers it
internally.

## Selection

Selection is deterministic and instant — pure functions over the metric vectors stamped at insert:

```python
from spire.design_db import select_design, constrained, weighted, lexicographic, pareto_front

select_design(key, objective="area")                                   # argmin
select_design(key, objective="delay", metric="aig")                    # explicit system
select_design(key, objective=constrained(minimize="area", subject_to={"delay": 500}))
select_design(key, objective=weighted({"area": 1.0, "delay": 0.1}))
select_design(key, objective=lexicographic(("area", "delay")))
select_design(key, pin="spire:1a2b3c4d5e")                             # reproducibility lock
pareto_front(key)                                                      # non-dominated set
```

- **Objectives**: `area | delay | adp | edap` (or a combinator above).
- **`metric`** — the measurement system the objective is evaluated in: `"aig"` (structural
  nodes/depth), `"transistors"` (heavy-pipeline Yosys estimate), or a technology (e.g. `"asap7"`,
  once PPA scoring has run). `metric=None` resolves deterministically: technology → transistors →
  aig. Constraints are evaluated in the same system.
- The resolved `(selected_id, objective, metric)` is recorded in the manifest, so builds are
  reproducible and auditable.

**How a design stores its metrics.** Each design's `metrics.json` holds one *self-describing block
per measurement system* — its raw `metrics` plus an `objectives` map saying which field plays each
axis (a bare field is local; a `system.field` path borrows from a sibling system). The gate stamps
`aig` and `transistors`; the reader (`metric_value`) is fully data-driven — no system is special-cased:

```json
{
  "aig":         { "metrics": {"aig_nodes":105, "aig_depth":12, "aig_latches":8},
                   "objectives": {"area":"aig_nodes", "delay":"aig_depth"} },
  "transistors": { "metrics": {"transistors_heavy":202},
                   "objectives": {"area":"transistors_heavy", "delay":"aig.aig_depth"} }
}
```

`transistors` has no timing of its own, so its `delay` axis explicitly **borrows** `aig.aig_depth`
rather than duplicating the number. `adp` is derived (`area·delay`) whenever a system doesn't map
it; `edap` (and any other axis) is available only if a system maps it. The `objectives` map is
tooling-written and identical across a slot's designs — one interpretation per slot.

### Enriching designs with more metrics (`annotate`)

The gate stamps only what yosys gives it (the `aig` and `transistors` systems). Richer
per-technology PPA is produced *outside* spire — a real ASAP7 flow, another tool, or by hand — and
attached to a design through **`annotate`** (API and `spire db annotate` share the name) as one more
self-describing system the design can then be selected in:

```
spire db annotate --slot adder8 --design agent:9f3c1a2b7d --tech asap7 area=118.3 delay=94.6
```
```python
from spire.design_db import annotate
annotate(key, "agent:9f3c1a2b7d", tech="asap7", values={"area": 118.3, "delay": 94.6})
```

A design's `metrics.json` **before** (as the gate left it):

```json
{ "aig":         { "metrics": {"aig_nodes":105, "aig_depth":12, "aig_latches":8},
                   "objectives": {"area":"aig_nodes", "delay":"aig_depth"} },
  "transistors": { "metrics": {"transistors_heavy":202},
                   "objectives": {"area":"transistors_heavy", "delay":"aig.aig_depth"} } }
```

and **after** the annotate above (the `asap7` block is added; the gate's blocks are untouched):

```json
{ "aig":         { "metrics": {"aig_nodes":105, "aig_depth":12, "aig_latches":8},
                   "objectives": {"area":"aig_nodes", "delay":"aig_depth"} },
  "transistors": { "metrics": {"transistors_heavy":202},
                   "objectives": {"area":"transistors_heavy", "delay":"aig.aig_depth"} },
  "asap7":       { "metrics": {"area":118.3, "delay":94.6},
                   "objectives": {"area":"area", "delay":"delay"} } }
```

Now `select_design(key, objective="area", metric="asap7")` (or `@from_design_db(metric="asap7")`)
ranks on the ASAP7 area. `annotate` builds the block's `objectives` as the identity over the
standard axes present in `values` (`area | delay | adp | edap`); other numeric keys are stored but
not selectable; `--raw <file>` optionally stashes the full tool-stats blob under `.raw`. Reserved
system names (`aig`/`transistors`/…) are refused, re-annotating a technology needs `--force`, and
the map must agree with any sibling design's for the same system — measurements are commitments and
a slot keeps one interpretation. Spire owns the write (`metrics.json` is authoritative; the
`index.json` cache re-derives from it), so producers just hand it the numbers; RTLScout's `db-score` is exactly such a producer
(its ASAP7 pipeline → `annotate`).

> **Cover the whole slot.** A technology counts as "present" once *any* design in the slot has it,
> and a design missing that system is ineligible for selection in it. Annotate **all** of a slot's
> designs for a technology (the seeded `original:*` included) before selecting on it, or keep
> `metric=` pinned — otherwise the default resolver may switch to a system that excludes the floor.

## The CLI

Installed as the `spire` console script (also `python -m spire.design_db …`):

```
spire db init                      # create (or print) the DB root
spire db ls                        # slots: name, class, #designs, key, selected
spire db show adder8 --pareto      # one slot as JSON (spec, verification, designs, Pareto front)
spire db insert cand.py --slot adder8 --source agent:rtl-subcircuit   # spire design: check + admit
spire db insert cand.v --slot adder8 --source handwritten [--budget 300]   # Verilog: check + admit
spire db verify cand.py --slot adder8  # advisory: run the set oracle, no admit (PASS/FAIL)
spire db verify cand.v --slot adder8   # (both commands take .py or .v/.sv)
spire db seed --slot adder8            # insert the slot's own golden as the baseline candidate
spire db annotate --slot adder8 --design 9f3c1a2b7d --tech asap7 area=118.3 delay=94.6 [--force]
spire db set-verification --slot mypipe --auto [--vectors 256 --seed 0 --sim-budget 300]
spire db set-verification --slot mypipe --stimulus stim.py --check   # dry-run the generator (no freeze)
spire db set-verification --slot mypipe --stimulus stim.py     # authored stimulus (Tier 2)
spire db set-verification --slot mypipe --stimulus stim.py --author agent:rtl-dv-prep   # attribution
spire db set-verification --slot adder8 --cec [--budget 300]   # (re)confirm CEC on a combinational slot
```

The three verification commands are orthogonal — one configures the oracle, two apply it:
- **`set-verification`** — *slot-level, once*: choose the method every later candidate is judged by
  (CEC, or a sim tier). Fail-and-choose: a bare call defaults to CEC for **combinational** slots and
  **errors with the options** for sequential ones (no safe default — CEC is inapplicable and sim
  needs stimulus); there is never an auto-fallback, and a frozen sim verification is immutable. It
  never checks a candidate — for CEC it is pure config, for sim it simulates the *golden* once to
  freeze the reference. `--stimulus` records the author in `stimulus_author` (default `human`;
  `--author agent:rtl-dv-prep` keeps agent-authored stimulus honestly attributed).
- **`verify <design>`** — *candidate-level, advisory*: run the set oracle against a candidate and
  report `PASS`/`FAIL` (exit 2 on fail), writing nothing. The dry run before committing.
- **`insert <design>`** — *candidate-level*: the same check, then **admit** on pass. `verify` and
  `insert` apply whatever `set-verification` established, so a slot judges every design the same way.

**Spire designs first; Verilog is the IR.** Both `insert` and `verify` accept a **python design
file** — a `.py` defining `build() -> Component/Netlist` — as the primary way in: the gate
elaborates it itself, the generated Verilog becomes the canonical `design.v` (all downstream
processing — dedup, metrics, splice — runs on Verilog, the DB's intermediate representation), and
the **python source is stored with the design**, correct by construction (the `.v` *is* its
elaboration; there is no way for the stored source to lie about the stored design). Multi-file
designs work: the entry's transitive *project-local* import closure (helpers under the entry's git
root; stdlib/site-packages/spire excluded) is vendored under the design's `source/` dir and listed
in provenance. Verilog inserts remain fully supported (external, handwritten, harvested
candidates) — they simply carry no python source.

**What "freeze" means.** Freezing turns the chosen sim verification into the slot's *permanent
acceptance oracle*. Concretely it (1) simulates the **golden** with the chosen stimulus and stores
the input + expected-output trace as `vectors.dat`, (2) stores the generated `tb.sv` that replays
this exact trace against any candidate, (3) makes both files read-only (0444), and (4) writes
`verification.json` (tier, method, `stimulus_author`, vector count). From then on **every insert
into the slot is judged against exactly this trace**. That is why a freeze is one-shot and
immutable: re-freezing would silently swap the yardstick that already-admitted designs were
measured with, making designs admitted before and after incomparable. (Tier-0 CEC is not
"frozen" in this sense — it stores only parameters, since equivalence against `golden.v` needs no
recorded trace; a CEC slot may still be switched **once** to a sim tier, after which the sim
freeze is final.)

Because the freeze is **one-shot**, iterate on the generator with `--check` first (API:
`check_stimulus(spec_key, stimulus_file=…)`): it loads the file and produces the masked vectors
against the slot's interface but simulates, writes, and freezes **nothing** — a failing generator
is a clean error, a weak-but-working one can still be improved. Freeze only when the stimulus is
worth committing.

**Seeding the baseline.** `spire db seed` (API: `seed_original(spec_key)`) admits the slot's own
golden as a design with `source="original"`. This gives selection a *floor* — argmin can never pick
something worse than the original — and gives reports/Pareto a baseline. When the slot has a
captured `starting_point.py` (decorator-registered slots), seed stores it with the seeded design as
its python source (`python_source: {kind: copied}` — the origin of the golden). Idempotent
(structural dedup). It is not done automatically at compile time (registration must stay
cost-free); fillers
are expected to seed before generating.

`show`/`ls` are read-only (they never create a DB); `insert` exits 2 with a `REJECTED (...)` line
on any verification failure. Slots are addressed by manifest name, full key, or a unique key
prefix (≥ 8 chars).

**Slot names are permanent bindings.** A manifest name maps to exactly one subcircuit, forever:
re-registering the same content under the same name is idempotent, several names may alias one
slot, but registering *different* logic under an existing name **raises** — change the name when
the behavior changes (rename the function, or pass `name=`: both `register_slot(m, name="…")`
and `@from_design_db(name="…")` take it; the decorator's default is the function's qualname).
This keeps `--slot <name>` references stable: a name can never silently start meaning a
different circuit.

## Where the DB lives

Resolution order (zero-config): explicit `db=`/`--db` → `$SPIREHDL_DB_PATH` → the nearest existing
`design_db/` directory upward from the cwd → **auto-create `./design_db`** (write paths only; a
one-line note is printed on first creation).

## On-disk layout (schema `v1`)

```
design_db/v1/<spec_key>/        # spec_key = sha256(structural AAG + port spec)
    spec.json                   # name, ports, class, clock, golden_sha, source_ref, registered_from
    golden.v                    # the golden reference candidates are verified against
    starting_point.py           # the decorated function's captured source (fidelity-tagged)
    verification.json           # the frozen verification (absent = unverified, inserts refused)
    tb.sv, vectors.dat          # sim tiers only: the frozen testbench + golden-simulated trace
                                #   (read-only once frozen)
    designs/<source>:<hash>/    # one admitted implementation
        design.v                #   the implementation (canonical IR — all processing runs on this)
        design.aag              #   precomputed splice input (structural AIG)
        design.py               #   python source when known (.py inserts: elaborated-by-the-gate;
                                #   seeded originals: copy of starting_point.py)
        source/<rel>.py         #   .py inserts only: the entry's project-local import closure
        metrics.json            #   {<system>: {metrics: {...}, objectives: {axis -> field}}, …}
        provenance.json         #   {source, created, verification: {...}, python_source: {kind, …}}
    index.json                  # DERIVED CACHE of the roll-up {design_id -> {struct_hash,
                                #   metrics, source, created}} — the designs/ dirs are the source
                                #   of truth; the cache self-heals on every read (inspection aid)
design_db/v1/manifest.json      # {registered name -> {spec_key, class, selected_id, …}} — names +
                                #   selection provenance (fcntl-locked writes); counts are derived
```

**Concurrency.** Admission is a single atomic directory rename, and the per-slot index is
*derived* from the admitted `designs/` dirs (each carries its own `provenance.json` +
`metrics.json`; `index.json` is only a materialized, self-healing cache). Concurrent inserts —
same slot or across slots, threads or processes — can therefore never lose a design. The
manifest's rare writes (name registration, selection recording) are serialized with an fcntl
lock. Parallel fills of one DB are supported; the only rule left is per-slot *courtesy*: one
filler per slot at a time beats two racing over the same search space.

Everything is plain JSON + Verilog — inspectable, diffable, committable. A committed DB makes
builds reproducible: selection reads it deterministically, nothing regenerates.

**Spec keys** hash the *structural AAG* (the numeric `AigerExporter` section) plus the port spec —
not the Verilog text, whose internal wire names depend on the build context. Identical logic with
an identical interface maps to one slot, wherever and whenever it is built.

**Starting points.** For decorator slots, registration captures the function's *current* source:
`starting_point.py` (runnable wrapper when the body is self-contained, honestly tagged
`fidelity: self-contained | fragment`) plus `source_ref = {file, qualname, line}` in `spec.json` —
a pointer to the defining file for full context (helpers, sub-components, imports).

## Verification: how correctness is gated

Every insert runs the slot's **frozen verification**; only passing designs are admitted. One
method-keyed ladder — the caller chooses, tooling only vetoes and fails loudly (no auto-fallback):

| Tier | Method | Applies to | Status |
|------|--------|-----------|--------|
| 0 | **CEC** vs `golden.v` (yosys → BLIF, `yosys-abc cec`) — formal, exhaustive | combinational only | **implemented** |
| 1 | **auto sim harness**: corners + seeded random stimulus (exhaustive for tiny combinational input spaces), **golden-simulated** outputs, frozen `tb.sv` + `vectors.dat` | sequential; combinational where the caller chose sim | **implemented** |
| 2 | **authored stimulus** (`--stimulus <file>`: a Python `generate(ports, n_vectors, seed)` generator), golden-simulated outputs | protocol-heavy sequential | **implemented (human path)** — the dv-agent filler is rtlscout-side |

- Combinational slots get Tier-0 CEC **by default** at registration; sequential slots register
  fine but stay **unverified** (inserts raise `SlotUnverified`) until a sim tier is frozen with
  `spire db set-verification --slot <key> --auto | --stimulus <file>`.
- CEC runs under a bounded budget (`budget_s`, default 120 s, per candidate at insert). A timeout
  is a clean failure: `CECTimeout` — *"CEC timed out after 120 s. Options: --budget <t> | --auto |
  --stimulus <file>"* — and the slot's verification is unchanged until the caller picks the next
  rung. Requesting CEC for a sequential slot raises `CECInapplicable` (no register mapping).
- **Sim-tier semantics: cycle-accurate trace equivalence** under the frozen stimulus. Expected
  outputs always come from simulating the golden (Verilator); a sequential candidate must match
  the golden's output trace cycle for cycle — a re-pipelined design with different latency is
  rejected, by design. The frozen `tb.sv` follows the rtlscout testbench contract
  (`TB_SUMMARY total=N errors=M`, `PASS`), and the DUT is bound by name via `-DDUT=<top>`.
- **Frozen means frozen**: `tb.sv`/`vectors.dat` are written read-only and a re-freeze is refused —
  it would silently change the oracle that admitted designs were checked against.

## API summary

| Symbol | What it does |
|---|---|
| `@from_design_db(objective=, metric=, pin=, fill=, name=, db=)` | The selection decorator: register → select → splice; miss ⇒ original logic. `name=` = manifest name (default: fn qualname; permanent binding). |
| `register_slot(module_or_component, db=None, name=None) -> spec_key` | Register a slot (idempotent): spec + golden + default verification + manifest entry. |
| `insert_design(spec_key, design, *, source, db=None, budget_s=None, python_copy=None, provenance=None) -> InsertResult` | The gate: verify → dedup → stamp metrics → record provenance → admit atomically. `design`: a **`.py` design file** (`build()` — elaborated here, source stored), a spire `Component`/`Netlist`, a Verilog path, or Verilog text. |
| `check_design(spec_key, design, *, db=None, budget_s=None) -> dict` | Advisory: run the slot's set oracle against a candidate (no admit, no write). The read-only sibling of `insert_design`; raises the same `VerificationError`s on failure. |
| `seed_original(spec_key, db=None, budget_s=None) -> InsertResult` | Insert the slot's golden as the baseline candidate (`source="original"`) — a selection floor; stores the slot's `starting_point.py` as its python source when present. |
| `annotate(spec_key, design_ref, *, tech, values, raw=None, force=False, db=None) -> dict` | Attach a per-technology metric block (`metrics[tech]`) to a stored design; makes `metric=<tech>` selectable. Writes `metrics.json`; the `index.json` cache refreshes from it. |
| `select_design(spec_key, *, objective=, metric=, pin=, sources=, record=)` | Deterministic selection → `SelectionResult` (or None on an empty slot). |
| `pareto_front(spec_key, objectives=("area","delay"), metric=None)` | The non-dominated set. |
| `constrained / weighted / lexicographic` | Objective combinators. |
| `resolve_db_root(db=None)` / `DesignDB` | DB-root resolution / low-level store handle. |
| `cec_check(design_v, golden_v, workdir, budget_s=…)` | Standalone Tier-0 CEC (raises on non-PASS). |
| `freeze_sim_verification(spec_key, *, stimulus_file=None, n_vectors=, seed=, sim_budget_s=)` | Build + freeze a sim verification (Tier 1 auto / Tier 2 authored). |
| `run_frozen_tb(spec_key, candidate_v, workdir, budget_s=…)` | The sim-tier gate check (raises on non-PASS). |
| `detect_class(module)` | `"combinational"` / `"sequential"` (register scan). |
| Exceptions | `VerificationFailed`, `CECTimeout`, `SimTimeout`, `CECInapplicable`, `SlotUnverified`, `VerificationError`, `DesignDBError`. |

`import spire.design_db` is dependency-light: pyosys/aigverse are only imported when an insert or
splice actually needs them.

## Coming next (will extend this page)

- **Per-technology PPA scoring** (`db score`, rtlscout-side) enabling `metric="asap7"` selection.
- Agent fillers: campaign (`rtlscout fill-db` / the `fill=` hook) and orchestrator/subagent flows,
  including the dv-agent for authored stimulus on protocol-heavy slots.
