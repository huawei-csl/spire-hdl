# Design DB — a verification-gated library of implementations

> **Status: registration + verification gate + selection decorator + CLI in place.** The simulation
> verification tiers (`spire db verify`, sequential slots) are coming next and will extend this page.

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
print(res.design_id, res.metrics["transistors_heavy"], res.metrics["intrinsic"])
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
- **`metric`** — the measurement system the objective is evaluated in: `"aig"` (intrinsic
  nodes/depth), `"transistors"` (heavy-pipeline Yosys estimate), or a technology (e.g. `"asap7"`,
  once PPA scoring has run). `metric=None` resolves deterministically: technology → transistors →
  aig. Constraints are evaluated in the same system.
- The resolved `(selected_id, objective, metric)` is recorded in the manifest, so builds are
  reproducible and auditable.

## The CLI

Installed as the `spire` console script (also `python -m spire.design_db …`):

```
spire db init                      # create (or print) the DB root
spire db ls                        # slots: name, class, #designs, key, selected
spire db show adder8 --pareto      # one slot as JSON (spec, verification, designs, Pareto front)
spire db insert cand.v --slot adder8 --source handwritten [--budget 300]
```

`show`/`ls` are read-only (they never create a DB); `insert` exits 2 with a `REJECTED (...)` line
on any verification failure. Slots are addressed by manifest name, full key, or a unique key
prefix (≥ 8 chars).

## Where the DB lives

Resolution order (zero-config): explicit `db=`/`--db` → `$SPIREHDL_DB_PATH` → the nearest existing
`design_db/` directory upward from the cwd → **auto-create `./design_db`** (write paths only; a
one-line note is printed on first creation).

## On-disk layout (schema `v1`)

```
design_db/v1/<spec_key>/        # spec_key = sha256(structural AAG + port spec)
    spec.json                   # name, ports, class, golden_sha, source_ref, registered_from
    golden.v                    # the golden reference candidates are verified against
    starting_point.py           # the decorated function's captured source (fidelity-tagged)
    verification.json           # the frozen verification (absent = unverified, inserts refused)
    designs/<source>:<hash>/    # one admitted implementation
        design.v                #   the implementation
        design.aag              #   precomputed splice input (structural AIG)
        metrics.json            #   {intrinsic: {aig_nodes, aig_depth, aig_latches}, transistors_heavy}
        provenance.json         #   {source, created, verification: {tier, method, verdict, budget_s}}
    index.json                  # roll-up {design_id -> {struct_hash, metrics, source, created}}
design_db/v1/manifest.json      # {registered name -> {spec_key, class, n_designs, selected_id, …}}
```

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
| 1 | auto sim harness (randomized + directed stimulus, golden-simulated outputs) | sequential; combinational where the caller chose sim | planned |
| 2 | authored stimulus (human or dv agent), coverage-gated | protocol-heavy sequential | planned |

- Combinational slots get Tier-0 CEC **by default** at registration; sequential slots register
  fine but stay **unverified** (inserts raise `SlotUnverified`) until a sim tier is frozen.
- CEC runs under a bounded budget (`budget_s`, default 120 s, per candidate at insert). A timeout
  is a clean failure: `CECTimeout` — *"CEC timed out after 120 s. Options: --budget <t> | --auto |
  --stimulus <file>"* — and the slot's verification is unchanged until the caller picks the next
  rung. Requesting CEC for a sequential slot raises `CECInapplicable` (no register mapping).

## API summary

| Symbol | What it does |
|---|---|
| `@from_design_db(objective=, metric=, pin=, fill=, db=)` | The selection decorator: register → select → splice; miss ⇒ original logic. |
| `register_slot(module_or_component, db=None, name=None) -> spec_key` | Register a slot (idempotent): spec + golden + default verification + manifest entry. |
| `insert_design(spec_key, design, *, source, db=None, design_py=None, budget_s=None, provenance=None) -> InsertResult` | The gate: verify → dedup → stamp metrics → record provenance → admit atomically. `design`: spire `Component`/`Netlist`, Verilog path, or Verilog text. |
| `select_design(spec_key, *, objective=, metric=, pin=, sources=, record=)` | Deterministic selection → `SelectionResult` (or None on an empty slot). |
| `pareto_front(spec_key, objectives=("area","delay"), metric=None)` | The non-dominated set. |
| `constrained / weighted / lexicographic` | Objective combinators. |
| `resolve_db_root(db=None)` / `DesignDB` | DB-root resolution / low-level store handle. |
| `cec_check(design_v, golden_v, workdir, budget_s=…)` | Standalone Tier-0 CEC (raises on non-PASS). |
| `detect_class(module)` | `"combinational"` / `"sequential"` (register scan). |
| Exceptions | `VerificationFailed`, `CECTimeout`, `CECInapplicable`, `SlotUnverified`, `VerificationError`, `DesignDBError`. |

`import spire.design_db` is dependency-light: pyosys/aigverse are only imported when an insert or
splice actually needs them.

## Coming next (will extend this page)

- **Sim verification tiers**: `spire db verify --slot <key> [--cec [--budget]| --auto | --stimulus]`
  — the auto sim harness + human-authored stimulus for sequential and CEC-infeasible combinational
  slots (coverage-measured, frozen).
- **Per-technology PPA scoring** (`db score`, rtlscout-side) enabling `metric="asap7"` selection.
- Agent fillers: campaign (`rtlscout fill-db` / the `fill=` hook) and orchestrator/subagent flows.
