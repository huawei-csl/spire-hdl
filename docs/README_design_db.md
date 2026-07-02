# Design DB — a verification-gated library of implementations

> **Status: core in place, API growing.** What's documented here (registration, the verification
> gate, inserts) is implemented and tested. The selection decorator (`@from_design_db`), the
> `spire db` CLI, and the simulation verification tiers are coming next and will extend this page.

`spire.design_db` is a content-addressed store of **correct implementations of a subcircuit**. For
each subcircuit — a *slot*, keyed by its golden specification — the DB holds any number of
implementations, each **proven correct against the slot's golden** before it is admitted, and each
carrying metric vectors and provenance. Producers (optimization tools, agents, humans) *insert*
through the verification gate; consumers *select* by objective. Generation and selection are fully
decoupled: filling a slot once lets every later build pick from it.

## Quick start

```python
from spire import Component, IORecord, Input, Output, UInt
from spire.design_db import register_slot, insert_design

class Adder(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(8)), b=Input(UInt(8)), s=Output(UInt(9)))
        self.elaborate()
    def elaborate(self):
        self.io.s <<= self.io.a + self.io.b

key = register_slot(Adder(), name="adder8")     # slot: spec.json + golden.v + default Tier-0 CEC

candidate = """
module cand_add(input [7:0] a, input [7:0] b, output [8:0] s);
  assign s = {1'b0, a} + {1'b0, b};
endmodule
"""
res = insert_design(key, candidate, source="handwritten")   # CEC-verified, then admitted
print(res.design_id, res.metrics["transistors_heavy"], res.metrics["intrinsic"])
```

An incorrect candidate raises `VerificationFailed` and leaves no trace in the DB. Inserting a
structurally identical design again returns `deduped=True` with the existing id. Spire-native
producers can skip the Verilog step entirely — `insert_design(key, MyOtherAdder(), source="spire")`
accepts a `Component`/`Netlist` and lowers it internally.

## Where the DB lives

Resolution order (zero-config): explicit `db=` argument → `$SPIREHDL_DB_PATH` → the nearest
existing `design_db/` directory upward from the cwd → **auto-create `./design_db`** (a one-line
note is printed on first creation).

## On-disk layout (schema `v1`)

```
design_db/v1/<spec_key>/        # spec_key = sha256(structural AAG + port spec)
    spec.json                   # name, ports, class, golden_sha, registered_from back-refs
    golden.v                    # the golden reference candidates are verified against
    verification.json           # the frozen verification (absent = unverified, inserts refused)
    designs/<source>:<hash>/    # one admitted implementation
        design.v                #   the implementation
        metrics.json            #   {intrinsic: {aig_nodes, aig_depth, aig_latches}, transistors_heavy}
        provenance.json         #   {source, created, verification: {tier, method, verdict, budget_s}}
    index.json                  # roll-up {design_id -> {struct_hash, metrics, source, created}}
design_db/v1/manifest.json      # reverse index {registered name -> {spec_key, class, n_designs}}
```

Everything is plain JSON + Verilog — inspectable, diffable, committable. A committed DB makes
builds reproducible: selection reads it deterministically, nothing regenerates.

**Spec keys** hash the *structural AAG* (the numeric `AigerExporter` section) plus the port spec —
not the Verilog text, whose internal wire names depend on the build context. Identical logic with
an identical interface maps to one slot, wherever and whenever it is built.

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

## API (S1 surface)

| Symbol | What it does |
|---|---|
| `register_slot(module_or_component, db=None, name=None) -> spec_key` | Register a slot (idempotent): spec + golden + default verification + manifest entry. Components are lowered with a fixed canonical top name. |
| `insert_design(spec_key, design, *, source, db=None, design_py=None, budget_s=None, provenance=None) -> InsertResult` | The gate: verify → dedup → stamp metrics → record provenance → admit atomically. `design` is a spire `Component`/`Netlist` (lowered to Verilog internally), a Verilog file path, or raw Verilog text. |
| `InsertResult` | `design_id`, `deduped`, `metrics`. |
| `resolve_db_root(db=None)` / `DesignDB` | DB-root resolution / low-level store handle. |
| `cec_check(design_v, golden_v, workdir, budget_s=…)` | Standalone Tier-0 CEC (raises on non-PASS). |
| `detect_class(module)` | `"combinational"` / `"sequential"` (register scan). |
| Exceptions | `VerificationFailed`, `CECTimeout`, `CECInapplicable`, `SlotUnverified`, `VerificationError`, `DesignDBError`. |

`import spire.design_db` is dependency-light: pyosys/aigverse are only imported when an insert
actually needs them.

## Coming next (will extend this page)

- **`@from_design_db(objective=, metric=, pin=, fill=)`** — the pure selection decorator: trace the
  decorated function, register its slot, select the best admitted implementation by objective and
  splice it in; miss ⇒ the original logic (a build never fails).
- **`spire db` CLI** — `init | ls | show <fn|key> --pareto | insert | verify`.
- **Sim verification tiers** (auto harness + authored stimulus) for sequential and CEC-infeasible
  combinational slots.
- Manifest-based slot discovery for tools and agents.
