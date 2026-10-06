# Design DB

> Registration, gated insertion, the selection decorator, simulation/CEC
> verification tiers, and the CLI are implemented. Candidate generation and campaign enrichment
> live in RTLScout; see the RTLScout README for those flows.

`spire.design_db` is a content-addressed library of correct implementations for reusable
subcircuits. Each subcircuit is a **slot**. A slot stores its golden implementation, its port
specification, its verification oracle, and any number of admitted candidate implementations.

The core idea is simple:

1. A producer inserts a candidate into a slot.
2. The DB verifies the candidate against the slot's golden.
3. Only passing candidates are admitted and stamped with metrics.
4. A consumer later selects the best admitted candidate for a chosen objective.

Generation and selection are deliberately separate. Filling a slot once makes every later build
able to reuse the stored implementations without rerunning the search that found them.

## Core Concepts

### Slot identity

A slot is identified by its **`spec_key`**:

```
sha256(structural golden AAG + port spec)
```

The key is 64 hex characters and is also the slot directory name. Python APIs use the key directly:

```python
pick_design(spec_key, ...)
insert_design(spec_key, ...)
```

Slots can also have human names in the DB manifest, such as `adder8`. A name is an alias for one
`spec_key`, not a second identity. Within one DB, the binding is permanent:

- registering the same structure under the same name is idempotent;
- registering different logic under an existing name raises;
- several names may alias the same slot;
- `spec_key`s remain universal across DBs.

The CLI accepts a manifest name, a full key, or a unique key prefix of at least 8 characters.

### Designs

An admitted implementation is stored under:

```
designs/<source>:<hash10>/
```

That full string is the **`design_id`**. The `source` part is the tag supplied at insert time and
may itself contain colons, so parse IDs from the right if needed:

```
agent:rtl-subcircuit-depth:15d010c21e
```

means source `agent:rtl-subcircuit-depth`, structural hash prefix `15d010c21e`.

Use exact `design_id`s for reproducibility pins. Take them from `spire db show <slot>`.

### Verification

Every slot has one acceptance oracle. Every `insert` and `verify` call uses that oracle, so all
designs in a slot are judged by the same yardstick.

Combinational slots get Tier-0 CEC by default at registration. Sequential slots register
successfully but stay unverified until a simulation tier is frozen.

### Selection

Selection is a pure query over admitted designs and their stored metric blocks. It does not generate
new designs, update the manifest, or spend verification budget. If no eligible design exists,
selection returns `None`; the decorator falls back to the original logic.

## Quick Start: Use a Slot from Source

```python
from spire import Bool, Component, IORecord, Input, Output, UInt
from spire.design_db import from_design_db
from spire.expr import mux

@from_design_db(objective="area")
def mac_step(a, b, acc, en, thresh):
    p = mux(en, a * b, 0)
    t = acc + p
    s = mux(t[8], 255, t[0:8])
    return mux(s > thresh, s, 0)

class Top(Component):
    def __init__(self):
        self.io = IORecord(
            a=Input(UInt(4)),
            b=Input(UInt(4)),
            c=Input(UInt(8)),
            en=Input(Bool()),
            thresh=Input(UInt(8)),
            y=Output(UInt(8)),
        )
        self.elaborate()

    def elaborate(self):
        self.io.y <<= mac_step(
            self.io.a, self.io.b, self.io.c, self.io.en, self.io.thresh
        )
```

A decorated function is treated as one optimization unit. Candidates may restructure across the
function body as long as they remain equivalent to the slot golden.

`@from_design_db` is a reader:

1. The best admitted implementation is selected for `objective` and `metric`.
2. The selected implementation is spliced into the circuit.
3. On a miss, the original function body is used inline.

The decorator never generates designs unless you explicitly provide `fill=`.

Decorator arguments:

| Argument | Meaning |
|---|---|
| `objective=` | Objective to minimize: `area`, `delay`, `adp`, `edap`, or a combinator. |
| `metric=` | Measurement system to use, such as `aig`, `transistors`, or `asap7`. |
| `pin=` | Exact `design_id` to splice. Missing pins raise. |
| `fill=` | Optional miss hook. Called once, then selection is retried. |
| `name=` | Manifest name. Defaults to the function qualname and is a permanent binding. |
| `db=` | Explicit DB root. |

The `fill` hook is called as:

```python
fill(spec_key, db_root=..., objective=..., metric=...)
```

It must populate the slot through the normal insert gate. If it succeeds, the decorator reselects
and can splice the new design in the same compile. RTLScout ships a ready-made hook,
`make_rtlscout_fill`.

### First compile with an empty DB

Decorating code is cheap when the DB is empty:

1. The traced call registers the slot.
2. If no DB exists, write paths create `./design_db` and print one note.
3. The slot receives `spec.json`, `golden.v`, default CEC verification for combinational slots,
   `starting_point.py` when source capture succeeds, and a manifest entry.
4. Selection misses, so the original function body is used inline.

The emitted circuit is structurally identical to the undecorated function. No CEC, yosys run, or
generation budget is spent on this miss path.

Two cases intentionally behave differently:

- `pin=` raises if the pinned design is absent;
- `fill=` runs the generation hook before falling back.

## Quick Start: Fill a Slot

Filling is independent of the decorator flow above — any registered slot can be filled. Here a
fresh 8-bit adder slot, registered directly:

```python
from spire import Component, IORecord, Input, Output, UInt
from spire.design_db import insert_design, register_slot

class Adder(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(8)), b=Input(UInt(8)), s=Output(UInt(9)))
        self.elaborate()

    def elaborate(self):
        self.io.s <<= self.io.a + self.io.b

key = register_slot(Adder(), name="adder8")

candidate = """
module cand_add(input [7:0] a, input [7:0] b, output [8:0] s);
  assign s = {1'b0, a} + {1'b0, b};
endmodule
"""

res = insert_design(key, candidate, source="handwritten")
print(res.design_id, res.metrics["transistors"]["metrics"], res.metrics["aig"]["metrics"])
```

An incorrect candidate raises `VerificationFailed` and leaves no admitted design behind. A
candidate with incompatible ports is rejected before admission. Re-inserting a structurally
identical design returns `deduped=True`.

Spire-native producers can pass a `Component` or `Netlist` directly:

```python
insert_design(key, MyOtherAdder(), source="spire")
```

Python design files are also accepted when they define `build() -> Component/Netlist`; the gate
elaborates them and stores both the canonical Verilog and the Python source provenance.

### Seed the baseline

`seed_original(spec_key)` or `spire db seed --slot <slot>` inserts the slot golden as a design with
`source="original"`. This gives selection a floor: an argmin cannot pick something worse than the
original if the original is present and eligible.

Seeding is idempotent through structural dedup. It is not automatic during registration because
registration must stay cost-free; fillers are expected to seed before generating.

## Selection and Metrics

```python
from spire.design_db import constrained, lexicographic, pareto_front, pick_design, weighted

pick_design(key, objective="area")
pick_design(key, objective="delay", metric="aig")
pick_design(key, objective=constrained(minimize="area", subject_to={"delay": 500}))
pick_design(key, objective=weighted({"area": 1.0, "delay": 0.1}))
pick_design(key, objective=lexicographic(("area", "delay")))
pick_design(key, pin="spire:1a2b3c4d5e")
pareto_front(key)
```

Built-in objective axes are `area`, `delay`, `adp`, and `edap`. `adp` is derived as
`area * delay` when a metric system maps `area` and `delay` but not `adp`. Other axes are available
only when the metric system defines them.

### Metric systems

Each design's `metrics.json` contains one block per measurement system. The insert gate stamps:

- `aig`: structural nodes, depth, and latch count;
- `transistors`: yosys heavy-pipeline transistor estimate, with delay borrowed from `aig`.

Additional systems, such as `asap7`, can be added with `annotate`.

When `metric=None`, selection resolves deterministically in this order:

1. the alphabetically first non-built-in technology system, if any exists;
2. `transistors`;
3. `aig`.

Constraints are evaluated in the same system as the objective.

Example metric block:

```json
{
  "aig": {
    "metrics": {"aig_nodes": 105, "aig_depth": 12, "aig_latches": 8},
    "objectives": {"area": "aig_nodes", "delay": "aig_depth"}
  },
  "transistors": {
    "metrics": {"transistors_heavy": 202},
    "objectives": {"area": "transistors_heavy", "delay": "aig.aig_depth"}
  }
}
```

The `objectives` map tells selection which metric field backs each axis. A value like
`aig.aig_depth` borrows from a sibling system instead of duplicating the number.

### Pins and selection logs

`pin=` always requires an exact `design_id`. It does not accept prefixes, because a prefix that is
unique today can become ambiguous as a slot grows. A missing pin raises instead of falling back to
metric selection.

Selection itself is not recorded in the DB. To audit what a compile actually spliced, set:

```bash
SPIREHDL_DB_SELECTION_LOG=selection.jsonl python design.py
```

Each splice appends one JSON line:

```json
{"spec_key": "...", "name": "adder8", "design_id": "...", "objective": "area", "metric": "aig"}
```

The log belongs to the compiled artifact. The library does not store "last selected" state.

### Temporary selection overrides

Use overrides for what-if compiles, A/B measurements, or reproducing one composition without
changing source:

```python
from spire.design_db import selection_overrides

with selection_overrides({"adder8": "verilog:e218599799"}):
    Top().to_verilog_file("design.v")
```

The same map can cross process boundaries — RTLScout's `measure_db_compositions.py` uses it to
measure every splice combination of a run:

```bash
SPIREHDL_DB_PINS='{"adder8": "verilog:e218599799"}' python design.py
```

Keys are exact `spec_key`s or manifest names. Values are exact `design_id`s. An explicit source
`pin=` wins over an override. Scopes nest (innermost wins) and overlay the environment. Overrides
are never written to the manifest.

### Inspecting a slot from Python

The Python API works on `spec_key`s. Starting from a manifest name is one lookup:

```python
from spire.design_db import DesignDB, pick_design

d = DesignDB.open()
spec_key = d.read_json(d.manifest_path)["slots"]["adder8"]["spec_key"]

for design_id, info in d.read_index(spec_key).items():
    print(design_id, info["source"], info["metrics"]["transistors"])
    sel = pick_design(spec_key, pin=design_id)
```

From the shell:

```bash
spire db show adder8 | jq -r '.designs | keys[]'
```

The per-slot index maps each `design_id` to its source, creation time, structural hash, metrics,
and rediscovery info. It is derived from the `designs/` directories on every read; `index.json`
is only a self-healing cache for inspection.

Timestamps are Unix epoch seconds (UTC by definition); `spire db ls` and `spire db show` render
them in local time. Sorting the index by `created` gives the slot's improvement timeline. A
design's `rediscoveries` count and `last_rediscovered` stamp are bumped whenever an insert dedups
against it, so rising rediscoveries without new admissions mean the search has converged.

## Adding Technology Metrics

The insert gate records only metrics Spire can compute itself. Richer per-technology PPA comes from
external tools and is attached with `annotate`; RTLScout's `db-score` is exactly such a producer:

```bash
spire db annotate --slot adder8 --design agent:9f3c1a2b7d --tech asap7 area=118.3 delay=94.6
```

```python
from spire.design_db import annotate

annotate(key, "agent:9f3c1a2b7d", tech="asap7", values={"area": 118.3, "delay": 94.6})
```

`design_ref` may be an exact `design_id` or a unique prefix within the slot. That looseness is for
interactive commands only; reproducibility pins still require exact IDs.

This adds an `asap7` block to `metrics.json` without changing the gate-stamped `aig` or
`transistors` blocks:

```json
{
  "aig": {
    "metrics": {"aig_nodes": 105, "aig_depth": 12, "aig_latches": 8},
    "objectives": {"area": "aig_nodes", "delay": "aig_depth"}
  },
  "transistors": {
    "metrics": {"transistors_heavy": 202},
    "objectives": {"area": "transistors_heavy", "delay": "aig.aig_depth"}
  },
  "asap7": {
    "metrics": {"area": 118.3, "delay": 94.6},
    "objectives": {"area": "area", "delay": "delay"}
  }
}
```

After that, `pick_design(key, objective="area", metric="asap7")` and
`@from_design_db(metric="asap7")` can rank designs using ASAP7 area.

Rules for annotation:

- reserved systems such as `aig` and `transistors` cannot be overwritten;
- re-annotating the same technology requires `--force`;
- `values` must be numeric;
- standard axes in `values` (`area`, `delay`, `adp`, `edap`) become selectable;
- other numeric keys are stored but not used for selection;
- `--raw <file>` stores an optional full tool-stats JSON blob under `.raw`;
- every design in a slot must use the same `objectives` map for a given technology.

Annotate all designs in a slot before selecting on a technology metric. A technology counts as
present once any design has it, and designs missing that system are ineligible in that metric
system. This includes the seeded `original:*` design if you use it as the baseline. Until coverage
is complete, keep `metric=` explicit — the default resolver prefers technology systems and would
otherwise select in a system that excludes the floor.

## Verification

Every insert runs the slot's configured verification. Passing designs are admitted; failing designs
leave no admitted implementation behind.

| Tier | Method | Applies to | Status |
|---:|---|---|---|
| 0 | CEC against `golden.v` using yosys and `yosys-abc cec` | combinational only | implemented |
| 1 | Auto simulation harness: corners plus seeded random stimulus (exhaustive for tiny input spaces), golden-simulated outputs, frozen `tb.sv` and `vectors.dat` | sequential, or combinational by choice | implemented |
| 2 | Authored Python stimulus generator: `generate(ports, n_vectors, seed)` | protocol-heavy sequential designs | implemented |

### CEC

Combinational slots receive CEC verification by default. CEC runs with a bounded budget
(`budget_s`, default 120 seconds). A timeout raises `CECTimeout` and leaves the slot unchanged.
Requesting CEC for a sequential slot raises `CECInapplicable`.

### Simulation tiers

Simulation tiers check cycle-accurate trace equivalence under frozen stimulus. Expected outputs are
always produced by simulating the golden with Verilator. A sequential candidate must match the
golden output trace cycle for cycle; a design with different latency is rejected.

Freezing a sim verification:

1. simulates the golden with the chosen stimulus;
2. stores the input and expected-output trace as `vectors.dat`;
3. stores the replay testbench as `tb.sv`;
4. marks both files read-only;
5. writes `verification.json`, recording tier, method, vector count, and `stimulus_author`
   (Tier-1 records `auto`; authored freezes record `--author`, or null when omitted).

After that, every candidate is checked against exactly that trace. The frozen `tb.sv` follows the
RTLScout testbench contract (`TB_SUMMARY total=N errors=M`, then `PASS`) and binds the DUT by name
via `-DDUT=<top>`. Re-freezing is refused because it would change the yardstick for designs
already admitted to the slot. A CEC slot may still be switched once to a sim tier; that sim
freeze is then final.

Use `--check` before freezing authored stimulus:

```bash
spire db set-verification --slot mypipe --stimulus stim.py --check
```

The dry run loads the generator and produces masked vectors against the slot interface, but it does
not simulate, write, or freeze anything. The API equivalent is `check_stimulus(...)`.

## DB Location

DB root resolution is zero-config:

1. explicit `db=` or `--db`;
2. `$SPIREHDL_DB_PATH`;
3. nearest existing `design_db/` directory upward from the current working directory;
4. auto-create `./design_db` for write paths only.

## On-Disk Layout

Schema `v1`:

```text
design_db/v1/<spec_key>/        # spec_key = sha256(structural AAG + port spec)
    spec.json                   # name, ports, class, clock, golden_sha, source_ref, registered_from
    golden.v                    # golden reference candidates are verified against
    starting_point.py           # captured decorated function source, when available
    verification.json           # configured/frozen verification; absent means inserts are refused
    tb.sv, vectors.dat          # sim tiers only; read-only after freeze
    designs/<source>:<hash>/    # one admitted implementation
        design.v                # canonical IR
        design.aag              # precomputed splice input
        design.py               # Python source when known
        source/<rel>.py         # vendored project-local imports for .py inserts
        metrics.json            # measurement blocks
        provenance.json         # source, created time, verification verdict, Python provenance
    index.json                  # derived cache; designs/ is the source of truth
design_db/v1/manifest.json      # name -> {spec_key, class}
```

Everything is plain JSON and Verilog, so a DB is inspectable, diffable, and committable. A committed
DB makes builds reproducible because selection is deterministic and nothing regenerates during a
read.

Spec keys hash the structural AAG and port spec, not Verilog text. Verilog wire names can depend on
build context; structural AAG avoids that instability.

For decorator-registered slots, source capture stores:

- `starting_point.py`, tagged by capture fidelity;
- `source_ref` in `spec.json`, including file, qualname, and line.

## Concurrency

Admission uses one atomic directory rename. The slot index is derived from the admitted
`designs/` directories, and each design carries its own `metrics.json` and `provenance.json`.
Concurrent inserts cannot lose a design.

Manifest writes are rare and serialized with an `fcntl` lock. Parallel fills of one DB are
supported. For efficiency, run one filler per slot at a time when possible.

## Python API

| Symbol | Purpose |
|---|---|
| `@from_design_db(objective=, metric=, pin=, fill=, name=, db=)` | Register, select, and splice. Misses use the original logic. |
| `register_slot(module_or_component, db=None, *, name=None) -> spec_key` | Register a slot idempotently. |
| `insert_design(spec_key, design, *, source, db=None, budget_s=None, python_copy=None, provenance=None)` | Verify, dedup, stamp metrics, record provenance, and admit atomically. |
| `check_design(spec_key, design, *, db=None, budget_s=None) -> dict` | Advisory verification with no admission. |
| `seed_original(spec_key, *, db=None, budget_s=None)` | Admit the golden as `source="original"`. |
| `annotate(spec_key, design_ref, *, tech, values, raw=None, force=False, db=None)` | Attach a technology metric block to a stored design. |
| `pick_design(spec_key, *, objective=, metric=, pin=, sources=, db=)` | Deterministic pure selection query. |
| `selection_overrides({slot: design_id, ...})` / `$SPIREHDL_DB_PINS` | Temporary what-if pins. |
| `$SPIREHDL_DB_SELECTION_LOG` | Compile-scoped JSONL log of decorator splices. |
| `pareto_front(spec_key, objectives=("area", "delay"), *, metric=None, db=None)` | Non-dominated set. |
| `constrained / weighted / lexicographic` | Objective combinators. |
| `resolve_db_root(db=None, *, create=True)` / `DesignDB` | DB-root resolution and low-level store access. |
| `cec_check(design_v, golden_v, workdir, *, budget_s=...)` | Standalone Tier-0 CEC. |
| `freeze_sim_verification(spec_key, *, stimulus_file=None, n_vectors=, seed=, sim_budget_s=, stimulus_author=None, db=None)` | Freeze Tier-1 or Tier-2 simulation verification. |
| `check_stimulus(spec_key, *, stimulus_file, n_vectors=, seed=, db=)` | Dry-run an authored stimulus generator against the slot interface; writes nothing (`set-verification --check`). |
| `run_frozen_tb(spec_key, candidate_v, workdir, *, db=None, budget_s=None)` | Run the frozen simulation oracle. |
| `detect_class(module)` | Return `"combinational"` or `"sequential"`. |
| Exceptions | `VerificationFailed`, `CECTimeout`, `SimTimeout`, `CECInapplicable`, `SlotUnverified`, `VerificationError`, `DesignDBError`. |

`import spire.design_db` is dependency-light. Heavy modules such as pyosys and aigverse are imported
only when insertion, verification, or splicing needs them.

## CLI

The console entry point is `spire`; the module form drops the `db` prefix
(`spire db ls` ≡ `python -m spire.design_db ls`).

```bash
spire db init
spire db ls
spire db ls --json
spire db show adder8 --pareto

spire db verify cand.py --slot adder8
spire db verify cand.v --slot adder8

spire db insert cand.py --slot adder8 --source agent:rtl-subcircuit
spire db insert cand.v --slot adder8 --source handwritten --budget 300
spire db seed --slot adder8

spire db annotate --slot adder8 --design 9f3c1a2b7d --tech asap7 area=118.3 delay=94.6

spire db set-verification --slot mypipe --auto --vectors 256 --seed 0 --sim-budget 300
spire db set-verification --slot mypipe --stimulus stim.py --check
spire db set-verification --slot mypipe --stimulus stim.py
spire db set-verification --slot mypipe --stimulus stim.py --author agent:rtl-dv-prep
spire db set-verification --slot adder8 --cec --budget 300
```

The verification-related commands have distinct jobs:

| Command | Scope | Writes? | Purpose |
|---|---:|---:|---|
| `set-verification` | slot | yes | Choose the oracle used by later checks. Sim tiers freeze a trace. |
| `verify <design>` | candidate | no | Run the slot oracle and report `PASS` or `FAIL`. |
| `insert <design>` | candidate | yes, on pass | Run the same check, then admit the candidate. |

`insert` and `verify` accept `.py`, `.v`, and `.sv` inputs. A `.py` design file must define
`build() -> Component/Netlist`; the gate elaborates it and stores the generated Verilog as
`design.v`, which is the DB's canonical intermediate representation. Project-local Python imports
are vendored under the design's `source/` directory and listed in provenance. External Verilog
inserts remain supported; they simply carry no Python source.

`ls`, `show`, and `verify` never create a DB; the only write among them is `show` refreshing the
self-healing `index.json` cache. `annotate` writes `metrics.json`, but it
opens an existing DB only and does not auto-create one. Write paths such as registration, `init`,
`insert`, `seed`, and `set-verification` may create a DB. Verification failures from `insert` exit
with code 2 and print `REJECTED (...)`.

### CLI ↔ Python equivalents

Most operations exist on both sides; the gaps are deliberate:

| CLI | Python | Remarks |
|---|---|---|
| `spire db init` | `resolve_db_root(create=True)` | |
| `spire db ls` | — | Compose from `DesignDB` primitives (manifest + `read_index`); see *Inspecting a slot from Python*. |
| `spire db show` | — | Same primitives; `--pareto` corresponds to `pareto_front()`. |
| `spire db insert` | `insert_design()` | |
| `spire db seed` | `seed_original()` | |
| `spire db verify` | `check_design()` | |
| `spire db annotate` | `annotate()` | |
| `spire db set-verification --auto / --stimulus` | `freeze_sim_verification()` | |
| `spire db set-verification --check` | `check_stimulus()` | |
| `spire db set-verification --cec` | — | CLI-only; combinational slots already get CEC by default at registration. |
| — | `register_slot()` / `@from_design_db` | Registration needs a live `Component`/`Netlist`, so it is Python-only. |
| — | `pick_design()` + combinators | Selection is Python-only; from the shell, parse `show` JSON. |
| — | `selection_overrides()` | `$SPIREHDL_DB_PINS` is its shell / cross-process form. |

## Tool Requirements

A plain pip install is enough for insertion and CEC in the common case. The gate uses the `yosys`
binary when available; otherwise it uses the pyosys wheel in a child interpreter. Candidate failures
stay isolated from the caller process.

If `yosys-abc` is unavailable, CEC falls back from ABC's `cec` to yosys' `equiv` flow. Both match
ports by name.

Verilator is required only for simulation tiers. Sim-tier unit tests skip when Verilator is absent.
