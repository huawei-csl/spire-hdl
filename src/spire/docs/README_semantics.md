# Spire semantics: types, evaluation, and backend conformance

Spire designs have three observable backends: the Python simulator, the AIGER export, and the emitted Verilog. This
document is the normative reference for what a design *means*, so that all three agree by construction and any
disagreement is unambiguously a bug in a specific backend. Authority order: this document > simulator implementation
> exported artifacts. If the simulator is found to violate a rule here, the simulator is corrected.

## 1. The two semantic domains

**Domain A — combinational / expression values: the spire IR defines the semantics; the simulator is the executable
specification.** Every expression node has an explicit `HDLType(width, signed)`, evaluates at its *own* width, and
wraps there; parents extend the wrapped value. This is a deliberate departure from IEEE-1364
context-determination (where an expression is re-sized as a whole and intermediate results never wrap): spire's rule
is local, explicit in the types, and independent of where an expression is later used. Emitted Verilog must be
*generated* so that IEEE evaluation of the text reproduces spire values (see §2, "Backend conformance") — spire
semantics are never reinterpreted to match raw Verilog sizing.

**Domain B — temporal and state behavior (registers, resets, memories, power-on): real hardware, i.e. the emitted
RTL, defines the semantics; the simulator models it.** Where a 2-state simulator cannot represent hardware reality
(X values), §3 defines the documented convention.

## 2. Expression evaluation model (Domain A, normative)

Definitions. *decode(pattern, typ)*: the pattern read as an unsigned integer, or as its two's-complement value if
`typ.signed`. *encode(value, width)*: `value mod 2^width`. A node stores a width-bit pattern; "the value of a node"
means decode of that pattern per the node's own type.

| Node | Result type | Result value |
|---|---|---|
| `a + b`, `a - b` | width `max(wa, wb) + 1`; signed iff either operand signed | exact integer math on decoded operands, encoded into the result width |
| `a * b` | width `wa + wb`; signed iff either signed | same (lossless for `+`, `-`, `*` at these widths) |
| `& \| ^` | width `max(wa, wb)`; **unsigned** | each operand's pattern extended to the result width per *its own* signedness, then bitwise |
| `~a`, `nand` | width `wa` (resp. `max`); **unsigned** | bitwise complement of the width-`wa` pattern |
| `==  !=  <  <=  >  >=` | `Bool` (1 bit, unsigned) | exact integer comparison of decoded operands |
| `a << k` (constant k ≥ 0) | width `wa + k`; sign of `a` | exact `value(a) * 2^k` (lossless) |
| `a << s` (variable s) | width `wa`; sign of `a` | `value(a) * 2^s` encoded into `wa` (wraps; `s ≥ wa` gives 0) |
| `a >> s` | width `wa`; sign of `a` | **logical** right shift of the width-`wa` pattern, zero fill — also for signed `a` (see §4) |
| `sel ? a : b` (`mux`) | width `max(wa, wb)`; signed iff **either** branch signed (see §4) | chosen branch decoded per its own type, encoded into the result type; `sel` is true iff nonzero |
| `cat(...)`, replication | sum of widths; unsigned | pattern concatenation, LSB-first |
| `a[i]`, `a[i:j]` | slice width; unsigned | pattern bits, Python-style `[start:stop]` bounds |
| `Resize` / `fit_width` | target type | truncation keeps low bits; extension per the **source** signedness |
| `Const(v, typ)` | `typ` | `v`, which must be representable in `typ` |

Construction rules:
- Shift amounts are interpreted as unsigned patterns regardless of their declared type.
- A negative constant shift amount is invalid (`ValueError` at construction) — it has no meaning in any backend.
- A `Const` whose value is not representable in its type (including negative values in an unsigned type) is invalid
  (`ValueError` at construction); implicit int coercion (`as_expr`) always chooses a sufficient type.
- Division, modulo, power, and reduction operators do not exist in the IR; adding one requires a row in the table
  above first.

Backend conformance (Domain A):
- **Simulator**: the executable specification; the table above restates its behavior.
- **AIGER export**: the exported network must evaluate identically to the simulator on all inputs.
- **Verilog emission**: IEEE-1364 evaluation of the emitted text must equal the simulator on all inputs. Because
  IEEE re-sizes context-determined operands (and self-determines concat parts, shift amounts, and `$signed`
  arguments), the emitter must place a width boundary — a named wire, or equivalent explicit sizing — around any
  operand whose IEEE evaluation could differ from its spire value: compound operands of operators and comparisons,
  ternary branches, extension-concat payloads, and shift left-operands/amounts. Emission options (sharing,
  `simplify`, `flat_emit`) may change netlist *structure*, never values or declared port types.
- **Identifiers**: simulation accepts any Python-legal signal name; emission must produce lexically legal Verilog
  identifiers (keyword-safe, ASCII, non-empty, no leading digit), sanitizing deterministically where needed. Naming
  is presentation, never a semantic input.

## 3. Temporal and state model (Domain B, normative)

1. **Clock edge**: registers and memory write ports commit on the active edge using values sampled *pre-edge*. A
   register (or memory location) read in the same cycle as its write returns the old value; chained registers shift
   exactly one stage per edge.
2. **Reset**: asynchronous assert. State elements whose RTL has a reset arm load their init/reset value both on a
   bare reset assertion and on any clocked step while reset is held. State elements whose RTL has **no** reset arm —
   registered-read capture registers, memory array contents — hold their value through both paths. The simulator
   must implement the hold on both paths, matching the RTL always-block structure. Init/reset values are constants
   (a register rejects a dynamic init expression at construction): an async reset arm loads set/clear pins in real
   cells, so a computed reset value is not implementable — model it as an explicit mux instead.
3. **Power-on**: RTL registers without an `initial` block are X until the first reset; the 2-state simulator starts
   at the declared init value. Convention: simulation and testbench flows assert reset before relying on register
   state, and generated replay testbenches begin with the reset event. This divergence is documented, not hidden.
4. **Out-of-range access**: reads of an out-of-range address are X in RTL; the simulator returns **0** (and every
   sim-side implementation variant must use the same convention). Out-of-range writes are dropped in both.
5. **X/Z**: the simulator is 2-state; no X/Z propagation is modeled anywhere.

## 4. Design decisions and rationale

- **Why the simulator is the value spec**: user designs are written and validated against it; the AIGER backend
  implements the same node-local semantics; and "every node wraps at its declared width" is a rule a reader can
  apply locally, whereas IEEE context-determination depends on the whole enclosing expression. The cost is that the
  emitter must isolate compound operands; the benefit is that what you simulate is what you synthesize.
- **Ternary signedness (signed-if-either)**: a mux between a signed and an unsigned branch yields a signed result in
  spire, unlike IEEE's signed-only-if-both. Chosen because it preserves the value of whichever branch is selected
  under later widening. The emitter implements it by giving mixed-sign muxes an explicitly typed boundary wire.
- **Right shift is logical, also for signed operands**: `>>` operates on the pattern. An arithmetic right shift is a
  possible future operator, not an alternate reading of `>>`; code needing sign-preserving division-by-power-of-two
  should build it explicitly.
- **Hardware is the state spec**: reset/hold/edge behavior exists to describe silicon; a simulator that "improves"
  on RTL reset behavior (e.g. clearing capture registers that hardware would hold) hides real bugs. Where 2-state
  simulation cannot express hardware (X at power-on, X on out-of-range reads), the convention is stated in §3 and
  testbench generators enforce it rather than papering over it.

## 5. Conformance testing

`testing/differential/` carries the permanent conformance harness for Domain A: an IEEE-1364 reference evaluator
that must first be *calibrated* — 100% agreement with the simulator on an exhaustive battery of single-operator,
leaf-operand shapes — before its verdicts on composite shapes count; a battery of nested-expression shapes covering
every row of the §2 table in compound positions; and a fixed-seed random-circuit fuzz comparing simulator, reference
evaluator, and reloaded AIGER export. Changes to expression construction, typing, simulation, or emission must keep
this harness green. Domain B is verified by differential simulation against behavioral golden models plus review of
the emitted always-block templates; the documented divergences (§3.3, §3.4) get doc-tests, not equivalence tests.
