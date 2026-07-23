# Automatic arithmetic optimization

Spire ships a library of configurable arithmetic building blocks: prefix adders (Kogge-Stone, Brent-Kung, Sklansky, ...), multipliers (stage-based with PPG/PPA/FSA selection), and subtractors.  Rather than requiring the user to pick the right topology, `replace_arithmetic_ops` can **automatically select the best configuration** for every `+`, `-`, and `*` operator in a design, guided by a pre-computed evaluation database. Note that `replace_arithmetic_ops` is called by the `@arithmetic_optimized` decorator internally.

Three optimization objectives are available:

| Objective | Minimizes | Good for |
|-----------|-----------|----------|
| `"area"`  | the size `metric` (default: Yosys transistor count) | Silicon area, power |
| `"delay"` | AIG depth (AND-gate levels) | Clock frequency |
| `"adp"`   | `metric` × AIG depth | Balanced designs |

How size is measured is set by the optional **`metric`** argument — accepted by `ArithmeticAutoConfig`
and `@arithmetic_optimized`, e.g. `ArithmeticAutoConfig(objective="area", metric="aig_count")`:

| Metric | Measures |
|--------|----------|
| `"transistors_heavy"` *(default)* | Yosys transistor count under the full `synth; clean -purge` pipeline |
| `"transistors"` | Yosys transistor count under the lite `abc -fast` pipeline (faster, rougher) |
| `"aig_count"` | AIG gate count post-synth (aigverse) — technology-independent |

Delay is always AIG depth; `metric` sets only the size axis: `"area"` minimizes the metric (tie-break:
depth), `"delay"` minimizes depth (tie-break: the metric), `"adp"` minimizes metric × depth.

## Example: 16-bit unsigned adder

The auto-config sweep evaluates all prefix-adder topologies and picks the best one per objective:

| Objective | Topology | Transistors | AIG Depth |
|-----------|----------|------------:|----------:|
| plain (Yosys `+`) | *(default synthesis)* | 648 | 32 |
| `area`    | Sparse Kogge-Stone (2) | 626 | 32 |
| `delay`   | Kogge-Stone            | 952 | 22 |
| `adp`     | Brent-Kung             | 746 | 26 |

The `adp` objective automatically finds the Brent-Kung topology as a compromise between the area-optimized Sparse Kogge-Stone and the speed-optimized Kogge-Stone.

## Example: 8-bit ALU (add + sub + mul)

A complete 8-bit ALU with addition, subtraction, and multiplication, comparing Yosys default synthesis (`*` operator) against auto-optimized replacements:

| Configuration | Transistors | AIG Depth |
|---------------|------------:|----------:|
| plain (Yosys `+`/`-`/`*`)  | 6168 | 127 |
| `area`             | 3004 |  73 |
| `delay`            | 3830 |  30 |
| `adp`              | 3084 |  30 |

The area objective achieves a **51% transistor reduction** over default synthesis.  The delay objective cuts critical-path depth from 127 to 30 AND-gate levels (**4.2x**), and the balanced `adp` objective achieves both nearly minimal area *and* minimal depth.

## Usage

```python
from dataclasses import dataclass
from spire.expr import Signal, UInt
from spire.component import Component
from spire.arithmetic.int_arithmetic_config import ArithmeticAutoConfig, replace_arithmetic_ops

@dataclass
class ALUIO:
    a: Signal
    b: Signal
    y_add: Signal
    y_sub: Signal
    y_mul: Signal

class ALU(Component):
    def __init__(self, w: int):
        self.io = ALUIO(
            a=Signal(typ=UInt(w), kind="input"),
            b=Signal(typ=UInt(w), kind="input"),
            y_add=Signal(typ=UInt(w + 1), kind="output"),
            y_sub=Signal(typ=UInt(w + 1), kind="output"),
            y_mul=Signal(typ=UInt(2 * w), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        self.io.y_add <<= self.io.a + self.io.b  # addition
        self.io.y_sub <<= self.io.a - self.io.b  # subtraction
        self.io.y_mul <<= self.io.a * self.io.b  # multiplication

alu = ALU(w=8)

# Replace all +, -, * operators with optimized hardware — one line
replace_arithmetic_ops(alu, ArithmeticAutoConfig(objective="adp"))

module = alu.to_netlist("OptimizedALU")
print(module.to_verilog())
```

See [`testing/low_level_arithmetic/int_adders/test_arithmetic_auto_config.py`](../testing/low_level_arithmetic/int_adders/test_arithmetic_auto_config.py) for full test and benchmark code.

## Explicit builder functions

If you'd rather construct optimized arithmetic directly instead of writing Python operators and running `replace_arithmetic_ops` afterwards, three builder helpers accept the same `ArithmeticAutoConfig` and look up the best per-operation configuration on the fly:

- `build_adder(a, b, cfg)` — builds a single optimized adder from two expressions.
- `build_multiplier(a, b, cfg)` — builds a single optimized multiplier from two expressions.
- `adder_tree(values, cfg)` — reduces a sequence of expressions with a balanced binary tree of optimized adders.

```python
from spire.arithmetic.int_arithmetic_config import (
    ArithmeticAutoConfig, build_adder, build_multiplier, adder_tree,
)

cfg = ArithmeticAutoConfig(objective="adp")

# Single optimized adder / multiplier
sum_ab = build_adder(self.io.a, self.io.b, cfg)
prod   = build_multiplier(self.io.a, self.io.b, cfg)

# Balanced tree reduction over N terms
self.io.y <<= adder_tree(
    [self.io.c[i] * self.io.x[i] for i in range(n_taps)], cfg,
)
```

Use the builders when you want direct control over where optimized hardware is instantiated (e.g., inside a loop, or mixed with hand-written RTL); use `replace_arithmetic_ops` when you prefer to author the design with plain operators and let the optimizer rewrite the whole graph — including MAC/inner-product fusion, which the builders do not perform.

For the `@arithmetic_optimized` decorator (same one-liner ergonomics but using structural replacement instead of an external synthesizer), see [README_optimization_decorators.md](README_optimization_decorators.md).

## Replacement-based optimization details

Each `+`, `-`, and `*` in the expression graph is independently replaced with the empirically best prefix-adder or stage-based multiplier configuration for its specific bit-width and signedness.  For widths not in the evaluation database, the nearest data point is selected using logarithmic interpolation.

The optimizer also detects **multiply-accumulate (MAC) patterns** (`a * b + c`) and fuses them into a single hardware unit, absorbing the accumulate operand directly into the multiplier's column reduction and eliminating a full adder stage.

**Fusion in `replace_arithmetic_ops` / `@arithmetic_optimized` is purely syntactic: the `*` must be a
direct operand of the `+`.** Any node between them — `fit_type`/`reinterpret`
(explicit width/sign adjustment), a slice (fixed-point rescale), or a `mux` —
hides the pattern and the ops are replaced individually instead of fused
(individually replaced `*` still get an optimized implementation, but the shared
column-reduction of the fused unit is lost):

```python
y <<= c + a * b                          # fuses (mixed widths are fine — the fused core widens internally)
y <<= c + fit_type(a * b, SInt(32))      # does NOT fuse (Resize in between)
y <<= c + mux(en, a * b, zero)           # does NOT fuse (mux in between)
y <<= mux(en, c + a * b, c)              # fuses — place the selection AFTER the operation

y <<= c + (a * b)[4:16]                  # does NOT fuse (Slice between * and +)
y <<= (c + a * b)[4:20]                  # fuses — slice AFTER the full operation
```

Rule of thumb: For fusing, write the arithmetic bare (`c + a*b`, letting Spire's implicit
widening handle widths) and put muxes/slices/width adjustments *after* the full
operation.

It also detects **multi-input add chains** (`a + b + c + d + ...`, 3+ operands) and replaces them with a carry-save reduction tree plus a single final 2-input adder (mirroring what yosys's `alumacc` does internally). The chain's `(ppa, fsa, optim_type)` is picked from a dedicated `miaN` DB sweep. Example:

```python
@arithmetic_optimized(objective="area")
def add_chain(a, b, c, d, cin):
    return a + b + c + d + cin    # one CSA tree + final adder, not 4 chained 2-input adders
```

## Example: 4-tap FIR filter with MAC fusion

A common DSP pattern where MAC fusion shines — each tap is `coeff[i] * x[i]` accumulated into a sum:

```python
from dataclasses import dataclass
from spire.expr import UInt, Signal
from spire.component import Component
from spire.composite.array import Array
from spire.arithmetic.int_arithmetic_config import ArithmeticAutoConfig, replace_arithmetic_ops

@dataclass
class FIRIO:
    x: Array    # input samples
    c: Array    # coefficients
    y: Signal   # output

class FIR(Component):
    """N-tap FIR filter: y = c[0]*x[0] + c[1]*x[1] + ... + c[N-1]*x[N-1]"""
    def __init__(self, n_taps: int = 4, w: int = 8):
        self.n_taps = n_taps
        self.io = FIRIO(
            x=Array([Signal(name=f"x{i}", typ=UInt(w), kind="input") for i in range(n_taps)]),
            c=Array([Signal(name=f"c{i}", typ=UInt(w), kind="input") for i in range(n_taps)]),
            y=Signal(typ=UInt(2 * w + 2), kind="output"),
        )
        self.elaborate()

    def elaborate(self):
        # Plain Python operators — the optimizer handles the rest
        acc = self.io.c[0] * self.io.x[0]
        for i in range(1, self.n_taps):
            acc = acc + self.io.c[i] * self.io.x[i]
        self.io.y <<= acc

fir = FIR(n_taps=4, w=8)
replace_arithmetic_ops(fir, ArithmeticAutoConfig(objective="adp"))
module = fir.to_netlist("FIR4_optimized")
print(module.to_verilog())
```

The optimizer automatically detects the `c*x + ...` inner product pattern and fuses all four multiply-add pairs into a single column reduction with one final-stage adder, eliminating three intermediate adder stages:

| Configuration | Transistors | AIG Depth |
|---------------|------------:|----------:|
| plain (Yosys `*`, `+`) | 24556 | 142 |
| `area`             | 11356 |  49 |
| `delay`            | 11434 |  45 |
| `adp`              | 11434 |  45 |

The area objective achieves a **54% transistor reduction**, and the delay objective cuts critical-path depth from 142 to 45 AND-gate levels (**68%**).

See [`testing/low_level_arithmetic/int_adders/test_arithmetic_auto_config.py`](../testing/low_level_arithmetic/int_adders/test_arithmetic_auto_config.py) for the full test and benchmark code.
