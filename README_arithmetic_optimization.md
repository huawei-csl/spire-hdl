# Automatic arithmetic optimization

Sprout-HDL ships a library of configurable arithmetic building blocks: prefix adders (Kogge-Stone, Brent-Kung, Sklansky, ...), multipliers (stage-based with PPG/PPA/FSA selection), and subtractors.  Rather than requiring the user to pick the right topology, `replace_arithmetic_ops` can **automatically select the best configuration** for every `+`, `-`, and `*` operator in a design, guided by a pre-computed evaluation database.

Three optimization objectives are available:

| Objective | Minimizes | Good for |
|-----------|-----------|----------|
| `"area"`  | Yosys transistor count | Silicon area, power |
| `"delay"` | AIG depth (AND-gate levels) | Clock frequency |
| `"adp"`   | Area-delay product | Balanced designs |

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
| plain (Yosys `*`)  | 6168 | 127 |
| `area`             | 3004 |  73 |
| `delay`            | 3830 |  30 |
| `adp`              | 3084 |  30 |

The area objective achieves a **51% transistor reduction** over default synthesis.  The delay objective cuts critical-path depth from 127 to 30 AND-gate levels (**4.2x**), and the balanced `adp` objective achieves both nearly minimal area *and* minimal depth.

## Usage

```python
from sprouthdl.arithmetic.int_arithmetic_config import ArithmeticAutoConfig, replace_arithmetic_ops

# Define your design using plain operators
class ALU(Component):
    def elaborate(self):
        self.io.y_add <<= self.io.a + self.io.b
        self.io.y_sub <<= self.io.a - self.io.b
        self.io.y_mul <<= self.io.a * self.io.b

alu = ALU(width=8)

# Replace operators with optimized hardware — one line
replace_arithmetic_ops(alu, ArithmeticAutoConfig(objective="adp"))

module = alu.to_module("OptimizedALU")
print(module.to_verilog())
```

Each `+`, `-`, and `*` in the expression graph is independently replaced with the empirically best prefix-adder or stage-based multiplier configuration for its specific bit-width and signedness.  For widths not in the evaluation database, the nearest data point is selected using logarithmic interpolation.

The optimizer also detects **multiply-accumulate (MAC) patterns** (`a * b + c`) and fuses them into a single hardware unit, absorbing the accumulate operand directly into the multiplier's column reduction and eliminating a full adder stage.

## Example: 4-tap FIR filter with MAC fusion

A common DSP pattern where MAC fusion shines — each tap is `coeff[i] * x[i]` accumulated into a sum:

```python
from dataclasses import dataclass
from sprouthdl.sprouthdl import UInt, Signal
from sprouthdl.sprouthdl_module import Component
from sprouthdl.aggregate.aggregate_array import Array
from sprouthdl.arithmetic.int_arithmetic_config import ArithmeticAutoConfig, replace_arithmetic_ops

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
            y=Signal(name="y", typ=UInt(2 * w + 2), kind="output"),
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
module = fir.to_module("FIR4_optimized")
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

See [`testing/low_level_arithmetic/int_adders/test_arithmetic_auto_config.py`](testing/low_level_arithmetic/int_adders/test_arithmetic_auto_config.py) for the full test and benchmark code.
