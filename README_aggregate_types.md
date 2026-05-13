# Aggregate data types

Sprout-HDL ships a small family of *aggregates*: structured Python objects that wrap one or
more HDL `Expr` leaves and behave as a single, bit-packable value.  All aggregates derive from
`HDLAggregate` and share a common API for flattening to bits, packed assignment (`<<=`), and
element-wise assignment (`@=`).

Source: [`src/sprouthdl/aggregate/`](src/sprouthdl/aggregate).

| Type | Purpose | Source |
|---|---|---|
| `HDLAggregate` | Abstract base — `to_bits`, `assign`, `<<=`, `@=` (see [Common API](#common-api)) | [`hdl_aggregate.py`](src/sprouthdl/aggregate/hdl_aggregate.py) |
| [`Array`](#array) | N-dimensional vector of `Expr` or nested aggregates | [`aggregate_array.py`](src/sprouthdl/aggregate/aggregate_array.py) |
| [`AggregateRecord`](#aggregaterecord) | Declarative bundle with named fields (class attributes) | [`aggregate_record.py`](src/sprouthdl/aggregate/aggregate_record.py) |
| [`AggregateRecordDynamic`](#aggregaterecorddynamic) | Bundle defined from instance attributes / `@dataclass` | [`aggregate_record_dynamic.py`](src/sprouthdl/aggregate/aggregate_record_dynamic.py) |
| [`FixedPoint`](#fixedpoint) | Fixed-point view of a bitvector with arithmetic + quantization | [`aggregate_fixed_point.py`](src/sprouthdl/aggregate/aggregate_fixed_point.py) |
| [`FloatingPoint`](#floatingpoint) | Floating-point view with `add` / `mul` helpers | [`aggregate_floating_point.py`](src/sprouthdl/aggregate/aggregate_floating_point.py) |
| [`AggregateRegister`](#aggregateregister) | Single register holding any packed aggregate | [`aggregate_register.py`](src/sprouthdl/aggregate/aggregate_register.py) |

## Common API

Every aggregate exposes:

- `to_list_first_level()` — first-level children (`Expr` or nested aggregate).
- `to_list()` — fully flattened list of `Expr` leaves.
- `to_bits()` — single `Expr` bitvector (concat of all leaves).
- `width` — total bit width.
- `assign(rhs)` / `<<= rhs` — *packed* assignment: pack rhs to bits, slice across leaves.
- `@= rhs` — *element-wise* assignment: drive each leaf from the matching rhs leaf.
- `wire_like(template)` — classmethod returning a fresh, wire-backed instance with the same shape.

---

## `Array`

`Array` is an ordered, possibly nested, sequence of `Expr` or other aggregates.  It supports
N-dimensional indexing with `tuple`/`slice` keys.

```python
from sprouthdl.aggregate.aggregate_array import Array
from sprouthdl.sprouthdl import UInt, Const, Wire

# 1D vector of constants
v = Array([Const(1, UInt(8)), Const(2, UInt(8)), Const(3, UInt(8))])
print(v[0])         # first element (Expr)
print(v[1:3])       # Array of length 2

# 2D matrix — element access, row, column
mat = Array([
    Array([Const(1, UInt(8)), Const(2, UInt(8))]),
    Array([Const(3, UInt(8)), Const(4, UInt(8))]),
])
mat[0, 1]   # scalar at row 0 col 1
mat[0, :]   # row 0  (1D Array)
mat[:, 1]   # column 1  (1D Array)

# Element-wise assignment between two same-shape Arrays of wires
dst = Array([Wire(UInt(8)) for _ in range(4)])
src = Array([Wire(UInt(8)) for _ in range(4)])
dst @= src
```

---

## `AggregateRecord`

Declarative, class-based bundle.  Fields are declared as class attributes; each instance gets
its own freshly cloned wires (no sharing between instances).

```python
from sprouthdl.aggregate.aggregate_record import AggregateRecord
from sprouthdl.aggregate.aggregate_array import Array
from sprouthdl.sprouthdl import UInt, SInt, Wire, Signal

class Packet(AggregateRecord):
    addr:    Signal = Wire(UInt(8))
    payload: Signal = Wire(SInt(16))
    lanes:   Array  = Array([Wire(UInt(4)) for _ in range(4)])   # nested aggregate

p0 = Packet()
p1 = Packet()

assert p0.addr is not p1.addr          # each instance has its own wires
assert p0.width == 8 + 16 + 4 * 4      # = 40 bits

p0 @= p1                # element-wise (field-by-field) copy
p0 <<= p1.to_bits()     # packed: slice the 40-bit vector across all leaves
```

---

## `AggregateRecordDynamic`

Bundle defined from *instance* attributes (typically via `@dataclass`).  Use this when the
field set is built at construction time rather than declared statically on the class — for
example, when generating IO records for parameterized cores.

```python
from dataclasses import dataclass
from sprouthdl.aggregate.aggregate_array import Array
from sprouthdl.aggregate.aggregate_record_dynamic import AggregateRecordDynamic
from sprouthdl.sprouthdl import UInt, Wire

@dataclass
class MMAcIO(AggregateRecordDynamic):
    A: Array
    B: Array
    Y: Array

io = MMAcIO(
    A=Array([Wire(UInt(8)) for _ in range(4)]),
    B=Array([Wire(UInt(8)) for _ in range(4)]),
    Y=Array([Wire(UInt(20)) for _ in range(1)]),
)

assert io.width == 4*8 + 4*8 + 1*20
```

Unlike `AggregateRecord`, fields are read directly from the instance via `vars(self)` (or
dataclass `fields()`), so the same class can hold arbitrarily different shapes per instance.

---

## `FixedPoint`

Bitvector view with an explicit total width, fractional width, and sign.  Provides
`add` / `sub` / `mul` with optional output type and quantization mode (`ARITHQuant`).

```python
from sprouthdl.aggregate.aggregate_fixed_point import (
    ARITHQuant, FixedPoint, FixedPointType,
)
from sprouthdl.sprouthdl import Const

q8_8 = FixedPointType(width_total=16, width_frac=8, signed=True)

a = FixedPoint(q8_8, bits=Const(0x0180, q8_8.to_hdl_type()))   # 1.5
b = FixedPoint(q8_8, bits=Const(0x0040, q8_8.to_hdl_type()))   # 0.25

s_full = a + b                                  # full-precision sum (wider)
s_q88  = a.add(b, out_type=q8_8,                # quantized back to Q8.8
               q=ARITHQuant.WrpRnd)             # round, wrap on overflow
p_q88  = a.mul(b, out_type=q8_8, q=ARITHQuant.WrpTrc)
```

Supported quantization modes today: `WrpTrc` (wrap + truncate) and `WrpRnd` (wrap + round).
`Clp*` / `Sat*` modes are reserved.

---

## `FloatingPoint`

IEEE-style floating-point view (1 sign bit + exponent + fraction), parameterized by
`FloatingPointType(exponent_width, fraction_width)`.  Optional `subnormal_support`
selects a subnormal-aware multiplier.

```python
from sprouthdl.aggregate.aggregate_floating_point import (
    FloatingPoint, FloatingPointType,
)
from sprouthdl.sprouthdl import Const

# binary16 (half precision)
ft = FloatingPointType(exponent_width=5, fraction_width=10)

raw_1p0 = 0x3C00
fp = FloatingPoint(ft, bits=Const(raw_1p0, ft.to_hdl_type()))

# field accessors
sign     = fp.sign        # 1-bit Expr
exponent = fp.exponent    # 5-bit Expr
fraction = fp.fraction    # 10-bit Expr

# Arithmetic produces another FloatingPoint of the same ftype
a = FloatingPoint(ft, name="a")
b = FloatingPoint(ft, name="b")
y_mul = a * b
y_add = a + b
```

---

## `AggregateRegister`

Wraps any aggregate type in a single register: one packed `Signal(kind="reg")` underneath, a
structured `.value` view on top.  Use it when you want a register whose contents you read and
write as a structured aggregate, not raw bits.

```python
from sprouthdl.aggregate.aggregate_register import AggregateRegister
from sprouthdl.aggregate.aggregate_fixed_point import FixedPoint, FixedPointType
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl import UInt, as_expr

m = Module("AccDemo", with_clock=True, with_reset=False)
x = m.input(UInt(8), "x")

q8_8 = FixedPointType(width_total=16, width_frac=8, signed=False)

acc = AggregateRegister(FixedPoint, q8_8, name="acc_reg", init=0)
m._signals.append(acc.bits)   # expose the underlying reg signal

acc_val = acc.value           # FixedPoint view on the register
x_q     = FixedPoint(q8_8, bits=as_expr(x) << q8_8.width_frac)

acc <<= acc_val.add(x_q, out_type=q8_8)   # next-state assignment

y = m.output(UInt(q8_8.width_total), "y")
y <<= acc.bits
```

Working tests demonstrate the full simulation flow:
[`testing/test_fixed_point.py`](testing/test_fixed_point.py),
[`testing/test_floating_point.py`](testing/test_floating_point.py),
[`testing/test_aggregate_record.py`](testing/test_aggregate_record.py),
[`testing/test_array2.py`](testing/test_array2.py).
