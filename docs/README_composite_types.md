# Composite data types

Spire-HDL ships a small family of *composites*: structured Python objects that wrap one or
more HDL `Expr` leaves and behave as a single, bit-packable value.  All composites derive from
`HDLComposite` and share a common API for flattening to bits, packed assignment (`<<=`), and
element-wise assignment (`@=`).

Source: [`src/spire/composite/`](../src/spire/composite).

| Type | Purpose | Source |
|---|---|---|
| `HDLComposite` | Abstract base — `to_bits`, `assign`, `<<=`, `@=` (see [Common API](#common-api)) | [`base.py`](../src/spire/composite/base.py) |
| [`Array`](#array) | N-dimensional vector of `Expr` or nested composites | [`array.py`](../src/spire/composite/array.py) |
| [`CompositeRecord`](#compositerecord) | Declarative bundle with named fields (class attributes) | [`record.py`](../src/spire/composite/record.py) |
| [`CompositeRecordDynamic`](#compositerecorddynamic) | Bundle defined from instance attributes / `@dataclass` | [`record_dynamic.py`](../src/spire/composite/record_dynamic.py) |
| [`FixedPoint`](#fixedpoint) | Fixed-point view of a bitvector with arithmetic + quantization | [`fixed_point.py`](../src/spire/composite/fixed_point.py) |
| [`FloatingPoint`](#floatingpoint) | Floating-point view with `add` / `mul` helpers | [`floating_point.py`](../src/spire/composite/floating_point.py) |
| [`CompositeRegister`](#compositeregister) | Single register holding any packed composite | [`register.py`](../src/spire/composite/register.py) |

## Common API

Every composite exposes:

- `to_list_first_level()` — first-level children (`Expr` or nested composite).
- `to_list()` — fully flattened list of `Expr` leaves.
- `to_bits()` — single `Expr` bitvector (concat of all leaves).
- `width` — total bit width.
- `assign(rhs)` / `<<= rhs` — *packed* assignment: pack rhs to bits, slice across leaves.
- `@= rhs` — *element-wise* assignment: drive each leaf from the matching rhs leaf.
- `wire_like(template)` — classmethod returning a fresh, wire-backed instance with the same shape.

---

## `Array`

`Array` is an ordered, possibly nested, sequence of `Expr` or other composites.  It supports
N-dimensional indexing with `tuple`/`slice` keys.

```python
from spire.composite.array import Array
from spire.expr import UInt, Const, Wire

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

## `CompositeRecord`

Declarative, class-based bundle.  Fields are declared as class attributes; each instance gets
its own freshly cloned wires (no sharing between instances).

```python
from spire.composite.record import CompositeRecord
from spire.composite.array import Array
from spire.expr import UInt, SInt, Wire, Signal

class Packet(CompositeRecord):
    addr:    Signal = Wire(UInt(8))
    payload: Signal = Wire(SInt(16))
    lanes:   Array  = Array([Wire(UInt(4)) for _ in range(4)])   # nested composite

p0 = Packet()
p1 = Packet()

assert p0.addr is not p1.addr          # each instance has its own wires
assert p0.width == 8 + 16 + 4 * 4      # = 40 bits

p0 @= p1                # element-wise (field-by-field) copy
p0 <<= p1.to_bits()     # packed: slice the 40-bit vector across all leaves
```

---

## `CompositeRecordDynamic`

Bundle defined from *instance* attributes (typically via `@dataclass`).  Use this when the
field set is built at construction time rather than declared statically on the class — for
example, when generating IO records for parameterized cores.

```python
from dataclasses import dataclass
from spire.composite.array import Array
from spire.composite.record_dynamic import CompositeRecordDynamic
from spire.expr import UInt, Wire

@dataclass
class MMAcIO(CompositeRecordDynamic):
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

Unlike `CompositeRecord`, fields are read directly from the instance via `vars(self)` (or
dataclass `fields()`), so the same class can hold arbitrarily different shapes per instance.

---

## `FixedPoint`

Bitvector view with an explicit total width, fractional width, and sign.  Provides
`add` / `sub` / `mul` with optional output type and quantization mode (`ARITHQuant`).

```python
from spire.composite.fixed_point import (
    ARITHQuant, FixedPoint, FixedPointType,
)
from spire.expr import Const

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
from spire.composite.floating_point import (
    FloatingPoint, FloatingPointType,
)
from spire.expr import Const

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

## `CompositeRegister`

Wraps any composite type in a single register: one packed `Signal(kind="reg")` underneath, a
structured `.value` view on top.  Use it when you want a register whose contents you read and
write as a structured composite, not raw bits.

```python
from spire import Component, IORecord, Input, Output, UInt
from spire.composite.register import CompositeRegister
from spire.composite.fixed_point import FixedPoint, FixedPointType
from spire.expr import as_expr

q8_8 = FixedPointType(width_total=16, width_frac=8, signed=False)

class AccDemo(Component):
    def __init__(self):
        self.io = IORecord(x=Input(UInt(8)), y=Output(UInt(q8_8.width_total)))
        self.elaborate()

    def elaborate(self):
        acc = CompositeRegister(FixedPoint, q8_8, name="acc_reg", init=0)
        acc_val = acc.value           # FixedPoint view on the register
        x_q     = FixedPoint(q8_8, bits=as_expr(self.io.x) << q8_8.width_frac)

        acc <<= acc_val.add(x_q, out_type=q8_8)   # next-state assignment
        self.io.y <<= acc.bits        # driving an output collects the register automatically
```

Working tests demonstrate the full simulation flow:
[`testing/test_fixed_point.py`](../testing/test_fixed_point.py),
[`testing/test_floating_point.py`](../testing/test_floating_point.py),
[`testing/test_record.py`](../testing/test_record.py),
[`testing/test_array2.py`](../testing/test_array2.py).
