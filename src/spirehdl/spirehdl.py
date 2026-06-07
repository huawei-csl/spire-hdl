# spire_hdl.py
# A tiny, SpinalHDL-inspired EDSL for Python → Verilog
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import itertools
import random
from typing import Optional, Union, Sequence
from spirehdl.signal_name_inference import (
    infer_signal_name_from_assignment,
    mark_expr_name,
    resolve_shared_wire_name,
)


# -----------------------------
# Shared sub-expression (CSE) support
# -----------------------------

class _SharedCache:
    """
    Tracks how many times an Expr *instance* is wrapped via as_expr(...).
    On the 2nd time, we create a Verilog wire (sig_{index}) with a driver = original expr.
    Further uses return that wire to shrink emitted Verilog.
    """
    counts: dict[int, int] = {}              # node_id -> count
    expr2sig: dict[int, "Signal"] = {}       # node_id -> created Signal
    wires: list["Signal"] = []               # all created wires in encounter order
    index: int = 0                           # for naming sig_{index}
    uid = itertools.count(1)                 # unique id for each expression
    used_names: set[str] = set()             # avoid duplicate auto-generated names
    opportunistic: bool = True               # wrap every subexpr into a wire (see flat_emit)

    @classmethod
    def reset(cls):
        cls.counts.clear()
        cls.expr2sig.clear()
        cls.wires.clear()
        cls.index = 0
        cls.used_names.clear()
        # note: `opportunistic` is intentionally NOT reset — it's a caller policy (see flat_emit), not per-module state.


def reset_shared_cache():
    """Call this before emitting each Verilog module to avoid cross-module bleed."""
    _SharedCache.reset()


import contextlib as _contextlib


@_contextlib.contextmanager
def flat_emit(enabled: bool = True):
    """Emit Verilog without *opportunistic* common-subexpression sharing.

    By default SpireHDL wraps every non-leaf subexpression into a named wire (``assign sig_k = ...``) 
    to shrink the emitted Verilog. This context manager disables this behavior, 
    The emitted Verilog is larger but can give better PPA results in some cases.
    """
    old = _SharedCache.opportunistic
    _SharedCache.opportunistic = not enabled
    try:
        yield
    finally:
        _SharedCache.opportunistic = old


def get_shared_wires() -> list["Signal"]:
    """Access the created wires (for inclusion in module's declarations/assigns)."""
    return list(_SharedCache.wires)

def _create_new_shared_wire(typ: HDLType, suggested_name: Optional[str] = None) -> "Signal":
    name, _SharedCache.index = resolve_shared_wire_name(
        suggested_name=suggested_name,
        used_names=_SharedCache.used_names,
        index=_SharedCache.index,
    )

    _SharedCache.used_names.add(name)
    sig = Signal(name, typ, "wire")
    sig._auto_generated = True
    _SharedCache.wires.append(sig)
    return sig


def _maybe_share(e: "Expr", force_share=False) -> "Expr":
    """
    If this exact Expr instance is seen the 2nd time via as_expr(...),
    create a 'wire sig_{index}' that drives from the original expression.
    On 3rd+ times, reuse the same wire.
    Leaf Signals/Consts are skipped (they're already "named"/literal).

    Each Expr is keyed by a monotonically increasing UID (assigned lazily)
    rather than ``id(e)``, so that Python's address recycling after GC
    cannot cause false cache hits.
    """
    if isinstance(e, (Signal, Const)):
        return e

    uid = getattr(e, '_cse_uid', None)
    if uid is None:
        uid = next(_SharedCache.uid)
        e._cse_uid = uid
    cnt = _SharedCache.counts.get(uid, 0) + 1
    _SharedCache.counts[uid] = cnt

    # Reuse an already-created wire for this exact instance.
    if uid in _SharedCache.expr2sig:
        return _SharedCache.expr2sig[uid]

    cnt_share = 1  # at what count start sharing
    opportunistic = _SharedCache.opportunistic and cnt == cnt_share
    if (force_share and cnt <= 1) or opportunistic:
        sig = _create_new_shared_wire(e.typ, getattr(e, "_suggested_name", None))
        sig._driver = e  # continuous assignment: assign sig = <original expr>;
        _SharedCache.expr2sig[uid] = sig
        return sig
    return e


# -----------------------------
# Types
# -----------------------------


@dataclass
class HDLType:
    width: int
    signed: bool = False
    is_bool: bool = False

    def __post_init__(self):
        if self.is_bool:
            self.width = 1
        if self.width < 0:
            raise ValueError("Type width must be >= 0")

    def range_str(self) -> str:
        return "" if self.width == 1 else f"[{self.width-1}:0]"

    def decl_str(self) -> str:
        sign = "signed " if self.signed else ""
        rng = self.range_str()
        return f"{sign}{rng}".strip()


def Bool() -> HDLType:
    return HDLType(1, signed=False, is_bool=True)


def UInt(w: int) -> HDLType:
    return HDLType(w, signed=False, is_bool=False)


def SInt(w: int) -> HDLType:
    return HDLType(w, signed=True, is_bool=False)


# -----------------------------
# Expressions
# -----------------------------


class Expr:
    typ: HDLType

    # Prevent accidental use in Python control flow.
    def __bool__(self):
        raise TypeError("HDL expressions cannot be used as Python booleans. Use mux()/comparators/etc.")

    # Arithmetic
    def __add__(self, other: ExprLike) -> "Expr":
        return op_add(self, as_expr(other))

    def __radd__(self, other: ExprLike) -> "Expr":
        return op_add(as_expr(other), self)

    def __sub__(self, other: ExprLike) -> "Expr":
        return op_sub(self, as_expr(other))

    def __rsub__(self, other: ExprLike) -> "Expr":
        return op_sub(as_expr(other), self)

    def __neg__(self) -> "Expr":
        return op_sub(as_expr(0), self)

    def __mul__(self, other: ExprLike) -> "Expr":
        return op_mul(self, as_expr(other))

    def __rmul__(self, other: ExprLike) -> "Expr":
        return op_mul(as_expr(other), self)

    # Bitwise and logic
    def __and__(self, other: ExprLike) -> "Expr":
        return op_bit(self, as_expr(other), "&")

    def __rand__(self, other: ExprLike) -> "Expr":
        return op_bit(as_expr(other), self, "&")

    def __or__(self, other: ExprLike) -> "Expr":
        return op_bit(self, as_expr(other), "|")

    def __ror__(self, other: ExprLike) -> "Expr":
        return op_bit(as_expr(other), self, "|")

    def __xor__(self, other: ExprLike) -> "Expr":
        return op_bit(self, as_expr(other), "^")

    def __rxor__(self, other: ExprLike) -> "Expr":
        return op_bit(as_expr(other), self, "^")

    def __invert__(self) -> "Expr":
        return op_not(self)

    # Shifts (logical). For variable shifts, result width is source width.
    def __lshift__(self, other: ExprLike) -> "Expr":
        return op_shift(self, as_expr(other), "<<")

    def __rshift__(self, other: ExprLike) -> "Expr":
        return op_shift(self, as_expr(other), ">>")

    # Comparators → Bool(1)
    def __eq__(self, other: ExprLike) -> "Expr":
        return op_cmp(self, as_expr(other), "==")

    def __ne__(self, other: ExprLike) -> "Expr":
        return op_cmp(self, as_expr(other), "!=")

    def __lt__(self, other: ExprLike) -> "Expr":
        return op_cmp(self, as_expr(other), "<")

    def __le__(self, other: ExprLike) -> "Expr":
        return op_cmp(self, as_expr(other), "<=")

    def __gt__(self, other: ExprLike) -> "Expr":
        return op_cmp(self, as_expr(other), ">")

    def __ge__(self, other: ExprLike) -> "Expr":
        return op_cmp(self, as_expr(other), ">=")

    # Indexing / slicing
    def __getitem__(self, sl: Union[int, slice]) -> "Expr":
        width = self.typ.width

        def _norm_idx(i: int, *, for_stop: bool = False) -> int:
            """
            Convert possibly-negative index to [0, width] (for_stop=True) or [0, width-1].
            Negative indices are interpreted Python-style: -1 == width-1, etc.
            """
            if i < 0:
                i += width
            # start index: 0 <= i < width
            # stop index:  0 <= i <= width  (for_stop=True allows i == width)
            if i < 0 or i > width or (not for_stop and i == width):
                raise ValueError(f"Index {i} out of range for width {width}")
            return i

        # -------------------------
        # Single-bit indexing
        # -------------------------
        if isinstance(sl, int):
            idx = _norm_idx(sl, for_stop=False)
            base = self if isinstance(self, (Const, Signal)) else _maybe_share(self, force_share=True)
            return Slice(base, idx, idx + 1)

        # -------------------------
        # Slicing
        # -------------------------
        if isinstance(sl, slice):
            if sl.step not in (None, 1):
                raise ValueError("Slice step must be 1")

            # Python-style defaults
            start = 0 if sl.start is None else sl.start
            stop = width if sl.stop is None else sl.stop

            # Normalize negatives
            start = _norm_idx(start, for_stop=False)
            stop = _norm_idx(stop, for_stop=True)

            if stop <= start:
                raise ValueError("Slice stop must be > start")

            base = self if isinstance(self, (Const, Signal)) else _maybe_share(self, force_share=True)
            return Slice(base, start, stop)

        raise TypeError("Unsupported index type")

    def to_verilog(self) -> str:
        raise NotImplementedError

    def as_expr(self) -> "Expr":
        """
        Returns self, but may replace it by a shared wire once this
        exact Expr instance has been seen multiple times.
        """
        return _maybe_share(self)


ExprLike = Union[Expr, int, bool]

# -----------------------------
# Leaf nodes
# -----------------------------


class Const(Expr):
    def __init__(self, value: int, typ: HDLType):
        self.value = int(value)
        self.typ = typ

    def to_verilog(self) -> str:
        if self.typ.is_bool:
            return "1'b1" if self.value != 0 else "1'b0"
        if self.typ.width == 0:
            raise ValueError("Cannot emit zero-width constants directly; use them only as placeholders in Concat.")

        val = int(self.value)

        # For negatives, use unary minus + *signed* literal: -<width>'sd<abs>
        if val < 0:
            return f"-{self.typ.width}'sd{abs(val)}"

        # Non-negative: choose signedness from the declared type
        base = "sd" if self.typ.signed else "d"
        return f"{self.typ.width}'{base}{val}"


class Signal(Expr):
    def __init__(self, name: str, typ: HDLType, kind: str): #, module: "Module"):
        self.name = name
        self.typ = typ
        self.kind = kind  # 'input' | 'output' | 'wire' | 'reg'
        self._driver: Optional[Expr] = None  # for wire/output
        self._init: Optional[Expr] = None  # for reg
        self._auto_generated: bool = False  # for internal use
        # Custom-verilog tagging — when a Component opts to emit a hand-written Verilog block in place of its
        # auto-generated logic, the framework tags the elaborate()-created signals so the emitter knows to skip
        # them. Sim still uses `_driver` normally (these flags are emission-time only).
        self._no_emit_decl: bool = False   # skip `wire …;` / `reg …;` declaration
        self._no_emit_drive: bool = False  # skip the `assign` or always-block driver line

    def __ilshift__(self, rhs: ExprLike) -> "Signal":
        """Connect combinational driver: y <<= expr"""
        rhs_e = fit_width(as_expr(rhs), self.typ)
        self._driver = rhs_e
        return self

    def set_init(self, init: ExprLike):
        if self.kind != "reg":
            raise TypeError("init can only be set on registers")
        self._init = fit_width(as_expr(init), self.typ)

    def to_verilog(self) -> str:
        return self.name

    def __repr__(self):
        return f"Signal(name={self.name!r}, kind={self.kind}, typ=<{self.typ.width}{'s' if self.typ.signed else 'u'}>)"


def fit_type(expr: ExprLike, to_type: HDLType) -> Signal:
    """Cast an expression to a specific type including sign extension (if source type is signed) or truncation as needed."""
    s = _create_new_shared_wire(to_type)
    s <<= fit_width(as_expr(expr), to_type)
    return s

def reinterpret(expr: ExprLike, to_type: HDLType) -> Signal:
    """Reinterpret an expression as a different type without changing bits. Widths must match."""
    e = as_expr(expr)
    if e.typ.width != to_type.width:
        raise ValueError("reinterpret requires same width")
    s = _create_new_shared_wire(to_type)
    s._driver = e
    return s

# explicit register
class Register(Signal):
    def __init__(self, typ: HDLType, init: Optional[ExprLike] = None, name: Optional[str]=None):
        if name is None:
            name = infer_signal_name_from_assignment("reg", "reg", __file__)
        super().__init__(name, typ, kind="reg")
        if init is not None:
            self.set_init(init)

# explicit wire
class Wire(Signal):
    def __init__(self, typ: HDLType, name: Optional[str]=None):
        if name is None:
            name = infer_signal_name_from_assignment("wire", "wire", __file__)
        super().__init__(name, typ, kind="wire")


# -----------------------------
# Memory primitive
# -----------------------------

class _MemoryStore(Signal):
    """Sim-only array storage for ``custom_verilog`` memory primitives (Middle path B).

    Provides O(1) simulation (``_mem_state`` array + ``step`` + the ``_ArrayIndex`` read
    leaf) and plugs into signal collection via its port wires' ``_memory_parent``
    back-edges. It emits **no Verilog of its own** — synthesis comes entirely from the
    wrapping primitive's ``custom_verilog()``. Construction marks the store and its ports
    no-emit so the parent emitter skips them.

    Ports are Signal attributes wired with ``<<=`` inside a primitive's ``elaborate()``::

        mem = _MemoryStore(UInt(9), depth=16, name="fifo")
        mem.write_addr   <<= addr_w
        mem.write_data   <<= din
        mem.write_enable <<= we       # default 1 if omitted
        mem.reset_enable <<= clr      # presence activates the reset arm
        mem.read_addr    <<= addr_r
        dout             <<= mem.read_data

    ``registered_read=True`` makes ``read_data`` a Register captured by ``step`` (the
    primitive's clock-only always block mirrors this in verilog).

    Not user-facing — designs use ``MemoryPrimitive`` / ``FIFOPrimitive``.
    """

    def __init__(self, elem_type: HDLType, depth: int, *,
                 init: Optional[Sequence[int]] = None,
                 registered_read: bool = False,
                 name: Optional[str] = None):
        if name is None:
            name = infer_signal_name_from_assignment("mem", "mem", __file__)
        if depth <= 0:
            raise ValueError(f"_MemoryStore depth must be > 0; got {depth}")
        if init is not None and len(init) != depth:
            raise ValueError(
                f"_MemoryStore init must have length == depth ({depth}); got {len(init)}")
        super().__init__(name, elem_type, kind="mem")
        self.depth = depth
        self.init = list(init) if init is not None else None
        self._registered_read = registered_read
        addr_t = UInt(max(1, (depth - 1).bit_length()))

        def _port(suffix: str, typ: HDLType, kind: str = "wire") -> Signal:
            s = Signal(f"{name}__{suffix}", typ, kind)
            s._memory_parent = self
            s._port_suffix = suffix
            return s

        # Required ports — user must drive when the corresponding group is active.
        self.write_addr   = _port("waddr", addr_t)
        self.write_data   = _port("wdata", elem_type)
        self.reset_enable = _port("rstn",  Bool())
        self.read_addr    = _port("raddr", addr_t)

        # Optional ports with defaults (user may override with `<<=`).
        self.write_enable = _port("we", Bool())
        self.write_enable._driver = Const(1, Bool())
        self.reset_value  = _port("rv", elem_type)
        self.reset_value._driver = Const(0, elem_type)
        if registered_read:
            self.read_enable = _port("re", Bool())
            self.read_enable._driver = Const(1, Bool())
            self.read_data = _port("rdata", elem_type, kind="reg")
            # Next-state for read_data is computed inside the memory's always block.
        else:
            self.read_enable = None
            self.read_data = _port("rdata", elem_type)
            self.read_data._driver = _ArrayIndex(self, self.read_addr, elem_type)

        # Sim-only store: never emits its own Verilog (the wrapping primitive's
        # custom_verilog does). Mark self + ports no-emit so the parent emitter skips them.
        for s in (self, *self._iter_ports()):
            s._no_emit_decl = True
            s._no_emit_drive = True

    # ----------------------------- introspection helpers ----------------------

    def _iter_ports(self) -> "list[Signal]":
        return [p for p in (
            self.write_addr, self.write_data, self.write_enable,
            self.reset_value, self.reset_enable,
            self.read_addr, self.read_data, self.read_enable,
        ) if p is not None]

    def _has_write_port(self) -> bool:
        return self.write_addr._driver is not None

    def _has_reset_arm(self) -> bool:
        return self.reset_enable._driver is not None

    # ----------------------------- simulator hooks ----------------------------

    def init_sim_state(self) -> "list[int]":
        if self.init is None:
            return [0] * self.depth
        from spirehdl.spirehdl_simulator import _to_bits  # local import (lazy to avoid cycles)
        return [_to_bits(v, self.typ.width) for v in self.init]

    def step(self, sim) -> None:
        """One clock-edge update: writes/reset to mem state + rdata reg capture.

        Mirrors verilog non-blocking semantics: all right-hand sides are sampled from the pre-edge state before any
        updates apply. We sample the rdata next-state *before* mutating the memory array, so a same-cycle write
        followed by a registered read gives the old value (write-before-read race resolved as it would be in
        hardware).
        """
        from spirehdl.spirehdl_simulator import _to_bits, _sid  # lazy, avoids cycles
        ev = sim._eval_signal_bits
        arr = sim._mem_state[id(self)]
        w = self.typ.width

        # 1. Sample rdata next-state from pre-edge mem state.
        rdata_next = None
        if self._registered_read and ev(self.read_enable) & 1:
            a = _to_bits(ev(self.read_addr), self.read_addr.typ.width)
            rdata_next = arr[a] if 0 <= a < self.depth else 0

        # 2. Apply write/reset to mem state.
        if self._has_reset_arm() and ev(self.reset_enable) & 1:
            rv = _to_bits(ev(self.reset_value), w)
            for i in range(self.depth):
                arr[i] = rv
        elif self._has_write_port() and ev(self.write_enable) & 1:
            a = _to_bits(ev(self.write_addr), self.write_addr.typ.width)
            if 0 <= a < self.depth:
                arr[a] = _to_bits(ev(self.write_data), w)

        # 3. Commit rdata reg next-state.
        if rdata_next is not None:
            sim._reg[_sid(self.read_data)] = _to_bits(rdata_next, w)


class _ArrayIndex(Expr):
    """Leaf Expr that emits ``mem.name[addr_wire.name]`` in verilog.

    Used as the ``_driver`` of an async-read memory's ``read_data`` wire. Walkers treat this as a leaf (no
    children): the address signal is reached through Memory's port traversal, not through this Expr's fields. The
    simulator's ``visit_array_index`` reads from ``_mem_state``.
    """

    def __init__(self, mem: "_MemoryStore", addr_wire: Signal, typ: HDLType):
        self.mem = mem
        self.addr_wire = addr_wire
        self.typ = typ

    def to_verilog(self) -> str:
        return f"{self.mem.name}[{self.addr_wire.name}]"


# -----------------------------
# Compound nodes
# -----------------------------


class Op2(Expr):
    def __init__(self, a: Expr, b: Expr, op: str, typ: HDLType):
        self.a = a
        self.b = b
        self.op = op
        self.typ = typ

    def to_verilog(self) -> str:
        if self.op != "nand":
            return f"({self.a.to_verilog()} {self.op} {self.b.to_verilog()})"
        else:
            return f"~({self.a.to_verilog()} & {self.b.to_verilog()})"  # nand # experimental feature


class Op1(Expr):
    def __init__(self, a: Expr, op: str, typ: HDLType):
        self.a = a
        self.op = op
        self.typ = typ

    def to_verilog(self) -> str:
        return f"({self.op}{self.a.to_verilog()})"


class Ternary(Expr):
    def __init__(self, sel: Expr, a: Expr, b: Expr):
        self.sel = sel
        self.a = a
        self.b = b
        # widen to max width, signed if either signed
        w = max(self.a.typ.width, self.b.typ.width)
        s = self.a.typ.signed or self.b.typ.signed
        self.typ = HDLType(w, s, is_bool=False)

    def to_verilog(self) -> str:
        a = fit_width(self.a, self.typ).to_verilog()
        b = fit_width(self.b, self.typ).to_verilog()
        return f"({self.sel.to_verilog()} ? {a} : {b})"


class Concat(Expr):
    def __init__(self, parts: Sequence[Expr]):
        self.parts = [p for p in (as_expr(x) for x in list(parts)) if p.typ.width > 0]
        if not self.parts:
            raise ValueError("Concat must include at least one non-zero-width part.")
        w = sum(p.typ.width for p in self.parts)
        self.typ = HDLType(w, signed=False, is_bool=False)

    def to_verilog(self) -> str:
        inner = ", ".join(p.to_verilog() for p in reversed(self.parts))
        return f"{{{inner}}}"


class Slice(Expr):
    def __init__(self, a: Expr, start: int, stop: int):
        if start < 0 or stop < 0:
            raise ValueError("Slice bounds must be >= 0")
        if stop <= start:
            raise ValueError("Slice stop must be > start")
        self.a = as_expr(a)
        if stop > self.a.typ.width:
            raise ValueError("Slice stop exceeds signal width")
        self.start = start
        self.lsb = start
        self.msb = stop - 1
        width = stop - start
        self.typ = HDLType(width, signed=False, is_bool=(width == 1))

    def to_verilog(self) -> str:
        # Full-width slice is a no-op — avoids illegal bit-select on 1-bit signals
        if self.start == 0 and self.msb + 1 == self.a.typ.width:
            return self.a.to_verilog()
        # Constant folding: evaluate bit-select at codegen time to avoid
        # illegal Verilog like 5'd31[4] (bit-select on a constant literal)
        if isinstance(self.a, Const):
            extracted = (self.a.value >> self.lsb) & ((1 << self.typ.width) - 1)
            return Const(extracted, self.typ).to_verilog()
        if self.typ.width == 1:
            return f"{self.a.to_verilog()}[{self.lsb}]"
        return f"{self.a.to_verilog()}[{self.msb}:{self.lsb}]"


class Resize(Expr):
    def __init__(self, a: Expr, to_width: int):
        self.a = a
        self.to_width = to_width
        self.typ = HDLType(to_width, signed=a.typ.signed, is_bool=(to_width == 1))

    def to_verilog(self) -> str:
        aw = self.a.typ.width
        tw = self.to_width
        if aw == tw:
            return self.a.to_verilog()
        
        # If operand is a constant, just re-emit it with the target width/signedness.
        # This avoids patterns like (8'sd0)[7] and nested replications.
        if isinstance(self.a, Const):
            adapted = Const(self.a.value, HDLType(tw, signed=self.a.typ.signed, is_bool=(tw == 1)))
            return adapted.to_verilog()
        
        if aw > tw:
            # truncate LSBs kept (common hardware pattern)
            return f"{self.a.to_verilog()}[{tw-1}:0]"
        # extend
        ext_bits = tw - aw
        if self.a.typ.signed:
            # Sign-extend: { ext_bits copies of MSB, src }
            if aw == 1:
                signbit = self.a.to_verilog()
            else:
                signbit = f"{self.a.to_verilog()}[{aw-1}]"
            return f"{{{{{ext_bits}{{{signbit}}}}}, {self.a.to_verilog()}}}"
        else:
            return f"{{{{{ext_bits}{{1'b0}}}}, {self.a.to_verilog()}}}"


# -----------------------------
# Operator helpers
# -----------------------------

def bits_required(v: int) -> int:
    if v == 0:
        return 1
    if v > 0:
        return v.bit_length()
    return (-v).bit_length() + 1  # include sign


def as_expr(x: ExprLike) -> Expr:
    if isinstance(x, Expr):
        # Route through the instance method so sharing can occur
        return x.as_expr()
    if isinstance(x, bool):
        return Const(1 if x else 0, Bool())
    if isinstance(x, int):
        signed = x < 0
        w = bits_required(x)
        return Const(x, HDLType(w, signed=signed))
    raise TypeError(f"Cannot convert {type(x)} to Expr")


def bitwise_result_type(a: Expr, b: Expr) -> HDLType:
    return HDLType(max(a.typ.width, b.typ.width), signed=False)


def add_result_type(a: Expr, b: Expr) -> HDLType:
    return HDLType(max(a.typ.width, b.typ.width) + 1, signed=a.typ.signed or b.typ.signed)


def mul_result_type(a: Expr, b: Expr) -> HDLType:
    return HDLType(a.typ.width + b.typ.width, signed=a.typ.signed or b.typ.signed)


def fit_width(e: Expr, t: HDLType) -> Expr:
    """Fits wdith including sign-extension or truncation as needed."""
    if e.typ.width == t.width:
        return e
    if e.typ.width > t.width:
        e = _maybe_share(e, force_share=True)  # for verilog emission
    return Resize(e, t.width)

# helper functions for sign/zero extension
def s_ext(expr: Expr, width: int) -> Expr:
    if expr.typ.width >= width:
        raise ValueError("s_ext: target width must be greater than expr width")
    return fit_width(fit_type(expr, SInt(expr.typ.width)), SInt(width))

def z_ext(expr: Expr, width: int) -> Expr:
    if expr.typ.width >= width:
        raise ValueError("z_ext: target width must be greater than expr width")
    return fit_width(fit_type(expr, UInt(expr.typ.width)), UInt(width))

# -----------------------------

def op_add(a: Expr, b: Expr) -> Expr:
    t = add_result_type(a, b)
    return mark_expr_name(Op2(a, b, "+", t), __file__)


def op_sub(a: Expr, b: Expr) -> Expr:
    t = add_result_type(a, b)
    return mark_expr_name(Op2(a, b, "-", t), __file__)


def op_mul(a: Expr, b: Expr) -> Expr:
    t = mul_result_type(a, b)
    return mark_expr_name(Op2(a, b, "*", t), __file__)


def op_bit(a: Expr, b: Expr, sym: str) -> Expr:
    t = bitwise_result_type(a, b)
    return mark_expr_name(Op2(fit_width(a, t), fit_width(b, t), sym, t), __file__)

# op bit with shared inputs
# def op_bit(a: Expr, b: Expr, sym: str) -> Expr:
#     t = bitwise_result_type(a, b)
#     a_s = _maybe_share(a, force_share=True)
#     b_s = _maybe_share(b, force_share=True)
#     return Op2(fit_width(a_s, t), fit_width(b_s, t), sym, t)


def op_not(a: Expr) -> Expr:
    return mark_expr_name(Op1(a, "~", HDLType(a.typ.width, signed=False, is_bool=a.typ.is_bool)), __file__)


def op_shift(a: Expr, b: Expr, sym: str) -> Expr:
    # if b is const, widen on left shift; otherwise keep width
    if isinstance(b, Const) and sym == "<<":
        t = HDLType(a.typ.width + b.value, signed=a.typ.signed)
    else:
        t = HDLType(a.typ.width, signed=a.typ.signed)
    return mark_expr_name(Op2(a, b, sym, t), __file__)


def op_cmp(a: Expr, b: Expr, sym: str) -> Expr:

    # for verilog emmission we need to align widths for all compares, and if either is signed, align as signed
    w = max(a.typ.width, b.typ.width)
    t_target = HDLType(w, signed=a.typ.signed or b.typ.signed)
    # if a is not const
    if not isinstance(a, Const):
        a_al = fit_width(a, t_target)
    else:
        a_al = a
    if not isinstance(b, Const):
        b_al = fit_width(b, t_target)
    else:
        b_al = b
    return mark_expr_name(Op2(a_al, b_al, sym, Bool()), __file__)


def mux(sel: ExprLike, a: ExprLike, b: ExprLike) -> Expr:
    return Ternary(as_expr(sel), as_expr(a), as_expr(b))

# alias
def mux_if(if_cond: ExprLike, then_expr: ExprLike, else_expr: ExprLike) -> Expr:
    return mux(if_cond, then_expr, else_expr)


def cat(*parts: ExprLike) -> Expr:
    return Concat([as_expr(p) for p in parts])
