"""Sim-only memory storage for the memory primitives (Middle path B).

Holds `_MemoryArray` — the shape-agnostic, O(1)-simulation array with a port factory — and
the `_ArrayIndex` read leaf, plus the small port records. Kept out of `spire.py` so the
core stays lean: everything synthesis-side (Verilog emission, read-under-write, masks,
registered-read pipelining) lives in `spire/primitives/`. Designs use `MemoryPrimitive` /
`RamPrimitive` / `FIFOPrimitive`, never these classes directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from spire.signal_name_inference import infer_signal_name_from_assignment
from spire.expr import Bool, Const, Expr, HDLType, Signal, UInt


@dataclass
class _WritePort:
    """A write port on a `_MemoryArray` (sim-only). Wired with `<<=` by a primitive."""
    addr: "Signal"
    data: "Signal"
    enable: "Signal"
    mask: "Optional[Signal]" = None
    chunk_w: int = 0


@dataclass
class _ReadPort:
    """An async read port: `data` is a wire driven by `_ArrayIndex(store, addr)` (combinational).

    A *registered* read is composed in the primitive (a normal capture `Register` over this
    async `data`), so the store has no registered-read concept and `step` never drives a reg.
    """
    addr: "Signal"
    data: "Signal"


@dataclass
class _ResetArm:
    """Broadcast-clear arm: when `enable` is high, every cell is loaded with `value`."""
    enable: "Signal"
    value: "Signal"


@dataclass
class _RWPort:
    """Read/write port (readWriteSync): writes `data_in` when `enable & write`; `data_out` reads `mem[addr]`."""
    addr: "Signal"
    data_in: "Signal"
    write: "Signal"
    enable: "Signal"
    data_out: "Signal"
    mask: "Optional[Signal]" = None


class _MemoryArray(Signal):
    """Sim-only multi-port array storage for `custom_verilog` primitives (Middle path B).

    O(1) simulation (`_mem_state` array + `step` + the `_ArrayIndex` read leaf); emits **no
    Verilog** — synthesis comes from the wrapping primitive's `custom_verilog()`. Ports are
    created on demand via `write_port()` / `read_port()` / `rw_port()` (+ optional
    `reset_arm()`) and wired with `<<=` inside a primitive's `elaborate()` — no scalar
    sugar. Not user-facing; designs use `MemoryPrimitive` / `RamPrimitive` / `FIFOPrimitive`.

    Lean by design: this module holds only array state, the port records above, the `*_port()`
    appenders, `_ArrayIndex`, `init_sim_state`, and `step`. Read-under-write policy, mask
    *emission*, registered-read pipelining, and all `custom_verilog` live in `primitives/`.
    The reset arm stays here only because a broadcast clear is a storage-`step()` op that a
    primitive cannot compose in sim.
    """

    def __init__(self, elem_type: HDLType, depth: int, *,
                 init: Optional[Sequence[int]] = None,
                 name: Optional[str] = None):
        if name is None:
            name = infer_signal_name_from_assignment("mem", "mem", __file__)
        if depth <= 0:
            raise ValueError(f"_MemoryArray depth must be > 0; got {depth}")
        if init is not None and len(init) != depth:
            raise ValueError(
                f"_MemoryArray init must have length == depth ({depth}); got {len(init)}")
        super().__init__(elem_type, kind="mem", name=name)
        self.depth = depth
        self.init = list(init) if init is not None else None
        self.elem_type = elem_type
        self._addr_t = UInt(max(1, (depth - 1).bit_length()))
        self.write_ports: "list[_WritePort]" = []
        self.read_ports: "list[_ReadPort]" = []
        self.reset: "Optional[_ResetArm]" = None
        # Sim-only store: never emits its own Verilog. Tag self no-emit (each port is tagged in `_port`).
        self._no_emit_decl = True
        self._no_emit_drive = True

    # ----------------------------- port factory -------------------------------

    def _port(self, suffix: str, typ: HDLType, kind: str = "wire") -> Signal:
        s = Signal(typ=typ, kind=kind, name=f"{self.name}__{suffix}")
        s._memory_parent = self
        s._port_suffix = suffix
        s._no_emit_decl = True
        s._no_emit_drive = True
        return s

    def _mask_port(self, prefix: str, mask_chunks: int) -> "tuple[Optional[Signal], int]":
        if not mask_chunks or mask_chunks <= 1:
            return None, 0
        w = self.elem_type.width
        if w % mask_chunks != 0:
            raise ValueError(f"mask_chunks={mask_chunks} must divide elem width {w}")
        mask = self._port(f"{prefix}_mask", UInt(mask_chunks))
        mask._driver = Const((1 << mask_chunks) - 1, UInt(mask_chunks))  # default: all chunks enabled
        return mask, w // mask_chunks

    def write_port(self, mask_chunks: int = 0) -> "_WritePort":
        i = len(self.write_ports)
        addr = self._port(f"w{i}_addr", self._addr_t)
        data = self._port(f"w{i}_data", self.elem_type)
        enable = self._port(f"w{i}_en", Bool())
        enable._driver = Const(1, Bool())
        mask, chunk_w = self._mask_port(f"w{i}", mask_chunks)
        wp = _WritePort(addr, data, enable, mask, chunk_w)
        self.write_ports.append(wp)
        return wp

    def read_port(self) -> "_ReadPort":
        i = len(self.read_ports)
        addr = self._port(f"r{i}_addr", self._addr_t)
        data = self._port(f"r{i}_data", self.elem_type)
        data._driver = _ArrayIndex(self, addr, self.elem_type)
        rp = _ReadPort(addr, data)
        self.read_ports.append(rp)
        return rp

    def rw_port(self, mask_chunks: int = 0) -> "_RWPort":
        n = len(self.write_ports)
        addr    = self._port(f"rw{n}_addr", self._addr_t)
        data_in = self._port(f"rw{n}_din", self.elem_type)
        write   = self._port(f"rw{n}_write", Bool()); write._driver = Const(0, Bool())
        enable  = self._port(f"rw{n}_en", Bool());    enable._driver = Const(1, Bool())
        # Write sub-port gated by `enable & write` — composed from the core `&` op, so `step`
        # needs no rw-specific branch (it just sees another write port).
        we = self._port(f"rw{n}_we", Bool())
        we._driver = enable & write
        mask, chunk_w = self._mask_port(f"rw{n}", mask_chunks)
        self.write_ports.append(_WritePort(addr, data_in, we, mask, chunk_w))
        # Async read sub-port on the same addr (combinational `mem[addr]`). A registered rw
        # read, if wanted, is a capture Register composed in the primitive.
        data_out = self._port(f"rw{n}_dout", self.elem_type)
        data_out._driver = _ArrayIndex(self, addr, self.elem_type)
        self.read_ports.append(_ReadPort(addr, data_out))
        return _RWPort(addr, data_in, write, enable, data_out, mask)

    def reset_arm(self) -> "_ResetArm":
        enable = self._port("rst_en", Bool())
        value = self._port("rst_val", self.elem_type)
        value._driver = Const(0, self.elem_type)
        self.reset = _ResetArm(enable, value)
        return self.reset

    # ----------------------------- introspection ------------------------------

    def _iter_ports(self) -> "list[Signal]":
        out: "list[Signal]" = []
        for wp in self.write_ports:
            out += [wp.addr, wp.data, wp.enable]
            if wp.mask is not None:
                out.append(wp.mask)
        for rp in self.read_ports:
            out += [rp.addr, rp.data]
        if self.reset is not None:
            out += [self.reset.enable, self.reset.value]
        return out

    # ----------------------------- simulator hooks ----------------------------

    def init_sim_state(self) -> "list[int]":
        if self.init is None:
            return [0] * self.depth
        from spire.simulator import _to_bits  # local import (lazy to avoid cycles)
        return [_to_bits(v, self.typ.width) for v in self.init]

    def step(self, sim) -> None:
        """One clock-edge update: reset arm (priority) then writes (last-write-wins order).

        Reads are combinational (`_ArrayIndex`); a registered read is a capture `Register`
        in the primitive, whose next-state the simulator evaluates *before* calling `step`
        (so it samples pre-edge memory → readFirst). The store therefore only commits writes.
        """
        from spire.simulator import _to_bits  # lazy, avoids cycles
        ev = sim._eval_signal_bits
        arr = sim._mem_state[id(self)]
        w = self.typ.width

        if self.reset is not None and ev(self.reset.enable) & 1:
            rv = _to_bits(ev(self.reset.value), w)
            for i in range(self.depth):
                arr[i] = rv
        else:
            for wp in self.write_ports:
                if not (ev(wp.enable) & 1):
                    continue
                a = _to_bits(ev(wp.addr), wp.addr.typ.width)
                if not (0 <= a < self.depth):
                    continue
                d = _to_bits(ev(wp.data), w)
                if wp.mask is None:
                    arr[a] = d
                else:
                    m = ev(wp.mask)
                    cur = arr[a]
                    for c in range(w // wp.chunk_w):
                        if (m >> c) & 1:
                            bits = ((1 << wp.chunk_w) - 1) << (c * wp.chunk_w)
                            cur = (cur & ~bits) | (d & bits)
                    arr[a] = cur


class _ArrayIndex(Expr):
    """Leaf Expr that emits ``mem.name[addr_wire.name]`` in verilog.

    Used as the ``_driver`` of an async-read memory's ``read_data`` wire. Walkers treat this as a leaf (no
    children): the address signal is reached through the store's port traversal, not through this Expr's fields.
    The simulator's ``visit_array_index`` reads from ``_mem_state``.
    """

    def __init__(self, mem: "_MemoryArray", addr_wire: Signal, typ: HDLType):
        self.mem = mem
        self.addr_wire = addr_wire
        self.typ = typ

    def to_verilog(self) -> str:
        return f"{self.mem.name}[{self.addr_wire.name}]"
