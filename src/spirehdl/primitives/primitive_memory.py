"""``MemoryPrimitive`` — array-of-registers memory built as a regular ``Component``.

Synthesisable Verilog comes from ``custom_verilog()`` (a Yosys-inferable RAM idiom,
identical in shape to the one the built-in ``Memory`` emits). Python simulation
runs ``elaborate()``'s reference model: per-cell ``Register`` plus a linear mux tree
for reads and per-cell conditional updates for writes. The two paths describe the
same hardware; the framework cannot prove equivalence — the tests do.

Aggregate element types are supported via user-side pack / unpack at the port
boundary. The port wires are always ``UInt(elem_w)``; callers do::

    class Bus(AggregateRecord):
        data  = Wire(UInt(8))
        valid = Wire(UInt(1))

    mem = MemoryPrimitive(Bus, depth=16).make_internal()
    # write
    bus_in = Bus()
    bus_in.data  <<= some_data
    bus_in.valid <<= some_valid
    mem.io.write_data <<= bus_in.to_bits()
    # read
    out_bus = Bus()
    out_bus <<= mem.io.read_data
    data_out  <<= out_bus.data
    valid_out <<= out_bus.valid
"""

from __future__ import annotations

from dataclasses import make_dataclass
from typing import Optional, Sequence

from spirehdl.spirehdl import (
    Bool,
    Const,
    Register,
    Signal,
    UInt,
    mux,
)
from spirehdl.spirehdl_module import Component
from spirehdl.aggregate.hdl_aggregate import HDLAggregate


# Per-process counter that suffixes internal Verilog names so multiple primitive
# instances in the same parent module never collide on the inlined `mem` array,
# `_rd` reg, etc.
_PRIMITIVE_UID = 0


def _next_uid() -> int:
    global _PRIMITIVE_UID
    _PRIMITIVE_UID += 1
    return _PRIMITIVE_UID


def _elem_bit_width(elem_type) -> int:
    """Packed bit-width of an HDLType or HDLAggregate (class or instance)."""
    # HDLType (UInt, SInt, Bool, …) — exposes `.width` directly
    if isinstance(elem_type, type) and issubclass(elem_type, HDLAggregate):
        # Aggregate class — instantiate (wire-only template) to read packed width
        return elem_type().width
    if isinstance(elem_type, HDLAggregate):
        return elem_type.width
    if hasattr(elem_type, "width"):
        return elem_type.width
    raise TypeError(
        f"elem_type must be HDLType or HDLAggregate (class or instance); got {type(elem_type).__name__}"
    )


class MemoryPrimitive(Component):
    """Component-based memory primitive (replacement candidate for built-in ``Memory``).

    Parameters (all kwargs except ``elem_type`` and ``depth``):
      ``elem_type``        HDLType (e.g. ``UInt(8)``) **or** HDLAggregate class / instance.
                           For aggregates, the user packs / unpacks at the port boundary.
      ``depth``            Number of entries.
      ``init``             Optional per-cell init bit-pattern (length must equal ``depth``).
      ``registered_read``  ``True`` → adds ``read_enable`` port and registers ``read_data``.
      ``with_reset_arm``   ``True`` → adds ``reset_enable`` input port; on assertion all
                           cells are loaded with ``reset_value`` (static int).
      ``reset_value``      Static integer used when ``reset_enable`` fires (default 0).
      ``name``             Optional instance name; used as the inlined Verilog mem array name.

    Ports (all UInt at the boundary; aggregates pack via ``to_bits()`` / ``from_bits()``):
      | name           | dir | when                |
      |----------------|-----|---------------------|
      | ``write_addr``   | in  | always              |
      | ``write_data``   | in  | always              |
      | ``write_enable`` | in  | always              |
      | ``read_addr``    | in  | always              |
      | ``read_data``    | out | always              |
      | ``reset_enable`` | in  | with_reset_arm      |
      | ``read_enable``  | in  | registered_read     |
    """

    def __init__(
        self,
        elem_type,
        depth: int,
        *,
        init: Optional[Sequence[int]] = None,
        registered_read: bool = False,
        with_reset_arm: bool = False,
        reset_value: int = 0,
        name: Optional[str] = None,
    ):
        if depth <= 0:
            raise ValueError(f"MemoryPrimitive depth must be > 0; got {depth}")
        if init is not None and len(init) != depth:
            raise ValueError(
                f"MemoryPrimitive init must have length == depth ({depth}); got {len(init)}"
            )

        self._elem_type = elem_type
        self._depth = depth
        self._init = list(init) if init is not None else None
        self._registered_read = registered_read
        self._with_reset_arm = with_reset_arm
        self._reset_value = reset_value
        self._elem_w = _elem_bit_width(elem_type)
        self._addr_w = max(1, (depth - 1).bit_length())
        self._uid = _next_uid()
        self._instance_name = name or f"mem_{self._uid}"

        kwargs = dict(
            write_addr   = Signal("write_addr",   UInt(self._addr_w), "input"),
            write_data   = Signal("write_data",   UInt(self._elem_w), "input"),
            write_enable = Signal("write_enable", Bool(),             "input"),
            read_addr    = Signal("read_addr",    UInt(self._addr_w), "input"),
            read_data    = Signal("read_data",    UInt(self._elem_w), "output"),
        )
        if with_reset_arm:
            kwargs["reset_enable"] = Signal("reset_enable", Bool(), "input")
        if registered_read:
            kwargs["read_enable"] = Signal("read_enable", Bool(), "input")

        IO = make_dataclass("MemoryPrimitiveIO", [(k, Signal) for k in kwargs.keys()])
        self.io = IO(**kwargs)
        self.elaborate()

    @property
    def name(self) -> str:
        return f"MemoryPrimitive_{self._uid}"

    # --------------------------------------------------------- sim model

    def elaborate(self) -> None:
        """Python sim model: register file + explicit mux tree.

        This describes the same hardware as ``custom_verilog()`` but is shaped so the
        Spire simulator can evaluate it directly (no special-case handler). It is
        ``O(depth)`` per cycle in expression evaluations — acceptable for the
        small-depth tests; would need attention for very large memories.
        """
        cells = []
        for i in range(self._depth):
            init_v = self._init[i] if self._init is not None else 0
            c = Register(UInt(self._elem_w), init=init_v, name=f"cell_{self._uid}_{i}")
            cells.append(c)

        # write (with optional reset arm taking priority)
        for i, c in enumerate(cells):
            write_match = self.io.write_enable & (self.io.write_addr == i)
            next_v = mux(write_match, self.io.write_data, c)
            if self._with_reset_arm:
                next_v = mux(self.io.reset_enable,
                             Const(self._reset_value, UInt(self._elem_w)),
                             next_v)
            c <<= next_v

        # read: linear mux tree over all cells, indexed by read_addr
        read_expr = cells[0]
        for i in range(1, self._depth):
            read_expr = mux(self.io.read_addr == i, cells[i], read_expr)

        if self._registered_read:
            rd = Register(UInt(self._elem_w), init=0, name=f"rd_{self._uid}")
            rd <<= mux(self.io.read_enable, read_expr, rd)
            self.io.read_data <<= rd
        else:
            self.io.read_data <<= read_expr

    # ------------------------------------------------- synthesis verilog

    def custom_verilog(self) -> str:
        n = self._instance_name
        W = self._elem_w
        D = self._depth
        wa = self.io.write_addr.name
        wd = self.io.write_data.name
        we = self.io.write_enable.name
        ra = self.io.read_addr.name
        rd_port = self.io.read_data.name

        lines: list[str] = []
        lines.append(f"  // --- MemoryPrimitive (uid={self._uid}, depth={D}, width={W}) ---")
        lines.append(f"  reg [{W-1}:0] {n}[0:{D-1}];")

        if self._registered_read:
            rd_internal = f"{n}__rd"
            lines.append(f"  reg [{W-1}:0] {rd_internal};")

        if self._init is not None:
            lines.append("  initial begin")
            for i, v in enumerate(self._init):
                lines.append(f"    {n}[{i}] = {W}'d{v};")
            lines.append("  end")

        # clock-only always (yosys-recognised memory inference idiom)
        lines.append("  always @(posedge clk) begin")
        if self._with_reset_arm:
            rst_en = self.io.reset_enable.name
            lines.append(f"    if ({rst_en}) begin")
            for i in range(D):
                lines.append(f"      {n}[{i}] <= {W}'d{self._reset_value};")
            lines.append(f"    end else if ({we}) begin")
            lines.append(f"      {n}[{wa}] <= {wd};")
            lines.append("    end")
        else:
            lines.append(f"    if ({we}) begin")
            lines.append(f"      {n}[{wa}] <= {wd};")
            lines.append("    end")

        if self._registered_read:
            re_name = self.io.read_enable.name
            rd_internal = f"{n}__rd"
            lines.append(f"    if ({re_name}) begin")
            lines.append(f"      {rd_internal} <= {n}[{ra}];")
            lines.append("    end")
        lines.append("  end")

        if self._registered_read:
            rd_internal = f"{n}__rd"
            lines.append(f"  assign {rd_port} = {rd_internal};")
        else:
            lines.append(f"  assign {rd_port} = {n}[{ra}];")

        return "\n".join(lines)
