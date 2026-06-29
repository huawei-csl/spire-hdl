"""``MemoryPrimitive_via_reg`` — register-bank memory primitive (legacy reference).

This is the original ``MemoryPrimitive`` implementation: the Python sim model
(``elaborate()``) is a per-cell ``Register`` bank plus a linear mux tree, which
is **O(depth)** per access. The canonical ``MemoryPrimitive`` (in
``primitive_memory.py``) now backs simulation with the core's O(1) ``_MemoryArray``
instead; this variant is kept for comparison / fallback.

The synthesisable Verilog (``custom_verilog()``) is identical between the two —
both emit the same Yosys-inferable ``reg [W-1:0] mem[0:D-1]`` idiom.
"""

from __future__ import annotations

from typing import Optional, Sequence

from spire.expr import (
    Bool,
    Const,
    Register,
    Signal,
    UInt,
    mux,
)
from spire.component import CustomVerilogComponent
from spire.io_record import IORecord, Input, Output
from spire.primitives.primitive_memory import _elem_bit_width, _next_uid


class MemoryPrimitive_via_reg(CustomVerilogComponent):
    """Register-bank memory primitive (O(depth) sim). See module docstring.

    Same constructor and ``.io`` surface as ``MemoryPrimitive``; only the sim
    model differs.
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
            write_addr   = Input(UInt(self._addr_w)),
            write_data   = Input(UInt(self._elem_w)),
            write_enable = Input(Bool()),
            read_addr    = Input(UInt(self._addr_w)),
            read_data    = Output(UInt(self._elem_w)),
        )
        if with_reset_arm:
            kwargs["reset_enable"] = Input(Bool())
        if registered_read:
            kwargs["read_enable"] = Input(Bool())

        self.io = IORecord(**kwargs)
        self.elaborate()

    @property
    def name(self) -> str:
        return f"MemoryPrimitive_via_reg_{self._uid}"

    # --------------------------------------------------------- sim model

    def elaborate(self) -> None:
        """Python sim model: register file + explicit mux tree (O(depth) per cycle)."""
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
        lines.append(f"  // --- MemoryPrimitive_via_reg (uid={self._uid}, depth={D}, width={W}) ---")
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
