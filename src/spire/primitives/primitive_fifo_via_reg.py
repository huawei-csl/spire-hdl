"""``FIFOPrimitive_via_reg`` — register-bank synchronous FIFO (legacy reference).

Original ``FIFOPrimitive`` implementation: storage is an inline per-cell
``Register`` bank + mux tree (O(depth) sim). The primary ``FIFOPrimitive``
(in ``primitive_fifo.py``) now backs storage with the core's O(1) ``_MemoryArray``;
this variant is kept for comparison / fallback. The synthesisable Verilog is
behaviourally equivalent but NOT identical: here the memory write sits in the
``posedge clk or posedge rst`` block, which flattens the array to flip-flops
(no BRAM inference); the primary variant emits the split clock-only idiom.

Semantics (one-cycle read latency, *not* first-word-fallthrough) — see
``primitive_fifo.py`` for the full description.
"""

from __future__ import annotations

from typing import Optional

from spire.expr import (
    Bool,
    Register,
    Signal,
    UInt,
    Wire,
    mux,
)
from spire.component import CustomVerilogComponent
from spire.io_record import IORecord, Input, Output
from spire.primitives.primitive_memory import _elem_bit_width, _next_uid
from spire.primitives.primitive_fifo import FifoIO


class FIFOPrimitive_via_reg(CustomVerilogComponent):
    """Register-bank sync FIFO (O(depth) sim). Same ``.io`` as ``FIFOPrimitive``."""

    io: FifoIO

    def __init__(
        self,
        elem_type,
        depth: int,
        *,
        name: Optional[str] = None,
    ):
        if depth < 2:
            raise ValueError(f"FIFOPrimitive depth must be >= 2; got {depth}")
        if (depth & (depth - 1)) != 0:
            raise ValueError(f"FIFOPrimitive depth must be a power of two; got {depth}")

        self._elem_type = elem_type
        self._depth = depth
        self._elem_w = _elem_bit_width(elem_type)
        self._addr_w = (depth - 1).bit_length()    # log2(depth) for power-of-2 depth
        self._count_w = self._addr_w + 1           # count goes 0..depth inclusive
        self._uid = _next_uid()
        self._instance_name = name or f"fifo_{self._uid}"

        self.io = FifoIO(
            push  = Input(Bool()),
            pop   = Input(Bool()),
            din   = Input(UInt(self._elem_w)),
            dout  = Output(UInt(self._elem_w)),
            full  = Output(Bool()),
            empty = Output(Bool()),
            count = Output(UInt(self._count_w)),
        )
        self.elaborate()

    @property
    def name(self) -> str:
        return f"FIFOPrimitive_via_reg_{self._uid}"

    # --------------------------------------------------------- sim model

    def elaborate(self) -> None:
        """Python sim model: register file + pointers + count + dout register."""
        u = self._uid
        elem_w = self._elem_w
        addr_w = self._addr_w
        count_w = self._count_w
        depth = self._depth

        cells = [Register(UInt(elem_w), init=0, name=f"fcell_{u}_{i}") for i in range(depth)]
        wr_ptr   = Register(UInt(addr_w),  init=0, name=f"wrptr_{u}")
        rd_ptr   = Register(UInt(addr_w),  init=0, name=f"rdptr_{u}")
        count_r  = Register(UInt(count_w), init=0, name=f"cnt_{u}")
        dout_r   = Register(UInt(elem_w),  init=0, name=f"dout_r_{u}")

        # Combinational intermediates wrapped in Wires so the tag walker reaches them.
        empty_w = Wire(Bool(), name=f"empty_w_{u}")
        full_w  = Wire(Bool(), name=f"full_w_{u}")
        do_push = Wire(Bool(), name=f"do_push_{u}")
        do_pop  = Wire(Bool(), name=f"do_pop_{u}")

        empty_w <<= count_r == 0
        full_w  <<= count_r == depth
        do_push <<= self.io.push & ~full_w
        do_pop  <<= self.io.pop  & ~empty_w

        # Per-cell write
        for i, c in enumerate(cells):
            match = do_push & (wr_ptr == i)
            c <<= mux(match, self.io.din, c)

        # Pointers
        wr_ptr <<= mux(do_push, wr_ptr + 1, wr_ptr)
        rd_ptr <<= mux(do_pop,  rd_ptr + 1, rd_ptr)

        # Count
        push_only = do_push & ~do_pop
        pop_only  = do_pop  & ~do_push
        count_r <<= mux(push_only, count_r + 1,
                        mux(pop_only, count_r - 1, count_r))

        # Read: linear mux tree over cells, captured into dout_r when do_pop fires.
        read_expr = cells[0]
        for i in range(1, depth):
            read_expr = mux(rd_ptr == i, cells[i], read_expr)
        dout_r <<= mux(do_pop, read_expr, dout_r)

        # Outputs
        self.io.dout  <<= dout_r
        self.io.full  <<= full_w
        self.io.empty <<= empty_w
        self.io.count <<= count_r

    # ------------------------------------------------- synthesis verilog

    def custom_verilog(self) -> str:
        u = self._uid
        W = self._elem_w
        D = self._depth
        AW = self._addr_w
        CW = self._count_w
        n = self._instance_name

        push  = self.io.push.name
        pop   = self.io.pop.name
        din   = self.io.din.name
        dout  = self.io.dout.name
        full  = self.io.full.name
        empty = self.io.empty.name
        count = self.io.count.name

        mem    = f"{n}__mem"
        wr_ptr = f"{n}__wr"
        rd_ptr = f"{n}__rd"
        cnt    = f"{n}__cnt"
        dout_r = f"{n}__dout_r"
        ep_w   = f"{n}__empty_w"
        fl_w   = f"{n}__full_w"
        dpu    = f"{n}__do_push"
        dpo    = f"{n}__do_pop"

        L: list[str] = []
        L.append(f"  // --- FIFOPrimitive_via_reg (uid={u}, depth={D}, width={W}) ---")
        L.append(f"  reg [{W-1}:0] {mem}[0:{D-1}];")
        L.append(f"  reg [{AW-1}:0] {wr_ptr};")
        L.append(f"  reg [{AW-1}:0] {rd_ptr};")
        L.append(f"  reg [{CW-1}:0] {cnt};")
        L.append(f"  reg [{W-1}:0] {dout_r};")
        L.append(f"  wire {ep_w} = ({cnt} == 0);")
        L.append(f"  wire {fl_w} = ({cnt} == {D});")
        L.append(f"  wire {dpu} = {push} & ~{fl_w};")
        L.append(f"  wire {dpo} = {pop}  & ~{ep_w};")
        L.append(f"  always @(posedge clk or posedge rst) begin")
        L.append(f"    if (rst) begin")
        L.append(f"      {wr_ptr} <= 0;")
        L.append(f"      {rd_ptr} <= 0;")
        L.append(f"      {cnt}    <= 0;")
        L.append(f"      {dout_r} <= 0;")
        L.append(f"    end else begin")
        L.append(f"      if ({dpu}) begin")
        L.append(f"        {mem}[{wr_ptr}] <= {din};")
        L.append(f"        {wr_ptr} <= {wr_ptr} + 1;")
        L.append(f"      end")
        L.append(f"      if ({dpo}) begin")
        L.append(f"        {dout_r} <= {mem}[{rd_ptr}];")
        L.append(f"        {rd_ptr} <= {rd_ptr} + 1;")
        L.append(f"      end")
        L.append(f"      case ({{{dpu}, {dpo}}})")
        L.append(f"        2'b10: {cnt} <= {cnt} + 1;")
        L.append(f"        2'b01: {cnt} <= {cnt} - 1;")
        L.append(f"        default: ;")
        L.append(f"      endcase")
        L.append(f"    end")
        L.append(f"  end")
        L.append(f"  assign {dout}  = {dout_r};")
        L.append(f"  assign {full}  = {fl_w};")
        L.append(f"  assign {empty} = {ep_w};")
        L.append(f"  assign {count} = {cnt};")
        return "\n".join(L)
