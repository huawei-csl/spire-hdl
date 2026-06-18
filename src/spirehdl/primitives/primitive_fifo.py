"""``FIFOPrimitive`` — standard synchronous FIFO built as a ``Component``.

Synthesisable Verilog comes from ``custom_verilog()``; Python simulation runs
``elaborate()``'s register-file reference model. Composite element types are
supported via user-side pack / unpack at the port boundary.

Semantics (one-cycle read latency, *not* first-word-fallthrough):
  * Push when ``push & ~full``: ``din`` is written to ``mem[wr_ptr]`` and ``wr_ptr`` advances.
  * Pop when ``pop & ~empty``: ``mem[rd_ptr]`` is captured into ``dout`` and ``rd_ptr`` advances.
    Result: after a pop on cycle T, ``dout`` shows the popped value on cycle T+1.
  * Simultaneous push + pop on a non-empty/non-full FIFO leaves ``count`` unchanged.
  * Underflow (pop while empty) and overflow (push while full) are silently ignored — the
    request gates itself off via ``~empty`` / ``~full``.

v1 requires ``depth`` to be a power of two so pointer wrap is free (no explicit modulo).

Storage is backed by the core's O(1) ``_MemoryArray`` (Middle path B): ``elaborate()``
uses it for the data array and keeps the pointers / count / flags / dout register as
ordinary ``Register``/``Wire`` logic. ``custom_verilog()`` emits one self-contained
block (the store contributes no Verilog of its own).

The legacy O(depth) register-bank model is preserved as ``FIFOPrimitive_via_reg``
(``primitive_fifo_via_reg.py``) for comparison / fallback.
"""

from __future__ import annotations

from typing import Optional

from spirehdl.spirehdl import (
    Bool,
    Register,
    Signal,
    UInt,
    Wire,
    mux,
)
from spirehdl.spirehdl_module import Component
from spirehdl.io_record import IORecord, Input, Output
from spirehdl.primitives.primitive_memory import _elem_bit_width, _next_uid


class FIFOPrimitive(Component):
    """Standard sync FIFO with registered output (one-cycle read latency).

    Parameters:
      ``elem_type``  HDLType or HDLComposite (class / instance).
      ``depth``      Power of two, ``>= 2``.
      ``name``       Optional instance name; used as Verilog mem-array prefix.

    Ports:
      | name      | dir | width                        |
      |-----------|-----|------------------------------|
      | ``push``    | in  | 1                            |
      | ``pop``     | in  | 1                            |
      | ``din``     | in  | ``elem_w``                   |
      | ``dout``    | out | ``elem_w`` (registered head) |
      | ``full``    | out | 1                            |
      | ``empty``   | out | 1                            |
      | ``count``   | out | ``log2(depth)+1``            |
    """

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

        self.io = IORecord(
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
        return f"FIFOPrimitive_{self._uid}"

    # --------------------------------------------------------- sim model

    def elaborate(self) -> None:
        """Python sim model: O(1) ``_MemoryArray`` data array + pointer/count/flag logic.

        Combinational intermediates are wrapped in tagged ``Wire``s so the
        ``_apply_custom_verilog_tags`` walker marks them no-emit (otherwise CSE
        might lift a duplicate sub-expression into an auto-generated wire that the
        emitter does *not* skip, leaking into the Verilog output). The store reads
        ``mem[rd_ptr]`` combinationally (pre-edge) and writes ``mem[wr_ptr]`` at the
        edge — matching the via_reg model and the verilog NBA semantics.
        """
        from spirehdl.spirehdl_memory import _MemoryArray

        u = self._uid
        elem_w = self._elem_w
        addr_w = self._addr_w
        count_w = self._count_w
        depth = self._depth

        store = _MemoryArray(UInt(elem_w), depth, name=f"{self._instance_name}__store")
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

        # Storage: write at wr_ptr (gated by do_push), async read at rd_ptr.
        wport = store.write_port()
        wport.addr   <<= wr_ptr
        wport.data   <<= self.io.din
        wport.enable <<= do_push
        rport = store.read_port()
        rport.addr <<= rd_ptr

        # Pointers
        wr_ptr <<= mux(do_push, wr_ptr + 1, wr_ptr)
        rd_ptr <<= mux(do_pop,  rd_ptr + 1, rd_ptr)

        # Count
        push_only = do_push & ~do_pop
        pop_only  = do_pop  & ~do_push
        count_r <<= mux(push_only, count_r + 1,
                        mux(pop_only, count_r - 1, count_r))

        # Read: capture mem[rd_ptr] (pre-edge) into dout_r when do_pop fires.
        dout_r <<= mux(do_pop, rport.data, dout_r)

        # Outputs
        self.io.dout  <<= dout_r
        self.io.full  <<= full_w
        self.io.empty <<= empty_w
        self.io.count <<= count_r
        self._store = store

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
        L.append(f"  // --- FIFOPrimitive (uid={u}, depth={D}, width={W}) ---")
        L.append(f"  reg [{W-1}:0] {mem}[0:{D-1}];")
        L.append(f"  reg [{AW-1}:0] {wr_ptr};")
        L.append(f"  reg [{AW-1}:0] {rd_ptr};")
        L.append(f"  reg [{CW-1}:0] {cnt};")
        L.append(f"  reg [{W-1}:0] {dout_r};")
        L.append(f"  wire {ep_w} = ({cnt} == 0);")
        L.append(f"  wire {fl_w} = ({cnt} == {D});")
        L.append(f"  wire {dpu} = {push} & ~{fl_w};")
        L.append(f"  wire {dpo} = {pop}  & ~{ep_w};")
        # Storage write lives in its OWN clock-only always block (no `posedge rst` in the
        # sensitivity list). yosys can only infer a memory for `mem` when its write port is
        # single-clock; with the write inside the async-reset block below it would flatten to
        # D discrete flip-flops. The async read `mem[rd_ptr]` (captured into dout_r) then
        # becomes the inferred memory's read port. Both blocks are posedge-clk NBA, so the
        # write/read ordering (readFirst) is unchanged.
        L.append(f"  always @(posedge clk) begin")
        L.append(f"    if ({dpu}) {mem}[{wr_ptr}] <= {din};")
        L.append(f"  end")
        # Control state (pointers / count / registered output) keeps the async reset.
        L.append(f"  always @(posedge clk or posedge rst) begin")
        L.append(f"    if (rst) begin")
        L.append(f"      {wr_ptr} <= 0;")
        L.append(f"      {rd_ptr} <= 0;")
        L.append(f"      {cnt}    <= 0;")
        L.append(f"      {dout_r} <= 0;")
        L.append(f"    end else begin")
        L.append(f"      if ({dpu}) {wr_ptr} <= {wr_ptr} + 1;")
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
