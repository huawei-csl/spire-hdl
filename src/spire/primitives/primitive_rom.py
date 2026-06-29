"""``RomPrimitive`` — read-only memory (init-backed lookup table) as a ``Component``.

A ROM is just a `_MemoryArray` with ``init`` values and a single read port — **no write
port, no reset arm** — so the array keeps its initial contents forever. Built entirely in
user-space (no core change): the write-less array's ``step()`` is a no-op, so simulation
holds the init values, and ``custom_verilog`` emits a ``reg [W-1:0] rom[0:D-1];`` + an
``initial`` block (and, for a registered read, a clock-only rdata capture).

This is ergonomic sugar over ``MemoryPrimitive(..., init=…)`` with its write port tied off:
``RomPrimitive`` simply drops the write/mask/reset ports from the interface.

Composite element types are supported via the same pack/unpack-at-the-boundary convention
as the other primitives (``init`` entries are packed bit-patterns).
"""

from __future__ import annotations

from typing import Optional, Sequence

from spire.expr import Bool, Register, Signal, UInt, mux
from spire.component import CustomVerilogComponent
from spire.io_record import IORecord, Input, Output
from spire.primitives.primitive_memory import _elem_bit_width, _next_uid
from spire.primitives._ram_template import ram_block


class RomPrimitive(CustomVerilogComponent):
    """Read-only memory.

    Parameters:
      ``elem_type``        HDLType (e.g. ``UInt(8)``) or HDLComposite (class / instance).
      ``depth``            Number of entries.
      ``init``             Per-cell contents (length must equal ``depth``). Required.
      ``registered_read``  ``True`` → 1-cycle registered read + a ``read_enable`` port
                           (low holds the previous output). Default async (combinational).
      ``name``             Optional instance name; used as the inlined Verilog array name.

    Ports (``rom.io.*``):
      | name          | dir | when               |
      |---------------|-----|--------------------|
      | ``read_addr`` | in  | always             |
      | ``read_data`` | out | always             |
      | ``read_enable`` | in | ``registered_read`` |
    """

    def __init__(
        self,
        elem_type,
        depth: int,
        init: Sequence[int],
        *,
        registered_read: bool = False,
        name: Optional[str] = None,
    ):
        if depth <= 0:
            raise ValueError(f"RomPrimitive depth must be > 0; got {depth}")
        if init is None or len(init) != depth:
            raise ValueError(
                f"RomPrimitive requires init of length == depth ({depth}); "
                f"got {None if init is None else len(init)}")

        self._elem_type = elem_type
        self._depth = depth
        self._init = list(init)
        self._registered_read = registered_read
        self._elem_w = _elem_bit_width(elem_type)
        self._addr_w = max(1, (depth - 1).bit_length())
        self._uid = _next_uid()
        self._instance_name = name or f"rom_{self._uid}"

        kwargs = dict(
            read_addr = Input(UInt(self._addr_w)),
            read_data = Output(UInt(self._elem_w)),
        )
        if registered_read:
            kwargs["read_enable"] = Input(Bool())

        self.io = IORecord(**kwargs)
        self.elaborate()

    @property
    def name(self) -> str:
        return f"RomPrimitive_{self._uid}"

    # --------------------------------------------------------- sim model

    def elaborate(self) -> None:
        from spire.memory import _MemoryArray

        m = _MemoryArray(UInt(self._elem_w), self._depth, init=self._init, name=self._instance_name)
        rp = m.read_port()                          # async read; no write_port() → contents are fixed
        rp.addr <<= self.io.read_addr
        if self._registered_read:
            rd = Register(UInt(self._elem_w), init=0, name=f"{self._instance_name}__rdata")
            rd <<= mux(self.io.read_enable, rp.data, rd)
            self.io.read_data <<= rd
        else:
            self.io.read_data <<= rp.data
        self._store = m

    # ------------------------------------------------- synthesis verilog

    def custom_verilog(self) -> str:
        read = dict(addr=self.io.read_addr.name, out=self.io.read_data.name,
                    registered=self._registered_read)
        if self._registered_read:
            read["en"] = self.io.read_enable.name
        return ram_block(
            name=self._instance_name, depth=self._depth, elem_w=self._elem_w,
            writes=[], reads=[read], reset=None, init=self._init,
            comment=f"--- RomPrimitive (uid={self._uid}, depth={self._depth}, width={self._elem_w}) ---",
        )
