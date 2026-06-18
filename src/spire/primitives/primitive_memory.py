"""``MemoryPrimitive`` — array memory built as a regular ``Component``.

Synthesisable Verilog comes from ``custom_verilog()`` (a Yosys-inferable RAM idiom).
Python simulation is backed by the core's O(1) ``_MemoryArray`` (Middle path B):
``elaborate()`` instantiates a sim-only store and wires this primitive's ``.io``
ports to it; the store emits no Verilog of its own (``custom_verilog()`` is the
single synthesis source). The two paths describe the same hardware; the framework
cannot prove equivalence — the tests do.

The legacy O(depth) register-bank model is preserved as ``MemoryPrimitive_via_reg``
(``primitive_memory_via_reg.py``) for comparison / fallback.

Composite element types are supported via user-side pack / unpack at the port
boundary. The port wires are always ``UInt(elem_w)``; callers do::

    class Bus(CompositeRecord):
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

from typing import Optional, Sequence

from spire.expr import (
    Bool,
    Const,
    Register,
    Signal,
    UInt,
    mux,
)
from spire.component import Component
from spire.io_record import IORecord, Input, Output
from spire.composite.base import HDLComposite
from spire.primitives._ram_template import ram_block


# Per-process counter that suffixes internal Verilog names so multiple primitive
# instances in the same parent module never collide on the inlined `mem` array,
# `_rd` reg, etc.
_PRIMITIVE_UID = 0


def _next_uid() -> int:
    global _PRIMITIVE_UID
    _PRIMITIVE_UID += 1
    return _PRIMITIVE_UID


def _elem_bit_width(elem_type) -> int:
    """Packed bit-width of an HDLType or HDLComposite (class or instance)."""
    # HDLType (UInt, SInt, Bool, …) — exposes `.width` directly
    if isinstance(elem_type, type) and issubclass(elem_type, HDLComposite):
        # Composite class — instantiate (wire-only template) to read packed width
        return elem_type().width
    if isinstance(elem_type, HDLComposite):
        return elem_type.width
    if hasattr(elem_type, "width"):
        return elem_type.width
    raise TypeError(
        f"elem_type must be HDLType or HDLComposite (class or instance); got {type(elem_type).__name__}"
    )


class MemoryPrimitive(Component):
    """Component-based memory primitive (replacement candidate for built-in ``Memory``).

    Parameters (all kwargs except ``elem_type`` and ``depth``):
      ``elem_type``        HDLType (e.g. ``UInt(8)``) **or** HDLComposite class / instance.
                           For composites, the user packs / unpacks at the port boundary.
      ``depth``            Number of entries.
      ``init``             Optional per-cell init bit-pattern (length must equal ``depth``).
      ``registered_read``  ``True`` → adds ``read_enable`` port and registers ``read_data``.
      ``with_reset_arm``   ``True`` → adds ``reset_enable`` input port; on assertion all
                           cells are loaded with ``reset_value`` (static int).
      ``reset_value``      Static integer used when ``reset_enable`` fires (default 0).
      ``name``             Optional instance name; used as the inlined Verilog mem array name.

    Ports (all UInt at the boundary; composites pack via ``to_bits()`` / ``from_bits()``):
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
        mask_chunks: int = 0,
        read_under_write: str = "readFirst",
        name: Optional[str] = None,
    ):
        if depth <= 0:
            raise ValueError(f"MemoryPrimitive depth must be > 0; got {depth}")
        if init is not None and len(init) != depth:
            raise ValueError(
                f"MemoryPrimitive init must have length == depth ({depth}); got {len(init)}"
            )
        if read_under_write not in ("readFirst", "writeFirst", "dontCare"):
            raise ValueError(f"read_under_write must be readFirst/writeFirst/dontCare; got {read_under_write!r}")
        if read_under_write == "writeFirst" and registered_read:
            raise ValueError("MemoryPrimitive: writeFirst is supported for async read only "
                             "(registered_read=False); use RamPrimitive for registered RUW.")

        self._elem_type = elem_type
        self._depth = depth
        self._init = list(init) if init is not None else None
        self._registered_read = registered_read
        self._with_reset_arm = with_reset_arm
        self._reset_value = reset_value
        self._read_under_write = read_under_write
        self._elem_w = _elem_bit_width(elem_type)
        if mask_chunks and mask_chunks > 1 and self._elem_w % mask_chunks != 0:
            raise ValueError(f"mask_chunks={mask_chunks} must divide elem width {self._elem_w}")
        self._mask_chunks = mask_chunks if (mask_chunks and mask_chunks > 1) else 0
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
        if self._mask_chunks:
            kwargs["write_mask"] = Input(UInt(self._mask_chunks))
        if with_reset_arm:
            kwargs["reset_enable"] = Input(Bool())
        if registered_read:
            kwargs["read_enable"] = Input(Bool())

        self.io = IORecord(**kwargs)
        self.elaborate()

    @property
    def name(self) -> str:
        return f"MemoryPrimitive_{self._uid}"

    # --------------------------------------------------------- sim model

    def elaborate(self) -> None:
        """Python sim model: wire this primitive's ``.io`` to a core ``_MemoryArray``.

        The store gives O(1) simulation (``_mem_state`` array + ``step`` + the
        ``_ArrayIndex`` read leaf) and emits no Verilog — ``custom_verilog()``
        below is the sole synthesis source.
        """
        from spire.memory import _MemoryArray

        m = _MemoryArray(UInt(self._elem_w), self._depth, init=self._init, name=self._instance_name)
        wp = m.write_port(mask_chunks=self._mask_chunks)
        wp.addr   <<= self.io.write_addr
        wp.data   <<= self.io.write_data
        wp.enable <<= self.io.write_enable
        if self._mask_chunks:
            wp.mask <<= self.io.write_mask
        if self._with_reset_arm:
            ra = m.reset_arm()
            ra.enable <<= self.io.reset_enable
            if self._reset_value != 0:
                ra.value <<= Const(self._reset_value, UInt(self._elem_w))
        rp = m.read_port()
        rp.addr <<= self.io.read_addr
        if self._registered_read:
            # Registered read = capture the async read into a Register, composed here (not in
            # the store). The simulator evaluates this reg's next-state before the store's
            # step(), so it samples pre-edge memory → readFirst. The reg is auto-tagged
            # no-emit (reachable from read_data); custom_verilog emits the BRAM-idiom rdata reg.
            rd = Register(UInt(self._elem_w), init=0, name=f"{self._instance_name}__rdata")
            rd <<= mux(self.io.read_enable, rp.data, rd)
            self.io.read_data <<= rd
        elif self._read_under_write == "writeFirst":  # async only (validated in __init__)
            collision = self.io.write_enable & (self.io.write_addr == self.io.read_addr)
            self.io.read_data <<= mux(collision, self.io.write_data, rp.data)
        else:
            self.io.read_data <<= rp.data
        self._store = m

    # ------------------------------------------------- synthesis verilog

    def custom_verilog(self) -> str:
        W = self._elem_w
        write = dict(addr=self.io.write_addr.name, data=self.io.write_data.name,
                     en=self.io.write_enable.name)
        if self._mask_chunks:
            write.update(mask=self.io.write_mask.name, mask_chunks=self._mask_chunks,
                         chunk_w=W // self._mask_chunks)
        read = dict(addr=self.io.read_addr.name, out=self.io.read_data.name,
                    registered=self._registered_read)
        if self._registered_read:
            read["en"] = self.io.read_enable.name
        if self._read_under_write == "writeFirst":
            read["ruw"] = "writeFirst"
            read["fwd"] = dict(en=self.io.write_enable.name, addr=self.io.write_addr.name,
                               data=self.io.write_data.name)
        reset = None
        if self._with_reset_arm:
            reset = dict(en=self.io.reset_enable.name, val=f"{W}'d{self._reset_value}")
        return ram_block(
            name=self._instance_name, depth=self._depth, elem_w=W,
            writes=[write], reads=[read], reset=reset, init=self._init,
            comment=f"--- MemoryPrimitive (uid={self._uid}, depth={self._depth}, width={W}) ---",
        )
