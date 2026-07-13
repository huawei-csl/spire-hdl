"""``RamPrimitive`` — general multi-port RAM built as a regular ``Component``.

Supports the SpinalHDL-style shapes that the single-port ``MemoryPrimitive`` can't express:
multiple write ports, multiple read ports, true dual-port (``rw_ports`` = read/write ports),
write masks, and read-under-write forwarding — all over **one shared array** via the core's
``_MemoryArray`` port factory (O(1) sim) with synthesis from ``_ram_template.ram_block``.

Ports (indexed; all ``UInt`` at the boundary, composites pack via ``to_bits``/``from_bits``):
  write k:  ``w{k}_addr`` ``w{k}_data`` ``w{k}_en`` (+ ``w{k}_mask`` if masked)
  read  k:  ``r{k}_addr`` ``r{k}_data`` (+ ``r{k}_en`` if ``registered_read``)
  rw    k:  ``rw{k}_addr`` ``rw{k}_din`` ``rw{k}_write`` ``rw{k}_en`` ``rw{k}_dout``
            (+ ``rw{k}_mask`` if masked). Writes ``din`` when ``en & write``; ``dout`` reads
            ``mem[addr]`` (readFirst, or writeFirst-forwarded from this port's own write).

read_under_write applies to ``rw`` ports (collision is well-defined — the port's own write).
Plain read ports are readFirst. Multiple writers to one address resolve last-write-wins
(registration order), matching verilog NBA.
"""

from __future__ import annotations

from typing import Optional, Sequence

from spire.expr import Bool, Register, Signal, UInt, mux
from spire.component import CustomVerilogComponent
from spire.io_record import IORecord, Input, Output
from spire.primitives.primitive_memory import _elem_bit_width, _next_uid
from spire.primitives._ram_template import check_ruw_mask, norm_elem_values, ram_block


class RamPrimitive(CustomVerilogComponent):
    def __init__(
        self,
        elem_type,
        depth: int,
        *,
        num_read_ports: int = 1,
        num_write_ports: int = 1,
        rw_ports: int = 0,
        mask_chunks: int = 0,
        read_under_write: str = "readFirst",
        registered_read: bool = False,
        init: Optional[Sequence[int]] = None,
        name: Optional[str] = None,
    ):
        if depth <= 0:
            raise ValueError(f"RamPrimitive depth must be > 0; got {depth}")
        if init is not None and len(init) != depth:
            raise ValueError(f"RamPrimitive init must have length == depth ({depth}); got {len(init)}")
        if read_under_write not in ("readFirst", "writeFirst", "dontCare"):
            raise ValueError(f"read_under_write must be readFirst/writeFirst/dontCare; got {read_under_write!r}")
        if num_read_ports + rw_ports == 0:
            raise ValueError("RamPrimitive needs at least one read or rw port")
        if num_write_ports + rw_ports == 0:
            raise ValueError("RamPrimitive needs at least one write or rw port")

        self._elem_type = elem_type
        self._depth = depth
        self._init = list(init) if init is not None else None
        self._nr = num_read_ports
        self._nw = num_write_ports
        self._nrw = rw_ports
        self._ruw = read_under_write
        self._registered_read = registered_read
        self._elem_w = _elem_bit_width(elem_type)
        if self._init is not None:
            self._init = norm_elem_values(self._init, width=self._elem_w,
                                          signed=getattr(elem_type, "signed", False), what="init")
        check_ruw_mask(read_under_write, mask_chunks)
        if mask_chunks and mask_chunks > 1 and self._elem_w % mask_chunks != 0:
            raise ValueError(f"mask_chunks={mask_chunks} must divide elem width {self._elem_w}")
        self._mask_chunks = mask_chunks if (mask_chunks and mask_chunks > 1) else 0
        self._addr_w = max(1, (depth - 1).bit_length())
        self._uid = _next_uid()
        self._instance_name = name or f"ram_{self._uid}"

        AW, EW, MW = UInt(self._addr_w), UInt(self._elem_w), UInt(self._mask_chunks or 1)
        kw: dict = {}
        for k in range(self._nw):
            kw[f"w{k}_addr"] = Input(AW)
            kw[f"w{k}_data"] = Input(EW)
            kw[f"w{k}_en"]   = Input(Bool())
            if self._mask_chunks:
                kw[f"w{k}_mask"] = Input(MW)
        for k in range(self._nr):
            kw[f"r{k}_addr"] = Input(AW)
            kw[f"r{k}_data"] = Output(EW)
            if self._registered_read:
                kw[f"r{k}_en"] = Input(Bool())
        for k in range(self._nrw):
            kw[f"rw{k}_addr"]  = Input(AW)
            kw[f"rw{k}_din"]   = Input(EW)
            kw[f"rw{k}_write"] = Input(Bool())
            kw[f"rw{k}_en"]    = Input(Bool())
            kw[f"rw{k}_dout"]  = Output(EW)
            if self._mask_chunks:
                kw[f"rw{k}_mask"] = Input(MW)

        self.io = IORecord(**kw)
        self.elaborate()

    @property
    def name(self) -> str:
        return f"RamPrimitive_{self._uid}"

    def _io(self, n):
        return getattr(self.io, n)

    # --------------------------------------------------------- sim model

    def elaborate(self) -> None:
        from spire.memory import _MemoryArray

        m = _MemoryArray(UInt(self._elem_w), self._depth, init=self._init, name=self._instance_name)
        for k in range(self._nw):
            wp = m.write_port(mask_chunks=self._mask_chunks)
            wp.addr   <<= self._io(f"w{k}_addr")
            wp.data   <<= self._io(f"w{k}_data")
            wp.enable <<= self._io(f"w{k}_en")
            if self._mask_chunks:
                wp.mask <<= self._io(f"w{k}_mask")
        for k in range(self._nr):
            rp = m.read_port()
            rp.addr <<= self._io(f"r{k}_addr")
            out = self._io(f"r{k}_data")
            if self._registered_read:
                # Capture register composed here (see MemoryPrimitive): pre-edge readFirst.
                rd = Register(UInt(self._elem_w), init=0, name=f"{self._instance_name}__rrd{k}")
                rd <<= mux(self._io(f"r{k}_en"), rp.data, rd)
                out <<= rd
            else:
                out <<= rp.data
        for k in range(self._nrw):
            rw = m.rw_port(mask_chunks=self._mask_chunks)   # async read sub-port
            rw.addr    <<= self._io(f"rw{k}_addr")
            rw.data_in <<= self._io(f"rw{k}_din")
            rw.write   <<= self._io(f"rw{k}_write")
            rw.enable  <<= self._io(f"rw{k}_en")
            if self._mask_chunks:
                rw.mask <<= self._io(f"rw{k}_mask")
            dout = self._io(f"rw{k}_dout")
            if self._ruw == "writeFirst":
                collision = self._io(f"rw{k}_en") & self._io(f"rw{k}_write")
                dout <<= mux(collision, self._io(f"rw{k}_din"), rw.data_out)
            else:
                dout <<= rw.data_out
        self._store = m

    # ------------------------------------------------- synthesis verilog

    def custom_verilog(self) -> str:
        W = self._elem_w
        cw = (W // self._mask_chunks) if self._mask_chunks else 0
        writes: list = []
        reads: list = []

        for k in range(self._nw):
            spec = dict(addr=self._io(f"w{k}_addr").name, data=self._io(f"w{k}_data").name,
                        en=self._io(f"w{k}_en").name)
            if self._mask_chunks:
                spec.update(mask=self._io(f"w{k}_mask").name, mask_chunks=self._mask_chunks, chunk_w=cw)
            writes.append(spec)
        for k in range(self._nr):
            spec = dict(addr=self._io(f"r{k}_addr").name, out=self._io(f"r{k}_data").name,
                        registered=self._registered_read)
            if self._registered_read:
                spec["en"] = self._io(f"r{k}_en").name
            reads.append(spec)
        for k in range(self._nrw):
            en = self._io(f"rw{k}_en").name
            wr = self._io(f"rw{k}_write").name
            we = f"({en} && {wr})"
            wspec = dict(addr=self._io(f"rw{k}_addr").name, data=self._io(f"rw{k}_din").name, en=we)
            if self._mask_chunks:
                wspec.update(mask=self._io(f"rw{k}_mask").name, mask_chunks=self._mask_chunks, chunk_w=cw)
            writes.append(wspec)
            rspec = dict(addr=self._io(f"rw{k}_addr").name, out=self._io(f"rw{k}_dout").name, registered=False)
            if self._ruw == "writeFirst":
                rspec["ruw"] = "writeFirst"
                rspec["fwd"] = dict(en=we, addr=self._io(f"rw{k}_addr").name, data=self._io(f"rw{k}_din").name)
            reads.append(rspec)

        return ram_block(
            name=self._store.name, depth=self._depth, elem_w=W,  # live name survives uniquification
            writes=writes, reads=reads, reset=None, init=self._init,
            comment=f"--- RamPrimitive (uid={self._uid}, depth={self._depth}, width={W}, "
                    f"w={self._nw} r={self._nr} rw={self._nrw}) ---",
        )
