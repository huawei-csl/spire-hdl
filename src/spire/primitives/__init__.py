"""Spire HDL primitives — Component-based building blocks that emit their own
synthesisable Verilog via ``custom_verilog`` while providing a Python sim model
via ``elaborate``.

Compared to the core's sim-only storage (``spire.memory._MemoryArray``), the
primitives here live entirely in user-space: no simulator / emitter changes are
needed to support them. They also support composite element types — the port
boundary is always a flat ``UInt(width)``, and the user packs / unpacks with
``HDLComposite.to_bits()`` / ``from_bits()``.
"""

from spire.primitives.primitive_memory import MemoryPrimitive
from spire.primitives.primitive_fifo import FIFOPrimitive
from spire.primitives.primitive_ram import RamPrimitive
from spire.primitives.primitive_rom import RomPrimitive
from spire.primitives.primitive_memory_via_reg import MemoryPrimitive_via_reg
from spire.primitives.primitive_fifo_via_reg import FIFOPrimitive_via_reg

__all__ = [
    "MemoryPrimitive",
    "FIFOPrimitive",
    "RamPrimitive",
    "RomPrimitive",
    "MemoryPrimitive_via_reg",
    "FIFOPrimitive_via_reg",
]
