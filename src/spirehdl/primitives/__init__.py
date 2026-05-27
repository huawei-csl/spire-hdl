"""Spire HDL primitives — Component-based building blocks that emit their own
synthesisable Verilog via ``custom_verilog`` while providing a Python sim model
via ``elaborate``.

Compared to the special-cased built-in ``Memory`` (in ``spirehdl.spirehdl``), the
primitives here live entirely in user-space: no simulator / emitter changes are
needed to support them. They also support aggregate element types — the port
boundary is always a flat ``UInt(width)``, and the user packs / unpacks with
``HDLAggregate.to_bits()`` / ``from_bits()``.
"""

from spirehdl.primitives.primitive_memory import MemoryPrimitive
from spirehdl.primitives.primitive_fifo import FIFOPrimitive

__all__ = ["MemoryPrimitive", "FIFOPrimitive"]
