"""Memory-access interface: an addressed :class:`MemPort`.

Declared in canonical *initiator/master* polarity (the master drives the address). Use
``Flipped(MemPort(addr_w, data_w))`` for the memory (slave) side.
"""
from spire.io_record import IORecord, Input, Output
from spire.expr import Bool, UInt


class MemPort(IORecord):
    """Simple memory-access port. Canonical = **initiator/master**: drives ``addr`` / ``wdata`` /
    ``wen`` and receives ``rdata``.

    ``wen`` high = write ``wdata`` at ``addr``; ``wen`` low = read ``addr`` into ``rdata``. The
    memory (slave) side is ``Flipped(MemPort(addr_w, data_w))``.
    """

    def __init__(self, addr_w, data_w):
        super().__init__(
            addr=Output(UInt(addr_w)),
            wdata=Output(UInt(data_w)),
            wen=Output(Bool()),
            rdata=Input(UInt(data_w)),
        )

    def is_write(self):
        """``Expr`` that is high when this access is a write."""
        return self.wen
