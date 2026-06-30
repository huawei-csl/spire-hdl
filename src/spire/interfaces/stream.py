"""Ready/valid handshake interface: :class:`Stream`.

Declared in canonical *source* polarity (the producer drives ``valid`` + ``data``; the sink drives
``ready``). Use ``Flipped(Stream(payload))`` for the sink side.
"""
from spire.io_record import IORecord, Input, Output
from spire.expr import Bool


class Stream(IORecord):
    """Ready/valid handshake. Canonical = **source**: drives ``valid`` + ``data``; sink drives ``ready``.

    A transfer happens on the cycle where ``valid & ready``. The sink side is
    ``Flipped(Stream(payload))``.
    """

    def __init__(self, payload):
        super().__init__(valid=Output(Bool()), ready=Input(Bool()), data=Output(payload))

    def fire(self):
        """``Expr`` that is high on a transfer (``valid & ready``)."""
        return self.valid & self.ready
