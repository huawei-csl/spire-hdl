"""Valid-only dataflow interface: :class:`Flow` (no backpressure).

Declared in canonical *source* polarity (the producer drives the data). Use ``Flipped(Flow(payload))``
for the sink side.
"""
from spire.io_record import IORecord, Output
from spire.expr import Bool


class Flow(IORecord):
    """Valid-only stream — no backpressure. Canonical = **source**: drives ``valid`` + ``data``.

    Use when the consumer can always accept (a pipeline that never stalls). The sink side is
    ``Flipped(Flow(payload))``.
    """

    def __init__(self, payload):
        super().__init__(valid=Output(Bool()), data=Output(payload))

    def fire(self):
        """``Expr`` that is high on a transfer (just ``valid`` — there is no ``ready``)."""
        return self.valid
