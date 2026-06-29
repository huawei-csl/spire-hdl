"""Composite-backed IO container for :class:`~spire.component.Component`.

``IORecord`` is the canonical way to declare a Component's IO. Field names become signal names
(no name triplication), and direction is given by the :class:`~spire.expr.Input` /
:class:`~spire.expr.Output` signal classes rather than a ``kind=`` string. Because
``IORecord`` *is* an ``HDLComposite``, IO fields may themselves be nested composites (``Array``,
records, fixed/float types) and ``to_list()`` flattening, ``width``, ``<<=`` / ``@=`` all come for free.

Usage::

    self.io = IORecord(
        a=Input(UInt(8)),
        b=Input(UInt(8)),
        sum=Output(UInt(9)),
    )

The dynamic form (``IORecord(**fields)``) is primary because a Component's IO usually depends on
``__init__`` arguments (widths, depths, ...), which class-level templates cannot see.

``Input`` / ``Output`` are ``Signal`` subclasses defined in :mod:`spire.expr` (siblings of
``Wire`` / ``Register``); they are re-exported here for convenience. A field built without an
explicit name carries the ``_io_autoname`` flag, and ``IORecord`` fills its name from the field key.
"""
from __future__ import annotations

from spire.composite.base import Connectable, HDLComposite
from spire.composite.record import CompositeRecord
from spire.expr import Input, Output, Signal  # re-exported for `from spire.io_record import ...`

__all__ = ["IORecord", "Input", "Output", "Flipped", "connect"]


def Flipped(bundle: HDLComposite) -> HDLComposite:
    """Return ``bundle`` with every leaf direction reversed (in place; safe on a freshly
    constructed instance, which is the usual case). ``Flipped(Stream(UInt(8)))`` is the sink
    view of a source ``Stream`` -- same type, mirrored directions."""
    return bundle.flip()


def connect(a: Connectable, b: Connectable) -> None:
    """Bulk-connect two interfaces (``output`` drives ``input`` per leaf). Symmetric.

    Either side may be a bundle or a ``.view_as_flipped()`` view. Peer: ``connect(consumer, producer)``.
    Feedthrough: wrap your own boundary, e.g. ``connect(my_io.view_as_flipped(), child_io)``. See
    :meth:`HDLComposite.connect`."""
    a.connect(b)


class IORecord(CompositeRecord):
    """A Component's IO bundle — a :class:`CompositeRecord` used at the module boundary.

    Field names become signal names and direction is explicit via :class:`Input` /
    :class:`Output`. Fields may be ``Signal`` ports or nested ``HDLComposite`` values
    (e.g. ``Array``). Construction, field-key autonaming, and ``to_list`` flattening are
    inherited from ``CompositeRecord``; ``IORecord`` exists to mark intent ("this is IO").
    """
