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

from spire.composite.record_dynamic import CompositeRecordDynamic
from spire.expr import Input, Output, Signal  # re-exported for `from spire.io_record import ...`

__all__ = ["IORecord", "Input", "Output"]


class IORecord(CompositeRecordDynamic):
    """Composite IO container; field names become signal names, direction is explicit.

    Extends :class:`CompositeRecordDynamic` (not ``CompositeRecord``, which rejects non-``wire``
    kinds). Fields may be ``Signal`` ports (typically :class:`Input` / :class:`Output`) or nested
    ``HDLComposite`` values (e.g. ``Array``).
    """

    def __init__(self, **fields: object) -> None:
        for field_name, val in fields.items():
            # A port built without an explicit name (`_io_autoname`) inherits the field key.
            # Inside an `IORecord(a=Input(...))` call the signal can't self-name reliably, so the
            # field key — actual data, not inspected source — is the robust source of truth.
            if isinstance(val, Signal) and getattr(val, "_io_autoname", False):
                val.name = field_name
                val._io_autoname = False
            setattr(self, field_name, val)
