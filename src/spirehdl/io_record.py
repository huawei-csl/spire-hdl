"""Aggregate-backed IO container for :class:`~spirehdl.spirehdl_module.Component`.

``IORecord`` is the canonical way to declare a Component's IO. Field names become signal names
(no name triplication), and direction is given by the :class:`~spirehdl.spirehdl.Input` /
:class:`~spirehdl.spirehdl.Output` signal classes rather than a ``kind=`` string. Because
``IORecord`` *is* an ``HDLAggregate``, IO fields may themselves be nested aggregates (``Array``,
records, fixed/float types) and ``to_list()`` flattening, ``width``, ``<<=`` / ``@=`` all come for free.

Usage::

    self.io = IORecord(
        a=Input(UInt(8)),
        b=Input(UInt(8)),
        sum=Output(UInt(9)),
    )

The dynamic form (``IORecord(**fields)``) is primary because a Component's IO usually depends on
``__init__`` arguments (widths, depths, ...), which class-level templates cannot see.

``Input`` / ``Output`` are ``Signal`` subclasses defined in :mod:`spirehdl.spirehdl` (siblings of
``Wire`` / ``Register``); they are re-exported here for convenience. A field built without an
explicit name carries the ``_io_autoname`` flag, and ``IORecord`` fills its name from the field key.
"""
from __future__ import annotations

from spirehdl.aggregate.aggregate_record_dynamic import AggregateRecordDynamic
from spirehdl.spirehdl import Input, Output, Signal  # re-exported for `from spirehdl.io_record import ...`

__all__ = ["IORecord", "Input", "Output"]


class IORecord(AggregateRecordDynamic):
    """Aggregate IO container; field names become signal names, direction is explicit.

    Extends :class:`AggregateRecordDynamic` (not ``AggregateRecord``, which rejects non-``wire``
    kinds). Fields may be ``Signal`` ports (typically :class:`Input` / :class:`Output`) or nested
    ``HDLAggregate`` values (e.g. ``Array``).
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
