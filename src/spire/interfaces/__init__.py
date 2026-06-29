"""Reusable IO interfaces (bundles) for Component IO.

Each interface is an :class:`~spire.io_record.IORecord` subclass declared ONCE in its canonical
*source* / *master* polarity. From that single declaration you get:

* the sink/slave side via ``Flipped(Interface(...))`` (mirrors every leaf direction),
* peer wiring via ``connect(a, b)`` (the ``output`` side drives the ``input`` side per leaf, in
  either argument order), and
* feedthrough via ``iface.view_as_flipped()`` (re-export a port without changing its declared
  direction).

Because an interface *is* an ``IORecord``, behaviour can live on it as plain methods
(``Stream.fire()`` etc.), and nesting one inside a Component's ``IORecord`` yields hierarchical
port names (``out_valid``, ``out_ready``, ``out_data``).

    from spire.interfaces import Stream, Flow, MemPort

Definitions live one-per-module: :mod:`spire.interfaces.flow`, :mod:`spire.interfaces.stream`,
and :mod:`spire.interfaces.memory`.
"""
from spire.interfaces.flow import Flow
from spire.interfaces.stream import Stream
from spire.interfaces.memory import MemPort

__all__ = ["Flow", "Stream", "MemPort"]
