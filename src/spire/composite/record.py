from __future__ import annotations

from dataclasses import fields, is_dataclass, make_dataclass
from typing import Any, Iterable, List, Tuple, Union

from spire.composite.base import HDLComposite
from spire.expr import Expr, Signal
from spire.hdl_traits import BitSerializable


class CompositeRecord(HDLComposite):
    """Record composite: field names become signal names, direction is explicit.

    Fields may be ``Signal`` ports (typically :class:`~spire.expr.Input` /
    :class:`~spire.expr.Output`) or nested ``HDLComposite`` values (``Array``,
    fixed/float types, other records). Anything in the instance that is not a
    ``BitSerializable`` (plain ints, bookkeeping, ...) is ignored, not rejected.

    Construct one of three ways, all instance-based:
      * inline / dynamic — ``CompositeRecord(a=Input(...), b=Output(...))``;
      * a named/parameterized subclass with an ``__init__`` that calls
        ``super().__init__(...)`` (e.g. a reusable ``Stream`` / ``Bus`` interface);
      * a ``@dataclass`` subclass, for declarative fixed fields.

    ``to_list()`` flattening, ``width``, ``to_bits``, and ``<<=`` / ``@=`` are all
    inherited from :class:`HDLComposite`.
    """

    def __init__(self, **field_values: object) -> None:
        for field_name, val in field_values.items():
            # Direct Signal port: a leaf built without an explicit name (`_io_autoname`)
            # inherits the field key as its port name.
            if isinstance(val, Signal):
                if getattr(val, "_io_autoname", False):
                    val.name = field_name
                    val._io_autoname = False
            # Nested bundle/array: name its leaves hierarchically by the field path
            # (``up`` + ``valid`` -> ``up_valid``) so nested-interface ports don't collide.
            elif isinstance(val, HDLComposite):
                val._assign_port_names(field_name)
            setattr(self, field_name, val)

    def _raw_fields(self) -> list:
        """Field values without flattening (avoids recursion into to_list)."""
        if is_dataclass(self):
            return [getattr(self, f.name) for f in fields(self)]
        return list(vars(self).values())

    def to_list_first_level(self) -> List[BitSerializable]:
        return [v for v in self._raw_fields() if isinstance(v, BitSerializable)]

    def _named_children(self) -> List[Tuple[str, BitSerializable]]:
        """Field (name, value) pairs — used to build hierarchical port names."""
        if is_dataclass(self):
            items = [(f.name, getattr(self, f.name)) for f in fields(self)]
        else:
            items = list(vars(self).items())
        return [(k, v) for k, v in items if isinstance(v, BitSerializable)]


# IO normalization — lives here (the composite layer) so the dependency points one way:
# ir.py / component.py import these downward instead of importing the composite inside a
# function body. ``_to_composite`` references CompositeRecord directly (same module, no import).
def _to_composite(obj: Any) -> "CompositeRecord":
    """Convert any IO container into a CompositeRecord for uniform handling.

    Accepts an existing composite (returned as-is), a ``@dataclass``, a namedtuple, a plain dict,
    or a plain object with ``__dict__``. Single IO-normalization point behind ``Component.get_ios()``.
    """
    if isinstance(obj, CompositeRecord):  # already a composite — no conversion needed
        return obj
    if is_dataclass(obj):  # @dataclass IO
        pairs = [(f.name, getattr(obj, f.name)) for f in fields(obj)]
    elif hasattr(obj, "_fields"):  # namedtuple IO
        pairs = [(n, getattr(obj, n)) for n in obj._fields]
    elif isinstance(obj, dict):  # plain dict IO
        pairs = list(obj.items())
    else:  # plain object with __dict__
        pairs = list(vars(obj).items())
    DynIO = make_dataclass("DynIO", [(n, type(v)) for n, v in pairs], bases=(CompositeRecord,))
    return DynIO(**{n: v for n, v in pairs})


def iter_values(obj: Any) -> Iterable[Any]:
    """Flatten an IO container (dataclass, namedtuple, dict, or HDLComposite) into leaf Signals."""
    return _to_composite(obj).to_list()
