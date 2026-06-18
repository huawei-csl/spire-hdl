from dataclasses import fields, is_dataclass, make_dataclass
from typing import Any, Dict, Iterable, List, Type, TypeVar, Union

from spirehdl.aggregate.hdl_aggregate import HDLAggregate
from spirehdl.spirehdl import Expr, Signal, Wire

T_Record = TypeVar("T_Record", bound="AggregateRecordDynamic")


class AggregateRecordDynamic(HDLAggregate):

    def _raw_fields(self) -> list:
        """Get field values directly without flattening (avoids recursion into to_list)."""
        if is_dataclass(self):
            return [getattr(self, f.name) for f in fields(self)]
        return list(vars(self).values())

    def to_list_first_level(self) -> List[Expr | HDLAggregate]:
        return [v for v in self._raw_fields()
                if isinstance(v, (Expr, HDLAggregate))]


# IO normalization — lives here (the aggregate layer) so the dependency points one way:
# ir.py / spirehdl_module.py import these *downward* instead of importing the aggregate inside a
# function body. `_to_aggregate` references AggregateRecordDynamic directly (same module, no import).
def _to_aggregate(obj: Any) -> "AggregateRecordDynamic":
    """Convert any IO container into an AggregateRecordDynamic for uniform handling.

    Accepts an existing aggregate (returned as-is), a ``@dataclass``, a namedtuple, a plain dict, or a
    plain object with ``__dict__``. Single IO-normalization point behind ``Component.get_ios()``.
    """
    if isinstance(obj, AggregateRecordDynamic):  # already an aggregate — no conversion needed
        return obj
    if is_dataclass(obj):  # @dataclass IO (most common Component IO pattern)
        pairs = [(f.name, getattr(obj, f.name)) for f in fields(obj)]
    elif hasattr(obj, "_fields"):  # namedtuple IO
        pairs = [(n, getattr(obj, n)) for n in obj._fields]
    elif isinstance(obj, dict):  # plain dict IO
        pairs = list(obj.items())
    else:  # plain object with __dict__
        pairs = list(vars(obj).items())
    DynIO = make_dataclass("DynIO", [(n, type(v)) for n, v in pairs], bases=(AggregateRecordDynamic,))
    return DynIO(**{n: v for n, v in pairs})


def iter_values(obj: Any) -> Iterable[Any]:
    """Extract leaf Signals from an IO container (dataclass, namedtuple, dict, or HDLAggregate).

    Converts to AggregateRecordDynamic first, then flattens via ``to_list()`` — all aggregate fields
    (Array, Record, ...) and plain lists are recursively resolved into individual Signals.
    """
    return _to_aggregate(obj).to_list()