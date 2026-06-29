from dataclasses import fields, is_dataclass, make_dataclass
from typing import Any, Dict, Iterable, List, Type, TypeVar, Union

from spire.composite.base import HDLComposite
from spire.expr import Expr, Signal, Wire
from spire.hdl_traits import BitSerializable

T_Record = TypeVar("T_Record", bound="CompositeRecordDynamic")


class CompositeRecordDynamic(HDLComposite):

    def _raw_fields(self) -> list:
        """Get field values directly without flattening (avoids recursion into to_list)."""
        if is_dataclass(self):
            return [getattr(self, f.name) for f in fields(self)]
        return list(vars(self).values())

    def to_list_first_level(self) -> List[Expr | HDLComposite]:
        return [v for v in self._raw_fields()
                if isinstance(v, BitSerializable)]


# IO normalization — lives here (the composite layer) so the dependency points one way:
# ir.py / spire_module.py import these *downward* instead of importing the composite inside a
# function body. `_to_composite` references CompositeRecordDynamic directly (same module, no import).
def _to_composite(obj: Any) -> "CompositeRecordDynamic":
    """Convert any IO container into an CompositeRecordDynamic for uniform handling.

    Accepts an existing composite (returned as-is), a ``@dataclass``, a namedtuple, a plain dict, or a
    plain object with ``__dict__``. Single IO-normalization point behind ``Component.get_ios()``.
    """
    if isinstance(obj, CompositeRecordDynamic):  # already an composite — no conversion needed
        return obj
    if is_dataclass(obj):  # @dataclass IO (most common Component IO pattern)
        pairs = [(f.name, getattr(obj, f.name)) for f in fields(obj)]
    elif hasattr(obj, "_fields"):  # namedtuple IO
        pairs = [(n, getattr(obj, n)) for n in obj._fields]
    elif isinstance(obj, dict):  # plain dict IO
        pairs = list(obj.items())
    else:  # plain object with __dict__
        pairs = list(vars(obj).items())
    DynIO = make_dataclass("DynIO", [(n, type(v)) for n, v in pairs], bases=(CompositeRecordDynamic,))
    return DynIO(**{n: v for n, v in pairs})


def iter_values(obj: Any) -> Iterable[Any]:
    """Extract leaf Signals from an IO container (dataclass, namedtuple, dict, or HDLComposite).

    Converts to CompositeRecordDynamic first, then flattens via ``to_list()`` — all composite fields
    (Array, Record, ...) and plain lists are recursively resolved into individual Signals.
    """
    return _to_composite(obj).to_list()