from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Type, TypeVar, Union

from sprouthdl.aggregate.hdl_aggregate import HDLAggregate
from sprouthdl.sprouthdl import Expr, Signal, Wire

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