# hdl_traits.py
# Two orthogonal capability traits shared by every HDL value (leaf Expr or
# structured Composite). Pure-abc; the only spire import is a *lazy* one inside
# to_bits() to avoid a cycle (expr.py and composite/base.py both sit ABOVE this).
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from spire.expr import Expr, ExprLike


class BitSerializable(ABC):
    """Read side: flattens to an ordered list of Expr leaves.

    Subclasses supply the single primitive ``to_list()``; ``width`` and ``to_bits()``
    are derived from it once, here, for both leaves and composites.
    """

    @abstractmethod
    def to_list(self) -> List["Expr"]:
        """Ordered Expr leaves (a single-element list for a leaf)."""
        ...

    @property
    def width(self) -> int:
        return sum(leaf.typ.width for leaf in self.to_list())

    def to_bits(self) -> "Expr":
        from spire.expr import Concat  # lazy: expr.py imports this module
        parts = self.to_list()
        if not parts:
            raise ValueError(f"{type(self).__name__}.to_list() returned no leaves")
        return parts[0] if len(parts) == 1 else Concat(parts)


class Assignable(ABC):
    """Write side: anything that can be the target of ``<<=`` / driven.

    A ``Signal`` is Assignable; a ``Const`` / ``Op2`` is *not* (it is a value, not a
    storage location). A composite is Assignable iff its leaves are. ``__ilshift__`` is
    written once here in terms of the ``assign()`` primitive.
    """

    @abstractmethod
    def assign(self, rhs: "ExprLike") -> None:
        ...

    def __ilshift__(self, rhs: "ExprLike"):
        self.assign(rhs)
        return self
