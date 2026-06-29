# -----------------------------
# High-level composites (Bundle, Array, FixedPoint, ...)
# -----------------------------
from __future__ import annotations
from abc import abstractmethod
from typing import List, Tuple, Type, TypeVar, Union

from spire.expr import Expr, ExprLike, Signal, as_expr, fit_width
from spire.hdl_traits import BitSerializable, Assignable


T_Comp = TypeVar("T_Comp", bound="HDLComposite")
SelfComp = TypeVar("SelfComp", bound="HDLComposite")


class HDLComposite(BitSerializable, Assignable):
    """
    Base class for structured HDL values (Bundle, Array, FixedPoint, ...).

    Inherits the read side (``to_bits`` / ``width``) from :class:`BitSerializable` and the
    write side (``<<=``) from :class:`Assignable`; subclasses supply only the *structure*
    via ``to_list_first_level`` and the *drive* via ``assign``.

    Requirements for subclasses:
      - to_list_first_level(self) -> List[Expr | HDLComposite]
      - wire_like(cls, *shape_args, **shape_kwargs) -> instance
    """

    @abstractmethod
    def to_list_first_level(self) -> List[Union[Expr, "HDLComposite"]]:
        """Return the ordered list of Expr leaves (Signals, Consts, etc.)."""
        ...

    def to_list(self) -> List[Expr]:
        flat_list: List[Expr] = []
        for elem in self.to_list_first_level():
            if not isinstance(elem, BitSerializable):
                raise TypeError(
                    f"Unsupported field type in {self.__class__.__name__}.to_list(): {elem} -> {type(elem)}"
                )
            flat_list.extend(elem.to_list())  # Expr -> [self]; composite -> recurse
        if not flat_list:
            raise ValueError(f"CompositeRecord {self.__class__.__name__} has no fields")
        return flat_list

    # width / to_bits are inherited from BitSerializable.

    def _named_children(self) -> List[Tuple[str, Union[Expr, "HDLComposite"]]]:
        """(local_key, child) pairs for one structural level. Default keys are positional
        indices (``Array`` and friends); records override this with field names."""
        return [(str(i), c) for i, c in enumerate(self.to_list_first_level())]

    def _assign_port_names(self, prefix: str) -> None:
        """Give every leaf a hierarchical port name (``prefix_<key>_...``) from its field/index
        path, so nested bundles emit unique, readable ports (``up_valid``, ``bus_addr``,
        ``data_0``) instead of colliding on bare leaf names."""
        for key, child in self._named_children():
            full = f"{prefix}_{key}"
            if isinstance(child, HDLComposite):
                child._assign_port_names(full)
            elif isinstance(child, Signal):
                child.name = full

    # -------- Assignment API shared by all composites --------

    def _assign_from_bits(self, bits: Expr) -> None:
        """
        Default packed assignment: slice the incoming bits across assignable leaves.
        """
        leaves = self.to_list()
        bit_pos = 0
        for leaf in leaves:
            width = leaf.typ.width
            slice_bits = bits[bit_pos : bit_pos + width]
            bit_pos += width

            if not isinstance(leaf, Assignable):
                raise TypeError(
                    f"Composite assignment expects assignable leaves, got {type(leaf)} in {self.__class__.__name__}"
                )
            leaf <<= slice_bits

        if bit_pos != bits.typ.width:
            raise ValueError(
                f"Bit-slice consumption mismatch in {self.__class__.__name__}: "
                f"used {bit_pos} of {bits.typ.width} bits"
            )

    def _coerce_rhs_to_bits(self, rhs: Union["HDLComposite", ExprLike]) -> Expr:
        """
        Convert rhs into a bitvector Expr with the same width as self.
        - HDLComposite → rhs.to_bits()
        - Expr/int/bool → as_expr + fit_width(...)
        """
        lhs_bits = self.to_bits()
        t = lhs_bits.typ

        if isinstance(rhs, HDLComposite):
            rhs_bits = rhs.to_bits()
        else:
            rhs_bits = fit_width(as_expr(rhs), t)

        if rhs_bits.typ.width != t.width:
            raise ValueError(f"Width mismatch in composite assignment: " f"lhs width={t.width}, rhs width={rhs_bits.typ.width}")
        return rhs_bits

    def assign(self, rhs: Union["HDLComposite", ExprLike]) -> None:
        """
        Structural assignment: drive this composite from rhs.

        Example:
            my_bundle.assign(other_bundle)
            my_array.assign(0)
        """
        bits = self._coerce_rhs_to_bits(rhs)
        self._assign_from_bits(bits)

    def __imatmul__(self: SelfComp, rhs: "HDLComposite") -> SelfComp:
        """
        Element-wise assignment across assignable leaves:
          lhs @= rhs
        """
        if not isinstance(rhs, HDLComposite):
            raise TypeError(f"{self.__class__.__name__} @= expects an HDLComposite, got {type(rhs)}")

        lhs_leaves = self.to_list()
        rhs_leaves = rhs.to_list()
        if len(lhs_leaves) != len(rhs_leaves):
            raise ValueError(
                f"{self.__class__.__name__} @= leaf count mismatch: "
                f"{len(lhs_leaves)} vs {len(rhs_leaves)}"
            )

        for lhs_leaf, rhs_leaf in zip(lhs_leaves, rhs_leaves):
            if not isinstance(lhs_leaf, Assignable):
                raise TypeError(
                    f"Composite element-wise assignment expects assignable leaves, got {type(lhs_leaf)} "
                    f"in {self.__class__.__name__}"
                )
            lhs_leaf <<= rhs_leaf

        return self

    # __ilshift__ (<<=) is inherited from Assignable.

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(width={self.width})"
