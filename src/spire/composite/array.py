from typing import Iterable, List, Sequence, Union, Tuple

from spire.composite.base import HDLComposite
from spire.expr import (
    Expr,
    Signal,
    Wire,
    as_expr,
)
from spire.hdl_traits import BitSerializable, BitSerializableLike


# Each element is either a plain Expr (Signal/Const/etc.)
# or any higher-level composite (Bundle, FixedPoint, nested Array, ...)
ArrayElem = BitSerializable          # an Array element: a leaf Expr or a nested composite
InputElem = BitSerializableLike      # what Array accepts: an element, or an int/bool literal coercible to one


class Array(HDLComposite):
    def __init__(
        self,
        values: Sequence[InputElem],
    ):
        """
        Construct an Array from:
          - ExprLike (int, bool, Expr) → Expr (via as_expr)
          - HDLComposite                 → nested composite (Bundle, Array, FixedPoint, ...)
        """
        if not values:
            raise ValueError("Array requires at least one element")

        elems: List[ArrayElem] = []
        for v in values:
            if isinstance(v, HDLComposite):
                # Any composite (Bundle, Array, FixedPoint, ...)
                elems.append(v)
            else:
                # ExprLike (int, bool, Expr) → Expr
                elems.append(as_expr(v))

        self._elems: List[ArrayElem] = elems

    # --------------------------------------------------------------
    # Basic iteration / len
    # --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._elems)

    def __iter__(self) -> Iterable[ArrayElem]:
        return iter(self._elems)

    # --------------------------------------------------------------
    # N-dimensional indexing
    # --------------------------------------------------------------
    def __getitem__(self, key):
        """
        Generic N-dimensional indexing via tuples:

          - 1D:
              arr[i]
              arr[i:j]

          - 2D:
              mat[i, j]
              mat[i, :]
              mat[:, j]
              mat[i0:i1, j0:j1]

          - ND:
              tensor[i, j, k]
              tensor[:, 1, 2:]
              tensor[1:3, :, 0]

        Works as long as the underlying structure is nested Arrays
        where appropriate.
        """
        if not isinstance(key, tuple):
            key = (key,)
        return self._get_nd(key)

    def _get_nd(self, keys: Tuple) -> Union[ArrayElem, "Array"]:
        # No more axes: return this sub-array as-is
        if not keys:
            return self

        idx, *rest = keys

        # -------- Case 1: integer index at this dimension --------
        if isinstance(idx, int):
            elem = self._elems[idx]
            if not rest:
                # Last axis: return scalar element / sub-array / composite
                return elem
            # More axes: we must recurse into a nested Array
            if not isinstance(elem, Array):
                raise TypeError(f"Too many indices: element at this axis is {type(elem)}, not Array")
            return elem._get_nd(tuple(rest))

        # -------- Case 2: slice at this dimension --------
        if isinstance(idx, slice):
            sliced_elems = self._elems[idx]

            if not rest:
                # Just slicing this dimension: return a 1D Array with those elements
                return Array(sliced_elems)

            # There are more axes: recurse into each element, collecting results
            sub_arrays: List[ArrayElem] = []
            for elem in sliced_elems:
                if not isinstance(elem, Array):
                    raise TypeError("Too many indices for Array: expected nested Array " f"but found {type(elem)}")
                sub_arrays.append(elem._get_nd(tuple(rest)))

            return Array(sub_arrays)

        raise TypeError(f"Unsupported index type at this axis: {type(idx)}")

    # --------------------------------------------------------------
    # __setitem__ (still 1D for now; can be extended similarly)
    # --------------------------------------------------------------
    def __setitem__(
        self,
        idx,
        value: InputElem,
    ) -> None:
        if isinstance(idx, tuple):
            # ND assignment could be added similarly to _get_nd; for now we keep 1D.
            raise NotImplementedError("N-dimensional __setitem__ not implemented")
        elif isinstance(value, HDLComposite):
            coerced = value
        else:
            coerced = as_expr(value)

        self._elems[idx] = coerced

    # --------------------------------------------------------------
    # Width + bit packing
    # --------------------------------------------------------------
    @property
    def width(self) -> int:
        """
        Total bit width of this array (recursively).

        NOTE: This overrides HDLComposite.width for Arrays but is equivalent
        to `self.to_bits().typ.width`.
        """
        total = 0
        for elem in self._elems:
            if isinstance(elem, Expr):
                total += elem.typ.width
            elif isinstance(elem, HDLComposite):
                total += elem.width
            else:
                raise TypeError(f"Unsupported element type in width(): {type(elem)}")
        return total

    @classmethod
    def wire_like(cls, template: "Array") -> "Array":
        """
        Create a 'wire-filled' Array with the same recursive shape as 'template'.

        Convention:
          - Expr leaves        → new Wire(typ)
          - HDLComposite leaves → type(elem).wire_like(elem)

        So any composite type you put inside Array (Bundle, FixedPoint, nested Array, ...)
        should implement `@classmethod wire_like(template_instance)`.
        """
        new_elems: List[ArrayElem] = []

        for elem in template._elems:
            if isinstance(elem, Expr):
                if isinstance(elem, Signal):
                    new_elems.append(Wire(elem.typ, name=elem.name))
                else:
                    new_elems.append(Wire(elem.typ))
            elif isinstance(elem, HDLComposite):
                new_elems.append(type(elem).wire_like(elem))
            else:
                raise TypeError(f"Unsupported element type in Array.wire_like: {type(elem)}")

        return cls(new_elems)

    def to_list_first_level(self) -> List[BitSerializable]:
        list_first_level: List[BitSerializable] = []
        for elem in self._elems:
            list_first_level.append(elem)
        return list_first_level

    def __repr__(self) -> str:
        return f"Array(len={len(self)}, width={self.width})"
