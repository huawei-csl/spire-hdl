# -----------------------------
# High-level composites (Bundle, Array, FixedPoint, ...)
# -----------------------------
from __future__ import annotations
import warnings
from abc import abstractmethod
from dataclasses import fields, is_dataclass
from typing import Any, List, Tuple, Type, TypeVar, Union

from spire.expr import Expr, Signal, Wire, as_expr, fit_width
from spire.hdl_traits import BitSerializable, Assignable, BitSerializableLike


T_Comp = TypeVar("T_Comp", bound="HDLComposite")
SelfComp = TypeVar("SelfComp", bound="HDLComposite")

_FLIP_KIND = {"input": "output", "output": "input"}  # wire / reg are direction-less


class HDLComposite(BitSerializable, Assignable):
    """
    Base class for structured HDL values (Bundle, Array, FixedPoint, ...).

    Inherits the read side (``to_bits`` / ``width``) from :class:`BitSerializable` and the
    write side (``<<=``) from :class:`Assignable`; subclasses supply only the *structure*
    via ``to_list_first_level`` and the *drive* via ``assign``.

    Subclasses supply ``to_list_first_level(self) -> List[BitSerializable]``; everything else,
    including shape cloning (``get_wire_clone``), has a working default here.
    """

    @abstractmethod
    def to_list_first_level(self) -> List[BitSerializable]:
        """Return this composite's ordered first-level children (leaf ``Expr`` or nested ``HDLComposite``)."""
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

    # -------- Shape cloning --------

    def get_wire_clone(self: SelfComp) -> SelfComp:
        """Return a new composite with the same shape as this one, but with fresh wires.

        Each leaf becomes a new ``Wire`` of the same type. Nested composites are cloned the same
        way. Everything else (widths, a ``FixedPointType``, an adder config) is shared with the
        original, since it describes the shape rather than being part of it.

        ``__init__`` is not called. The shape usually depends on constructor arguments that the
        object does not keep (``Packet(n=4)``), so the attributes are copied directly instead.

        Override this when a plain copy is wrong for your class: for example when the clone is
        built from a type (``FixedPoint``), when the shape is fixed by the class
        (``TemplateRecord``), when cloning makes no sense (``CompositeRegister``), or when the
        object carries state such as a view flag that a fresh copy must not inherit.
        """
        clone = object.__new__(type(self))
        for key, val in self._clone_attrs():
            setattr(clone, key, _clone_shape(val))
        return clone

    def _clone_attrs(self) -> List[Tuple[str, Any]]:
        """(attribute, value) pairs for ``get_wire_clone`` to carry over: dataclass fields first,
        in declaration order (leaf order follows it), then any other instance attribute."""
        declared: List[Tuple[str, Any]] = []
        if is_dataclass(self):
            declared = [(f.name, getattr(self, f.name)) for f in fields(self)]
        seen = {key for key, _ in declared}
        return declared + [(k, v) for k, v in vars(self).items() if k not in seen]

    @classmethod
    def wire_like(cls: Type[SelfComp], template: SelfComp) -> SelfComp:
        """Shim for :meth:`get_wire_clone` — prefer ``template.get_wire_clone()``.

        Kept as the older spelling, and because it reads better where the shape comes from a
        *type* instead of a value: ``FixedPoint.wire_like(q8_8)``."""
        return template.get_wire_clone()

    def _named_children(self) -> List[Tuple[str, BitSerializable]]:
        """(local_key, child) pairs for one structural level. Default keys are positional
        indices (``Array`` and friends); records override this with field names."""
        return [(str(i), c) for i, c in enumerate(self.to_list_first_level())]

    def _leaf_paths(self, prefix: str = "") -> List[str]:
        """Structural field/index path per leaf, in ``to_list()`` order (``bus.hdr.addr``)."""
        paths: List[str] = []
        for key, child in self._named_children():
            p = f"{prefix}.{key}" if prefix else key
            if isinstance(child, HDLComposite):
                paths.extend(child._leaf_paths(p))
            else:
                paths.append(p)
        return paths

    def _assign_port_names(self, prefix: str) -> None:
        """Name every leaf ``<prefix>_<field/index path>``, with an explicit user-chosen name as
        the last segment. Rebuilt from scratch each call, so outer re-walks never duplicate segments."""
        for key, child in self._named_children():
            full = f"{prefix}_{key}"
            if isinstance(child, HDLComposite):
                child._assign_port_names(full)
            elif isinstance(child, Signal):
                child.name = f"{prefix}_{child._given_name}" if child._given_name else full

    def flip(self: SelfComp) -> SelfComp:
        """Reverse the direction of every leaf in place (``input``<->``output``; ``wire``/``reg``
        untouched), recursing through nested composites. An involution: ``x.flip().flip()`` == ``x``.
        Derives a sink interface from a source one; see also :func:`spire.Flipped`."""
        for child in self.to_list_first_level():
            if isinstance(child, HDLComposite):
                child.flip()
            elif isinstance(child, Signal):
                child.kind = _FLIP_KIND.get(child.kind, child.kind)
        return self

    def _directed_leaves(self) -> "List[Tuple[Signal, str]]":
        """(leaf, direction) pairs — the view connect() consumes. A plain bundle reports each
        leaf's own kind; a flipped view reports them reversed (see view_as_flipped)."""
        return [(leaf, leaf.kind) for leaf in self.to_list()]

    def view_as_flipped(self) -> "_FlippedView":
        """A non-mutating, direction-reversed *view* for expressing 'inside' orientation at
        connect time, without touching the real port directions. Reuses the bundle's actual
        leaves, so connecting the view drives the real signals.

        Distinct from :func:`spire.Flipped`, which *permanently* sets a sink polarity at
        declaration. Feedthrough is then just ``connect(my_io.view_as_flipped(), other)`` —
        the flip binds to the operand you call it on, so there's no ambiguity."""
        return _FlippedView(self)

    def connect(self, other: "Connectable") -> None:
        """Bulk-connect two interfaces: per matching leaf the ``output`` side drives the ``input``
        side (so a backward ``ready`` wires itself). Either side may be a bundle or a
        ``view_as_flipped()`` view; leaves pair positionally and must match in width and resolve
        to opposite directions.

        Peer:        ``connect(consumer.io, producer.io)``
        Feedthrough: ``connect(my_io.view_as_flipped(), child_io)`` (re-export) or
                     ``connect(up.view_as_flipped(), down.view_as_flipped())`` (own passthrough).
        """
        _connect_directed(self, other)

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

    def _coerce_rhs_to_bits(self, rhs: "BitSerializableLike") -> Expr:
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

    def assign(self, rhs: "BitSerializableLike") -> None:
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


def _clone_shape(val: Any) -> Any:
    """Clone one attribute value for :meth:`HDLComposite.get_wire_clone`.

    Leaves become fresh wires and composites are cloned recursively. Lists, tuples and dicts are
    rebuilt element by element, so children stored in a list (``Array._elems``) are cloned too.
    Anything else describes the shape and is returned as is.
    """
    if isinstance(val, HDLComposite):
        return val.get_wire_clone()
    if isinstance(val, Expr):
        # Keep the leaf's name if it has one. A view leaf such as `FixedPoint(bits=a + b)` has
        # none, so it gets the usual placeholder name (made unique per netlist at emission).
        return Wire(val.typ, name=getattr(val, "name", None) or "sig")
    if isinstance(val, list):
        return [_clone_shape(v) for v in val]
    if isinstance(val, tuple):
        return tuple(_clone_shape(v) for v in val)
    if isinstance(val, dict):
        return {k: _clone_shape(v) for k, v in val.items()}
    return val


# A "Connectable" is anything that can report (leaf, direction) pairs: an HDLComposite or a
# _FlippedView of one. connect() programs against this, so orientation is expressed by *which*
# operand you wrap with view_as_flipped(), never by a positional boolean flag.
Connectable = Union[HDLComposite, "_FlippedView"]


def _connect_directed(a: Connectable, b: Connectable) -> None:
    al, bl = a._directed_leaves(), b._directed_leaves()
    if len(al) != len(bl):
        raise ValueError(f"connect: leaf-count mismatch ({len(al)} vs {len(bl)})")
    pa, pb = a._leaf_paths(), b._leaf_paths()
    if pa != pb:
        first = next((x, y) for x, y in zip(pa, pb) if x != y)
        warnings.warn(f"connect: structural field paths differ ({first[0]!r} vs {first[1]!r}); "
                      f"leaves pair positionally — check the wiring", RuntimeWarning, stacklevel=3)
    for (la, ka), (lb, kb) in zip(al, bl):
        if la.typ.width != lb.typ.width:
            raise ValueError(
                f"connect: width mismatch on '{la.name}'/'{lb.name}' ({la.typ.width} vs {lb.typ.width})"
            )
        if {ka, kb} != {"input", "output"}:
            raise TypeError(
                f"connect: leaves '{la.name}'/'{lb.name}' resolve to the same direction "
                f"('{ka}'/'{kb}'). For a feedthrough, wrap your own boundary with .view_as_flipped()."
            )
        src, dst = (la, lb) if ka == "output" else (lb, la)
        if dst._driver is not None:
            warnings.warn(f"connect: '{dst.name}' is already driven — connecting a second producer "
                          f"rewires it and leaves the first dangling", RuntimeWarning, stacklevel=3)
        dst <<= src


class _FlippedView:
    """A non-mutating, direction-reversed view of a bundle (or of another view).

    Reuses the underlying bundle's real leaves — so connecting the view drives the actual
    signals — but reports every leaf's direction flipped. Created via ``HDLComposite.view_as_flipped()``;
    used to express that an operand is the enclosing module's own boundary (reversed from inside)
    without ever mutating the real port kinds.
    """
    __slots__ = ("_inner",)

    def __init__(self, inner: Connectable) -> None:
        self._inner = inner

    def _directed_leaves(self) -> "List[Tuple[Signal, str]]":
        return [(leaf, _FLIP_KIND.get(k, k)) for leaf, k in self._inner._directed_leaves()]

    def _leaf_paths(self, prefix: str = "") -> "List[str]":
        return self._inner._leaf_paths(prefix)

    def view_as_flipped(self) -> "_FlippedView":
        return _FlippedView(self)

    def connect(self, other: Connectable) -> None:
        _connect_directed(self, other)
