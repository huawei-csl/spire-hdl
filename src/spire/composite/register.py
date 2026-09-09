from typing import Generic, List, Optional, Type, Union

from spire.composite.base import HDLComposite, T_Comp
from spire.expr import Expr, ExprLike, Signal
from spire.hdl_traits import BitSerializable


class CompositeRegister(HDLComposite, Generic[T_Comp]):
    """A register that stores a packed composite value.

    Build it from a type and its constructor arguments, or from a value you already have::

        acc = CompositeRegister(FixedPoint, q8_8, name="acc")   # class + constructor args
        acc = CompositeRegister.like(a, name="acc")             # same shape as the value `a`

        acc <<= a + b       # next-state assignment (packed)
        y = acc.value       # structured view of the contents, here a FixedPoint

    One ``Signal(kind="reg")`` named ``name`` holds the bits. ``.value`` builds a fresh view every
    time it is read, so bind it once. The view's leaves are named after the register:
    ``<name>_<field path>`` when there are several (``acc_valid``, ``vec_0``), ``<name>_q`` when
    there is one.
    """

    def __init__(
        self,
        agg_cls: Type[T_Comp],
        *agg_args,
        name: Optional[str] = None,
        init: Optional[Union[T_Comp, ExprLike]] = None,
        **agg_kwargs,
    ):
        self._setup(agg_cls.wire_like(agg_cls(*agg_args, **agg_kwargs)), name, init)

    @classmethod
    def like(
        cls,
        template: T_Comp,
        name: Optional[str] = None,
        init: Optional[ExprLike] = None,
    ) -> "CompositeRegister[T_Comp]":
        """A register with the same shape as ``template``.

        Use this when you hold a value rather than a type. The shape comes from
        ``template.get_wire_clone()``, so records and arrays work without restating their
        constructor arguments. The template itself is left untouched.
        """
        reg = object.__new__(cls)
        reg._setup(template.get_wire_clone(), name, init)
        return reg

    def _setup(self, proto: T_Comp, name: Optional[str], init) -> None:
        self._proto = proto  # wire-backed shape; every `.value` read clones it
        self.name = name or f"reg_{type(proto).__name__}_{id(self)}"
        self._reg = Signal(typ=proto.to_bits().typ, kind="reg", name=self.name)
        # Composite leaves are wires, never constants, so init must be a packed constant.
        if init is not None:
            if isinstance(init, HDLComposite):
                raise ValueError("CompositeRegister init must be a constant packed value (int), not a composite instance")
            self._reg.set_init(init)

    # ---- HDLComposite API ----

    def to_list_first_level(self) -> List[BitSerializable]:
        """Expose the underlying register as the sole leaf."""
        return [self._reg]

    def get_wire_clone(self):
        """Refused: a register is storage, so a wire-backed copy of it is not a thing. Clone (or
        pipeline) its structured ``.value`` view instead. Also covers ``wire_like()``, which
        delegates here."""
        raise TypeError(
            "CompositeRegister.get_wire_clone() is not meaningful — a register is storage, not a "
            "shape. Use `.value` for a structured view of its contents and clone that."
        )

    # ---- Convenience views ----

    @property
    def value(self) -> T_Comp:
        """A structured view of the register contents (a new view on every read)."""
        view = self._proto.get_wire_clone()
        _name_view_leaves(view, self.name)
        view <<= self._reg
        return view

    @property
    def bits(self) -> Expr:
        """Raw register bits as Expr."""
        return self._reg


def _name_view_leaves(view: HDLComposite, prefix: str) -> None:
    """Name a view's leaves after its register: ``<prefix>_<field path>``, or ``<prefix>_q`` for a
    single leaf. Names the clone inherited from its template are dropped first, since they were
    chosen for the template, not for this view."""
    leaves = view.to_list()
    for leaf in leaves:
        if isinstance(leaf, Signal):
            leaf._given_name = None
    if len(leaves) == 1 and isinstance(leaves[0], Signal):
        leaves[0].name = f"{prefix}_q"
    else:
        view._assign_port_names(prefix)
