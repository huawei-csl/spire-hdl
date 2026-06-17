from typing import Generic, List, Optional
from typing import Type, TypeVar, Union
from spirehdl.aggregate.hdl_aggregate import HDLAggregate, T_Agg
from spirehdl.spirehdl import Expr, ExprLike, Signal, as_expr, fit_width


class AggregateRegister(HDLAggregate, Generic[T_Agg]):
    """
    Generic register that stores a packed HDLAggregate value.

    Usage:
        # Register that holds a FixedPoint(16, frac=8)
        acc = AggregateRegister(FixedPoint, total_width=16, frac_width=8, signed=True, name="acc_reg")

        # Read as a FixedPoint:
        acc_val = acc.value

        # Assign packed:
        acc <<= acc_val  # or acc <<= some_other_fixed

    Internally:
      - A single Signal(kind='reg') bitvector is allocated.
      - value/Q provide a structured HDLAggregate view via agg_cls.from_bits(...).
      - <<= drives the register's next-state (reg driver).
    """

    def __init__(
        self,
        agg_cls: Type[T_Agg],
        *agg_args,
        name: Optional[str] = None,
        init: Optional[Union[T_Agg, ExprLike]] = None,
        **agg_kwargs,
    ):
        # Store metadata so we can rebuild a structured view later
        self._agg_cls: Type[T_Agg] = agg_cls
        self._agg_args = agg_args
        self._agg_kwargs = dict(agg_kwargs)

        # Use a wire-like instance to infer shape/width
        proto: T_Agg = agg_cls.wire_like(agg_cls(*agg_args, **agg_kwargs))
        bits_typ = proto.to_bits().typ

        reg_name = name or f"reg_{agg_cls.__name__}_{id(self)}"
        self._reg = Signal(typ=bits_typ, kind="reg", name=reg_name)

        # Optional init value (packed)
        if init is not None:
            if isinstance(init, HDLAggregate):
                init_bits = init.to_bits()
            else:
                init_bits = as_expr(init)
            self._reg._init = fit_width(init_bits, bits_typ)

    # ---- HDLAggregate API ----

    def to_list_first_level(self) -> List[Expr | HDLAggregate]:
        """Expose the underlying register as the sole leaf."""
        return [self._reg]

    @classmethod
    def wire_like(cls, *args, **kwargs):
        raise TypeError("AggregateRegister.wire_like() is not meaningful")

    # ---- Convenience views ----

    @property
    def value(self) -> T_Agg:
        """Structured view of the register contents."""
        value = self._agg_cls(*self._agg_args, **self._agg_kwargs)
        value <<= self._reg
        return value

    @property
    def bits(self) -> Expr:
        """Raw register bits as Expr."""
        return self._reg
