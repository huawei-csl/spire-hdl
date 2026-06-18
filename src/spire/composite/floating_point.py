from dataclasses import dataclass
from typing import List, Optional, Union

from spire.composite.base import HDLComposite
from spire.arithmetic.floating_point.spire_hdl_float_mult import FpMul
from spire.arithmetic.floating_point.spire_hdl_float_mult_sn import FpMulSN
from spire.arithmetic.floating_point.spire_hdl_float_add import FpAdd
from spire.expr import Expr, ExprLike, HDLType, UInt, Wire, as_expr, fit_width
from spire.arithmetic.int_arithmetic_config import AdderConfig, MultiplierConfig


@dataclass(frozen=True)
class FloatingPointType:
    """Floating-point format description.

    exponent_width : number of exponent bits
    fraction_width : number of fraction bits (excluding hidden bit)
    """

    exponent_width: int
    fraction_width: int
    subnormal_support: bool = False
    always_subnormal_rounding: bool = False

    def __post_init__(self) -> None:
        if self.exponent_width < 1:
            raise ValueError("FloatingPointType.exponent_width must be >= 1")
        if self.fraction_width < 0:
            raise ValueError("FloatingPointType.fraction_width must be >= 0")

    @property
    def width_total(self) -> int:
        return 1 + self.exponent_width + self.fraction_width

    def to_hdl_type(self) -> HDLType:
        return UInt(self.width_total)


class FloatingPoint(HDLComposite):
    """Flat floating-point composite backed by a single Expr/Signal."""

    def __init__(
        self,
        ftype: FloatingPointType,
        name: Optional[str] = None,
        bits: Optional[ExprLike] = None,
        adder_cfg: Optional[AdderConfig] = None,
        mult_cfg: Optional[MultiplierConfig] = None,
    ):
        self.ftype = ftype
        self._typ = ftype.to_hdl_type()
        self.adder_cfg = adder_cfg
        self.mult_cfg = mult_cfg

        if bits is None:
            sig = Wire(self._typ, name=name)
            self._bits: Expr = sig
        else:
            bits_e = fit_width(as_expr(bits), self._typ)
            self._bits = bits_e

    # ---- Introspection ----
    @property
    def bits(self) -> Expr:
        return self._bits

    @property
    def typ(self) -> HDLType:
        return self._typ

    @property
    def width(self) -> int:
        return self.ftype.width_total

    @property
    def exponent(self) -> Expr:
        lo = self.ftype.fraction_width
        hi = lo + self.ftype.exponent_width
        return self._bits[lo:hi]

    @property
    def fraction(self) -> Expr:
        return self._bits[0 : self.ftype.fraction_width]

    @property
    def sign(self) -> Expr:
        return self._bits[self.width - 1]

    # ---- HDLComposite API ----
    def to_list_first_level(self) -> List[Expr | HDLComposite]:
        return [self._bits]

    @classmethod
    def wire_like(
        cls,
        arg: Union["FloatingPoint", FloatingPointType],
        name: Optional[str] = None,
    ) -> "FloatingPoint":
        if isinstance(arg, FloatingPoint):
            ftype = arg.ftype
        elif isinstance(arg, FloatingPointType):
            ftype = arg
        else:
            raise TypeError("FloatingPoint.wire_like expects FloatingPoint or FloatingPointType, " f"got {type(arg)}")
        return cls(ftype, name=name, bits=None)

    # ---------------------------------
    # Floating-point add helper
    # ---------------------------------
    def add(self, other: "FloatingPoint") -> "FloatingPoint":
        if not isinstance(other, FloatingPoint):
            raise TypeError(f"Expected FloatingPoint, got {type(other)}")
        if self.ftype != other.ftype:
            raise ValueError("FloatingPoint add requires matching types")

        core = FpAdd(EW=self.ftype.exponent_width, FW=self.ftype.fraction_width, adder_cfg=self.adder_cfg, subnormals=self.ftype.subnormal_support).make_internal()

        core.io.a <<= self.bits
        core.io.b <<= other.bits

        return FloatingPoint(self.ftype, bits=core.io.y, adder_cfg=self.adder_cfg, mult_cfg=self.mult_cfg)

    def __add__(self, other: "FloatingPoint") -> "FloatingPoint":
        return self.add(other)

    # ---------------------------------
    # Floating-point multiply helper
    # ---------------------------------
    def mul(self, other: "FloatingPoint") -> "FloatingPoint":
        if not isinstance(other, FloatingPoint):
            raise TypeError(f"Expected FloatingPoint, got {type(other)}")
        if self.ftype != other.ftype:
            raise ValueError("FloatingPoint multiply requires matching types")

        EW = self.ftype.exponent_width
        FW = self.ftype.fraction_width
        if self.ftype.subnormal_support or self.ftype.always_subnormal_rounding:
            core = FpMulSN(
                EW=EW, FW=FW, mult_cfg=self.mult_cfg,
                subnormals=self.ftype.subnormal_support,
                always_subnormal_rounding=self.ftype.always_subnormal_rounding,
            ).make_internal()
        else:
            core = FpMul(EW=EW, FW=FW, mult_cfg=self.mult_cfg).make_internal()

        core.io.a <<= self.bits
        core.io.b <<= other.bits

        return FloatingPoint(self.ftype, bits=core.io.y, adder_cfg=self.adder_cfg, mult_cfg=self.mult_cfg)

    def __mul__(self, other: "FloatingPoint") -> "FloatingPoint":
        return self.mul(other)

    def __repr__(self) -> str:
        t = self.ftype
        return f"FloatingPoint(exp={t.exponent_width}, frac={t.fraction_width})"