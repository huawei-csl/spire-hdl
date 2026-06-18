from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Sequence, Type, TypeVar, Generic

from spire.composite.base import HDLComposite
from spire.expr import Const, Expr, ExprLike, HDLType, Resize, Wire, as_expr, fit_width

# -----------------------------
# Fixed-point type description
# -----------------------------


@dataclass(frozen=True)
class FixedPointType:
    """
    Fixed-point format description.

    width_total : total number of bits
    width_frac  : number of fractional bits (LSB side)
    signed      : two's complement if True, pure unsigned if False
    """

    width_total: int
    width_frac: int
    signed: bool = True

    def __post_init__(self):
        if self.width_total < 1:
            raise ValueError("FixedPointType.width_total must be >= 1")
        if not (0 <= self.width_frac <= self.width_total):
            raise ValueError("FixedPointType.width_frac must be in [0, width_total]")

    @property
    def width_int(self) -> int:
        """Number of integer bits (including sign if signed)."""
        return self.width_total - self.width_frac

    def to_hdl_type(self) -> HDLType:
        """Underlying HDLType for the raw bitvector."""
        return HDLType(self.width_total, signed=self.signed, is_bool=False)

class ARITHQuant(Enum):
    """
    Quantization behaviour for fixed-point ops.

    - WrpTrc : wrap on overflow, truncate dropped LSBs
    - WrpRnd : wrap on overflow, round when dropping LSBs
    - ClpTrc / ClpRnd / SatTrc / SatRnd are reserved for
      clip/saturate semantics (not fully implemented yet).
    """
    WrpTrc = "WrpTrc"
    WrpRnd = "WrpRnd"
    ClpTrc = "ClpTrc"
    ClpRnd = "ClpRnd"
    SatTrc = "SatTrc"
    SatRnd = "SatRnd"


class FixedPoint(HDLComposite):
    """
    Fixed-point composite backed by a single Expr/Signal, with an explicit
    FixedPointType describing width/frac/sign.

    The raw bits are always in two's complement or unsigned form; the
    FixedPointType controls how they are interpreted.

    Typical usage:

        q8_8 = FixedPointType(width_total=16, width_frac=8, signed=True)

        a = FixedPoint(q8_8, name="a")             # own wire
        b = FixedPoint(q8_8, name="b")

        s = a + b                                  # full-precision add
        s16 = a.add(b, out_type=q8_8)              # quantized back to q8_8

    """

    def __init__(
        self,
        ftype: FixedPointType,
        name: Optional[str] = None,
        bits: Optional[ExprLike] = None,
    ):
        self.ftype = ftype
        self._typ = ftype.to_hdl_type()

        if bits is None:
            # Owning case: create a fresh wire
            sig = Wire(self._typ, name=name)
            self._bits: Expr = sig
        else:
            # View case: reinterpret existing bits as this format
            bits_e = fit_width(as_expr(bits), self._typ)
            self._bits = bits_e

    # ---- Introspection ----

    @property
    def bits(self) -> Expr:
        """Underlying HDL Expr for the raw fixed-point bits."""
        return self._bits

    @property
    def typ(self) -> HDLType:
        """Underlying HDLType of the raw bits."""
        return self._typ

    @property
    def width(self) -> int:
        """Total bit-width (HDLComposite-style)."""
        return self.ftype.width_total

    @property
    def width_frac(self) -> int:
        return self.ftype.width_frac

    @property
    def width_int(self) -> int:
        return self.ftype.width_int

    @property
    def signed(self) -> bool:
        return self.ftype.signed

    # ---- HDLComposite API ----

    def to_list_first_level(self) -> List[Expr | HDLComposite]:
        return [self._bits]

    @classmethod
    def wire_like(
        cls,
        arg: Union["FixedPoint", FixedPointType],
        name: Optional[str] = None,
    ) -> "FixedPoint":
        """
        Create a wire-backed FixedPoint with the same type as:
          - a template FixedPoint instance, or
          - an explicit FixedPointType.
        """
        if isinstance(arg, FixedPoint):
            ftype = arg.ftype
        elif isinstance(arg, FixedPointType):
            ftype = arg
        else:
            raise TypeError(f"FixedPoint.wire_like expects FixedPoint or FixedPointType, got {type(arg)}")
        return cls(ftype, name=name, bits=None)

    # ---------------------------------
    # Internal helpers for arithmetic
    # ---------------------------------

    def _check_compatible(self, other: "FixedPoint") -> None:
        if not isinstance(other, FixedPoint):
            raise TypeError(f"Expected FixedPoint, got {type(other)}")
        if self.ftype.signed != other.ftype.signed:
            raise ValueError("FixedPoint sign mismatch: " f"{self.ftype.signed} vs {other.ftype.signed}")

    @staticmethod
    def _quantize_bits(
        raw: Expr,
        full_type: FixedPointType,
        out_type: FixedPointType,
        q: ARITHQuant,
    ) -> Expr:
        """
        Quantize 'raw' from full_type to out_type.

        Currently implements:
          - WrpTrc : wrap on overflow, truncate dropped LSBs
          - WrpRnd : wrap on overflow, round when dropping LSBs

        Other modes (Clp*, Sat*) raise NotImplementedError for now.
        """
        if full_type.signed != out_type.signed:
            raise ValueError("Quantization sign mismatch between full_type and out_type")

        if q not in (ARITHQuant.WrpTrc, ARITHQuant.WrpRnd):
            raise NotImplementedError(f"ARITHQuant '{q.name}' not implemented yet; " "use WrpTrc or WrpRnd for now.")

        expr = raw

        # Adjust fractional bits
        frac_diff = full_type.width_frac - out_type.width_frac

        if frac_diff > 0:
            # We have more fractional bits in full_type than in out_type → drop LSBs
            if q == ARITHQuant.WrpRnd and frac_diff > 0:
                # Add 0.5 LSB before truncation for rounding
                rnd_val = 1 << (frac_diff - 1)
                rnd_const = Const(
                    rnd_val,
                    HDLType(raw.typ.width, signed=False, is_bool=False),
                )
                expr = expr + fit_width(rnd_const, expr.typ)

            # Logical right shift by frac_diff; width stays the same
            expr = expr >> frac_diff

        elif frac_diff < 0:
            # Need more fractional bits in out_type → shift left
            expr = expr << (-frac_diff)

        # Finally, resize to target total width (wrap/truncate MSBs)
        if expr.typ.width != out_type.width_total:
            expr = Resize(expr, out_type.width_total)

        return expr

    def _binary_raw(
        self,
        other: "FixedPoint",
        op: str,
    ) -> tuple[Expr, FixedPointType]:
        """
        Compute the raw Expr and full-precision FixedPointType for
        add/sub/mul before quantization.
        """
        self._check_compatible(other)

        if op in ("add", "sub"):
            # Align fractional bits
            full_frac = max(self.width_frac, other.width_frac)
            shift_a = full_frac - self.width_frac
            shift_b = full_frac - other.width_frac

            a = self.bits
            b = other.bits
            if shift_a > 0:
                a = a << shift_a
            if shift_b > 0:
                b = b << shift_b

            # Equalize widths before add/sub
            wa, wb = a.typ.width, b.typ.width
            w_common = max(wa, wb)
            if wa != w_common:
                a = Resize(a, w_common)
            if wb != w_common:
                b = Resize(b, w_common)

            raw = a + b if op == "add" else a - b
            full_type = FixedPointType(
                width_total=raw.typ.width,
                width_frac=full_frac,
                signed=self.signed,
            )
            return raw, full_type

        elif op == "mul":
            raw = self.bits * other.bits
            full_frac = self.width_frac + other.width_frac
            full_type = FixedPointType(
                width_total=raw.typ.width,
                width_frac=full_frac,
                signed=self.signed,
            )
            return raw, full_type

        else:
            raise ValueError(f"Unknown FixedPoint op '{op}'")

    def _binary_op(
        self,
        other: "FixedPoint",
        op: str,
        *,
        out_type: Optional[FixedPointType] = None,
        q: ARITHQuant = ARITHQuant.WrpTrc,
    ) -> "FixedPoint":
        raw, full_type = self._binary_raw(other, op)

        if out_type is None:
            # Full precision: use the natural full_type (no real quantization)
            out_type = full_type
            # Special-case: avoid unnecessary shifting when widths match
            bits = raw
            if bits.typ.width != out_type.width_total:
                bits = Resize(bits, out_type.width_total)
            return FixedPoint(out_type, bits=bits)

        # Out type given explicitly
        if out_type.signed != self.signed:
            raise ValueError("Output FixedPointType.signed must match operand sign " f"({out_type.signed} vs {self.signed})")

        bits_q = self._quantize_bits(raw, full_type, out_type, q)
        return FixedPoint(out_type, bits=bits_q)

    # ---------------------------------
    # Public arithmetic API
    # ---------------------------------

    def add(
        self,
        other: "FixedPoint",
        *,
        out_type: Optional[FixedPointType] = None,
        q: ARITHQuant = ARITHQuant.WrpTrc,
    ) -> "FixedPoint":
        return self._binary_op(other, "add", out_type=out_type, q=q)

    def sub(
        self,
        other: "FixedPoint",
        *,
        out_type: Optional[FixedPointType] = None,
        q: ARITHQuant = ARITHQuant.WrpTrc,
    ) -> "FixedPoint":
        return self._binary_op(other, "sub", out_type=out_type, q=q)

    def mul(
        self,
        other: "FixedPoint",
        *,
        out_type: Optional[FixedPointType] = None,
        q: ARITHQuant = ARITHQuant.WrpTrc,
    ) -> "FixedPoint":
        return self._binary_op(other, "mul", out_type=out_type, q=q)

    # Operator overloads (default: full precision, WrpTrc)

    def __add__(self, other: "FixedPoint") -> "FixedPoint":
        return self.add(other)

    def __sub__(self, other: "FixedPoint") -> "FixedPoint":
        return self.sub(other)

    def __mul__(self, other: "FixedPoint") -> "FixedPoint":
        return self.mul(other)

    def __repr__(self) -> str:
        t = self.ftype
        return f"FixedPoint(total={t.width_total}, frac={t.width_frac}, " f"signed={t.signed})"
