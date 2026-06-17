from dataclasses import dataclass

from spirehdl.spirehdl import Bool, Concat, Const, Signal, UInt, mux_if
from spirehdl.spirehdl_module import Component


@dataclass
class EncodingIO:
    i: Signal
    o: Signal

class EncoderDecoderBase(Component):
    io: EncodingIO


class TwosComplementToSignMagnitudeEncoder(EncoderDecoderBase):
    def __init__(self, width: int, *, clip_most_negative: bool = False) -> None:
        self.width = width
        assert self.width >= 2, "Width must be >= 2 for sign-magnitude conversions"
        self.clip_most_negative = clip_most_negative

        self.io: EncodingIO = EncodingIO(
            i=Signal(typ=UInt(width), kind="input"),
            o=Signal(typ=UInt(width), kind="output"),
        )

        self.elaborate()

    def elaborate(self) -> None:
        sign = self.io.i[self.width - 1]
        abs_val = mux_if(
            if_cond=sign,
            then_expr=(~self.io.i + Const(1, UInt(1)))[0 : self.width],
            else_expr=self.io.i,
        )
        magnitude = abs_val[0 : self.width - 1]
        unsigned_result = Concat([magnitude, sign])

        if self.clip_most_negative:
            mag_is_zero = self.io.i[0 : self.width - 1] == Const(0, UInt(self.width - 1))
            most_negative = sign & mag_is_zero
            clipped = Const((1 << self.width) - 1, UInt(self.width))
            self.io.o <<= mux_if(if_cond=most_negative, then_expr=clipped, else_expr=unsigned_result)
        else:
            self.io.o <<= unsigned_result


class SignMagnitudeToTwosComplementDecoder(EncoderDecoderBase):
    def __init__(self, width: int, *, clip_most_negative: bool = False) -> None:
        self.width = width
        assert self.width >= 2, "Width must be >= 2 for sign-magnitude conversions"
        self.clip_most_negative = clip_most_negative

        self.io: EncodingIO = EncodingIO(
            i=Signal(typ=UInt(width), kind="input"),
            o=Signal(typ=UInt(width), kind="output"),
        )

        self.elaborate()

    def elaborate(self) -> None:
        sign = self.io.i[self.width - 1]
        mag_raw = self.io.i[0 : self.width - 1]

        if self.clip_most_negative:
            mag = mux_if(
                if_cond=sign & (mag_raw == Const(0, UInt(self.width - 1))),
                then_expr=Const((1 << (self.width - 1)) - 1, UInt(self.width - 1)),
                else_expr=mag_raw,
            )
        else:
            mag = mag_raw

        positive = Concat([mag, Const(0, Bool())])
        negative = Concat([(~mag + Const(1, UInt(1)))[0 : self.width - 1], Const(1, Bool())])
        self.io.o <<= mux_if(if_cond=sign, then_expr=negative, else_expr=positive)
