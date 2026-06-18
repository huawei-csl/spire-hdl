import abc
from collections import defaultdict
from typing import ClassVar, DefaultDict, Dict, List, Optional, Tuple, Type

import numpy as np

from spire.arithmetic.encoding.sign_magnitude import SignMagnitudeToTwosComplementDecoder
from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import OptimType, CompressorTreeAccumulator, FinalStageAdderBase, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, SelectionMode, SplitMode, StageBasedMultiplierBasic, StageBasedMultiplierIO
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, from_encoding, to_encoding
from spire.component import Component
from spire.io_record import Input, Output
from spire.expr import Bool, Concat, Const, Expr, Signal, SInt, UInt, mux, mux_if
from spire.component import Module


class StageBasedMultiplierBase(Component):

    def __init__(
        self,
        a_w: int,
        b_w: int,
        a_encoding: Encoding = Encoding.unsigned,
        b_encoding: Encoding = Encoding.unsigned,
        optim_type: OptimType = "area",
        ppg_cls: Optional[Type[PartialProductGeneratorBase]] = None,
        ppa_cls: Optional[Type[PartialProductAccumulatorBase]] = None,
        fsa_cls: Optional[Type[FinalStageAdderBase]] = None,
        selection_mode: Optional[SelectionMode] = None,
        split_mode: Optional[SplitMode] = None,
    ) -> None:

        self.a_encoding = a_encoding
        self.b_encoding = b_encoding
        self.aw = a_w
        self.bw = b_w
        self.optim_type = optim_type
        self.ppg_cls = ppg_cls
        self.ppa_cls = ppa_cls
        self.fsa_cls = fsa_cls
        self.selection_mode = selection_mode
        self.split_mode = split_mode

class StageBasedMultiplier(StageBasedMultiplierBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.a_encoding == Encoding.unsigned or self.b_encoding == Encoding.twos_complement, "Only unsigned or two's complement encoding is supported"
        assert self.b_encoding == Encoding.unsigned or self.b_encoding == Encoding.twos_complement, "Only unsigned or two's complement encoding is supported"
        y_encoding = Encoding.twos_complement if (self.a_encoding == Encoding.twos_complement or self.b_encoding == Encoding.twos_complement) else Encoding.unsigned

        def get_type(enc: Encoding,) -> Type:
            if enc == Encoding.unsigned:
                return UInt
            elif enc == Encoding.twos_complement or enc == Encoding.twos_complement_symmetric:
                return SInt

        self.io: StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Input(get_type(self.a_encoding)(self.aw)),
            b=Input(get_type(self.b_encoding)(self.bw)),
            y=Output(get_type(y_encoding)(self.aw + self.bw)),
        )

        self.elaborate()

    def elaborate(self) -> None:

        # instantiate an unsigned multiplier for the magnitudes
        mult: StageBasedMultiplierBasic = StageBasedMultiplierBasic(
            signed_a=from_encoding(self.a_encoding),
            signed_b=from_encoding(self.b_encoding),
            a_w=self.aw,
            b_w=self.bw,
            ppg_cls=self.ppg_cls,
            ppa_cls=self.ppa_cls,
            fsa_cls=self.fsa_cls,
            optim_type=self.optim_type,
            selection_mode=self.selection_mode,
            split_mode=self.split_mode,
        ).make_internal()

        self.mult = mult

        # use the specified multiplier
        mult.io.a <<= self.io.a
        mult.io.b <<= self.io.b
        self.io.y <<= mult.io.y


class StageBasedSignMagnitudeMultiplier(StageBasedMultiplierBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.a_encoding == Encoding.sign_magnitude and self.b_encoding == Encoding.sign_magnitude, "Only sign-magnitude encoding is supported"

        self.io : StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Input(UInt(self.aw)),
            b=Input(UInt(self.bw)),
            y=Output(UInt(self.aw + self.bw - 1)),
        )

        self.elaborate()

    def elaborate(self) -> None:

        # instantiate an unsigned multiplier for the magnitudes
        mult: StageBasedMultiplierBasic = StageBasedMultiplierBasic(
            signed_a=False,
            signed_b=False,
            a_w=self.aw - 1,
            b_w=self.bw - 1,
            ppg_cls=self.ppg_cls,
            ppa_cls=self.ppa_cls,
            fsa_cls=self.fsa_cls,
            optim_type=self.optim_type,
            selection_mode=self.selection_mode,
            split_mode=self.split_mode,
        ).make_internal()
        
        self.mult = mult

        W = self.aw  # assume square for now

        sa = self.io.a[W - 1]
        sb = self.io.b[W - 1]
        mag_a = self.io.a[0 : W - 1]  # make magnitude unsigned
        mag_b = self.io.b[0 : W - 1]  # make magnitude unsigned

        # mag_y = mag_a * mag_b
        # use the specified multiplier
        mult.io.a <<= mag_a
        mult.io.b <<= mag_b
        mag_y = mult.io.y

        # sign
        sy = sa ^ sb

        # make sure sign is positive if value is zero
        is_zero = mux(mag_y == 0, Const(True, Bool()), Const(False, Bool()))
        sy = mux(is_zero, Const(False, Bool()), sy)

        self.io.y <<= Concat([mag_y[0 : 2 * W - 2], sy])  # sign + magnitude (drop overflow bit)


class StageBasedSignMagnitudeExtMultiplier(StageBasedMultiplierBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.a_encoding == Encoding.sign_magnitude_ext and self.b_encoding == Encoding.sign_magnitude_ext, "Only sign-magnitude extended encoding is supported"

        self.io : StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Input(UInt(self.aw)),
            b=Input(UInt(self.bw)),
            y=Output(UInt(self.aw + self.bw)),
        )

        self.elaborate()

    def elaborate(self) -> None:

        # instantiate an unsigned multiplier for the magnitudes
        mult: StageBasedMultiplierBasic = StageBasedMultiplierBasic(
            signed_a=False,
            signed_b=False,
            a_w=self.aw - 1,
            b_w=self.bw - 1,
            ppg_cls=self.ppg_cls,
            ppa_cls=self.ppa_cls,
            fsa_cls=self.fsa_cls,
            optim_type=self.optim_type,
            selection_mode=self.selection_mode,
            split_mode=self.split_mode,
        ).make_internal()

        self.mult = mult

        W = self.aw  # assume square for now

        sa = self.io.a[W - 1]
        sb = self.io.b[W - 1]
        mag_a = self.io.a[0 : W - 1]  # make magnitude unsigned
        mag_b = self.io.b[0 : W - 1]  # make magnitude unsigned

        # assign sc_a = operand_a_sign & (operand_a_mag == {(N_IN-1){1'b0} }); // special case when input is -0 (-8), 1000
        sign_a = self.io.a[W - 1]
        sign_b = self.io.b[W - 1]
        sc_a = sign_a & (mag_a == Const(0, UInt(W - 1)))
        sc_b = sign_b & (mag_b == Const(0, UInt(W - 1)))

        #  assign operand_a_mag_mod = sc_a
        #        ? {1'b1, operand_a_mag[N_IN-3:0]}  // 100 -> 4 ,  operand_a_mag[N_IN-3:0] is 00 in this case
        #          : operand_a_mag;

        mag_a_mod = mux(sc_a, Concat([mag_a[0 : W - 2], Const(1, Bool())]), mag_a)
        mag_b_mod = mux(sc_b, Concat([mag_b[0 : W - 2], Const(1, Bool())]), mag_b)

        # mag_y = mag_a * mag_b
        # use the specified multiplier
        mult.io.a <<= mag_a_mod
        mult.io.b <<= mag_b_mod
        mag_y = mult.io.y

        sel_crit = Concat([sc_b, sc_a])

        # assign mag_y_mod, shift one up
        mag_y_mod = mux_if(
            if_cond=((sel_crit == Const(0b10, UInt(2))) | (sel_crit == Const(0b01, UInt(2)))),
            then_expr=Concat(
                [
                    Const(0, Bool()),
                    mag_y[0 : 2 * W - 2],
                ]
            ),
            else_expr=mux_if(
                if_cond=(sel_crit == Const(0b11, UInt(2))),
                then_expr=Const(0, UInt(2 * W - 2)),
                else_expr=mag_y,
            ),
        )

        h = mux_if(
            if_cond=(sel_crit==Const(0b11, UInt(2))),
            then_expr=Const(1, Bool()),
            else_expr=Const(0, Bool()),
        )

        # sign
        sy = sa ^ sb

        # make sure sign is positive if value is zero
        is_zero = mux(mag_y == 0, Const(True, Bool()), Const(False, Bool()))
        sy = mux(is_zero, Const(False, Bool()), sy)

        self.io.y <<= Concat([mag_y_mod[0 : 2 * W - 2], h, sy])  # sign + magnitude (drop overflow bit)

class StageBasedSignMagnitudeExtUpMultiplier(StageBasedMultiplierBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.a_encoding == Encoding.sign_magnitude_ext and self.b_encoding == Encoding.sign_magnitude_ext, "Only sign-magnitude extended encoding is supported"

        self.io : StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Input(UInt(self.aw)),
            b=Input(UInt(self.bw)),
            y=Output(UInt(self.aw + self.bw - 1)),
        )

        self.elaborate()

    def elaborate(self) -> None:

        # instantiate an unsigned multiplier for the magnitudes
        mult: StageBasedMultiplierBasic = StageBasedMultiplierBasic(
            signed_a=False,
            signed_b=False,
            a_w=self.aw - 1,
            b_w=self.bw - 1,
            ppg_cls=self.ppg_cls,
            ppa_cls=self.ppa_cls,
            fsa_cls=self.fsa_cls,
            optim_type=self.optim_type,
            selection_mode=self.selection_mode,
            split_mode=self.split_mode,
        ).make_internal()

        self.mult = mult

        W = self.aw  # assume square for now

        sa = self.io.a[W - 1]
        sb = self.io.b[W - 1]
        mag_a = self.io.a[0 : W - 1]  # make magnitude unsigned
        mag_b = self.io.b[0 : W - 1]  # make magnitude unsigned

        # assign sc_a = operand_a_sign & (operand_a_mag == {(N_IN-1){1'b0} }); // special case when input is -0 (-8), 1000
        sign_a = self.io.a[W - 1]
        sign_b = self.io.b[W - 1]
        sc_a = sign_a & (mag_a == Const(0, UInt(W - 1)))
        sc_b = sign_b & (mag_b == Const(0, UInt(W - 1)))

        #  assign operand_a_mag_mod = sc_a
        #        ? {1'b1, operand_a_mag[N_IN-3:0]}  // 100 -> 4 ,  operand_a_mag[N_IN-3:0] is 00 in this case
        #          : operand_a_mag;

        mag_a_mod = mux(sc_a, Concat([mag_a[0 : W - 2], Const(1, Bool())]), mag_a)
        mag_b_mod = mux(sc_b, Concat([mag_b[0 : W - 2], Const(1, Bool())]), mag_b)

        # mag_y = mag_a * mag_b
        # use the specified multiplier
        mult.io.a <<= mag_a_mod
        mult.io.b <<= mag_b_mod
        mag_y = mult.io.y

        sel_crit = Concat([sc_b, sc_a])

        # assign mag_y_mod, shift one up
        mag_y_mod = mux_if(
            if_cond=((sel_crit == Const(0b10, UInt(2))) | (sel_crit == Const(0b01, UInt(2)))),
            then_expr=Concat(
                [
                    Const(0, Bool()),
                    mag_y[0 : 2 * W - 2],
                ]
            ),
            else_expr=mux_if(
                if_cond=(sel_crit == Const(0b11, UInt(2))),
                then_expr=Const(0, UInt(2 * W - 2)),
                else_expr=mag_y,
            ),
        )

        # upper limit
        h = mux_if(
            if_cond=(sel_crit==Const(0b11, UInt(2))),
            then_expr=Const(1, Bool()),
            else_expr=Const(0, Bool()),
        )

        # sign
        sy = sa ^ sb

        # make sure sign is positive if value is zero
        is_zero = mux(mag_y == 0, Const(True, Bool()), Const(False, Bool()))
        sy = mux(is_zero, Const(False, Bool()), sy)
        
        # make sure sign is negative if value is the upper limit
        sy = mux(h, Const(True, Bool()), sy)

        self.io.y <<= Concat([mag_y_mod[0 : 2 * W - 2], sy])  # sign + magnitude (drop overflow bit)

class StageBasedSignMagnitudeToTwosComplementMultiplier(StageBasedMultiplierBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.a_encoding == Encoding.sign_magnitude and self.b_encoding == Encoding.sign_magnitude, "Only sign-magnitude encoding is supported"

        self.io : StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Input(UInt(self.aw)),
            b=Input(UInt(self.bw)),
            y=Output(UInt(self.aw + self.bw-1)),
        )

        self.elaborate()

    def elaborate(self) -> None:

        # instantiate an unsigned multiplier for the magnitudes
        mult = StageBasedSignMagnitudeMultiplier(
            a_encoding=self.a_encoding,
            b_encoding=self.b_encoding,
            a_w=self.aw,
            b_w=self.bw,
            ppg_cls=self.ppg_cls,
            ppa_cls=self.ppa_cls,
            fsa_cls=self.fsa_cls,
            optim_type=self.optim_type,
            selection_mode=self.selection_mode,
            split_mode=self.split_mode,
        ).make_internal()      
        mult.io.a <<= self.io.a
        mult.io.b <<= self.io.b

        self.mult = mult

        enc = SignMagnitudeToTwosComplementDecoder(width=self.aw + self.bw - 1).make_internal()
        enc.io.i <<= mult.io.y
        self.io.y <<= enc.io.o

        self.enc = enc

class StageBasedSignMagnitudeExtToTwosComplementMultiplier(StageBasedMultiplierBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.a_encoding == Encoding.sign_magnitude or self.a_encoding == Encoding.sign_magnitude_ext, "Only sign-magnitude encoding is supported"
        assert self.b_encoding == Encoding.sign_magnitude or self.b_encoding == Encoding.sign_magnitude_ext, "Only sign-magnitude encoding is supported"

        self.io : StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Input(UInt(self.aw)),
            b=Input(UInt(self.bw)),
            y=Output(UInt(self.aw + self.bw)),
        )

        self.elaborate()

    def elaborate(self) -> None:

        # instantiate an unsigned multiplier for the magnitudes
        mult = StageBasedSignMagnitudeExtMultiplier(
            a_encoding=Encoding.sign_magnitude_ext,
            b_encoding=Encoding.sign_magnitude_ext,
            a_w=self.aw,
            b_w=self.bw,
            ppg_cls=self.ppg_cls,
            ppa_cls=self.ppa_cls,
            fsa_cls=self.fsa_cls,
            optim_type=self.optim_type,
            selection_mode=self.selection_mode,
            split_mode=self.split_mode,
        ).make_internal()      
        mult.io.a <<= self.io.a
        mult.io.b <<= self.io.b

        self.mult = mult

        enc = SignMagnitudeToTwosComplementDecoder(width=self.aw + self.bw).make_internal()
        enc.io.i <<= mult.io.y
        self.io.y <<= enc.io.o

        self.enc = enc

class StageBasedSignMagnitudeExtToTwosComplementUpperMultiplier(StageBasedMultiplierBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.a_encoding == Encoding.sign_magnitude or self.a_encoding == Encoding.sign_magnitude_ext, "Only sign-magnitude encoding is supported"
        assert self.b_encoding == Encoding.sign_magnitude or self.b_encoding == Encoding.sign_magnitude_ext, "Only sign-magnitude encoding is supported"

        self.io : StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Input(UInt(self.aw)),
            b=Input(UInt(self.bw)),
            y=Output(UInt(self.aw + self.bw-1)),
        )

        self.elaborate()

    def elaborate(self) -> None:

        # instantiate an unsigned multiplier for the magnitudes
        mult = StageBasedSignMagnitudeExtUpMultiplier(
            a_encoding=Encoding.sign_magnitude_ext,
            b_encoding=Encoding.sign_magnitude_ext,
            a_w=self.aw,
            b_w=self.bw,
            ppg_cls=self.ppg_cls,
            ppa_cls=self.ppa_cls,
            fsa_cls=self.fsa_cls,
            optim_type=self.optim_type,
            selection_mode=self.selection_mode,
            split_mode=self.split_mode,
        ).make_internal()      
        mult.io.a <<= self.io.a
        mult.io.b <<= self.io.b

        self.mult = mult

        enc = SignMagnitudeToTwosComplementDecoder(width=self.aw + self.bw - 1).make_internal()
        enc.io.i <<= mult.io.y
        self.io.y <<= enc.io.o

        self.enc = enc

class StarMultiplier(StageBasedMultiplierBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.a_encoding == Encoding.unsigned or self.b_encoding == Encoding.twos_complement, "Only unsigned or two's complement encoding is supported"
        assert self.b_encoding == Encoding.unsigned or self.b_encoding == Encoding.twos_complement, "Only unsigned or two's complement encoding is supported"
        y_encoding = (
            Encoding.twos_complement if (self.a_encoding == Encoding.twos_complement or self.b_encoding == Encoding.twos_complement) else Encoding.unsigned
        )

        def get_type(
            enc: Encoding,
        ) -> Type:
            if enc == Encoding.unsigned:
                return UInt
            elif enc == Encoding.twos_complement:
                return SInt

        self.io: StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Input(get_type(self.a_encoding)(self.aw)),
            b=Input(get_type(self.b_encoding)(self.bw)),
            y=Output(get_type(y_encoding)(self.aw + self.bw)),
        )

        self.elaborate()

    def elaborate(self) -> None:
        self.io.y <<= self.io.a * self.io.b
