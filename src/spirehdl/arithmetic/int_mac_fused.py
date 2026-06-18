from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import OptimType, SelectionMode, SplitMode

from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    PPAOption,
    PPGOption,
)
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import (
    Encoding,
    EncodingModel,
    is_signed,
)
from spirehdl.cores.matmul_accumulate.matmul_accumulate_core_fused import (
    MultiplierConfig,
    fused_inner_product,
)
from spirehdl.spirehdl import SInt, Signal, UInt
from spirehdl.spirehdl_module import Component
from spirehdl.io_record import IORecord

MAC_SUPPORTED_INPUT_ENCODINGS = {Encoding.unsigned, Encoding.twos_complement}


def resolve_mac_c_bits(n_bits: int, c_bits: int | None) -> int:
    if n_bits <= 0:
        raise ValueError("n_bits must be > 0")
    resolved = 2 * n_bits if c_bits is None else c_bits
    if resolved <= 0:
        raise ValueError("c_bits must be > 0")
    return resolved


def resolve_mac_output_encoding(input_encoding: Encoding, output_encoding: Encoding | None) -> Encoding:
    if input_encoding not in MAC_SUPPORTED_INPUT_ENCODINGS:
        raise ValueError(
            f"MAC input encoding {input_encoding.name} is not supported. "
            "Use Encoding.unsigned or Encoding.twos_complement."
        )

    if output_encoding is None:
        return input_encoding

    if input_encoding == Encoding.unsigned:
        allowed_outputs = {Encoding.unsigned, Encoding.unsigned_overflow}
    else:
        allowed_outputs = {Encoding.twos_complement, Encoding.twos_complement_overflow}

    if output_encoding not in allowed_outputs:
        allowed = ", ".join(enc.name for enc in sorted(allowed_outputs, key=lambda item: item.name))
        raise ValueError(
            f"MAC output encoding {output_encoding.name} is not supported for input encoding {input_encoding.name}. "
            f"Supported outputs: {allowed}"
        )
    return output_encoding


@dataclass(frozen=True)
class MacBuildConfig:
    n_bits: int
    c_bits: int
    ppg_opt: PPGOption = PPGOption.AND
    ppa_opt: PPAOption = PPAOption.ACCUMULATOR_TREE
    fsa_opt: FSAOption = FSAOption.RIPPLE_CARRY
    encoding: Encoding = Encoding.unsigned
    optim_type: OptimType = "area"
    use_operator: bool = False
    selection_mode: Optional[SelectionMode] = None
    split_mode: Optional[SplitMode] = None


class MacIO(IORecord):
    """IO for FusedMacComponent: a, b, c inputs and y output."""


class FusedMacComponent(Component):
    io: MacIO

    def __init__(self, cfg: MacBuildConfig):
        self.cfg = cfg
        if self.cfg.encoding not in MAC_SUPPORTED_INPUT_ENCODINGS:
            raise ValueError(
                f"MAC input encoding {self.cfg.encoding.name} is not supported. "
                "Use Encoding.unsigned or Encoding.twos_complement."
            )

        io_type = SInt if is_signed(self.cfg.encoding) else UInt
        self.a = Signal(typ=io_type(self.cfg.n_bits), kind="input")
        self.b = Signal(typ=io_type(self.cfg.n_bits), kind="input")
        self.c = Signal(typ=io_type(self.cfg.c_bits), kind="input")

        self.elaborate()
        self.io = MacIO(a=self.a, b=self.b, c=self.c, y=self.y)

    def elaborate(self) -> None:
        if self.cfg.use_operator:
            y_expr = self.a * self.b + self.c
        else:
            mult_cfg = MultiplierConfig(
                ppg_opt=self.cfg.ppg_opt,
                ppa_opt=self.cfg.ppa_opt,
                fsa_opt=self.cfg.fsa_opt,
                optim_type=self.cfg.optim_type,
                selection_mode=self.cfg.selection_mode,
                split_mode=self.cfg.split_mode,
            )
            y_expr = fused_inner_product([self.a], [self.b], self.c, mult_cfg, self.cfg.encoding)

        io_type = SInt if is_signed(self.cfg.encoding) else UInt
        self.y = Signal(typ=io_type(y_expr.typ.width), kind="output")
        self.y <<= y_expr


@dataclass
class MacTestVectors:
    a_w: int
    b_w: int
    c_w: int
    y_w: int
    num_vectors: int = 64
    tb_sigma: float | None = None
    input_encoding: Encoding = Encoding.unsigned
    output_encoding: Encoding = Encoding.unsigned

    def generate(self) -> list[tuple[str, dict[str, int], dict[str, int]]]:
        if self.num_vectors <= 0:
            raise ValueError("num_vectors must be > 0")

        in_enc = EncodingModel(self.input_encoding)
        out_enc = EncodingModel(self.output_encoding)
        vectors: list[tuple[str, dict[str, int], dict[str, int]]] = []

        for _ in range(self.num_vectors):
            if self.tb_sigma is None:
                a_value = in_enc.get_uniform_sample(self.a_w)
                b_value = in_enc.get_uniform_sample(self.b_w)
                c_value = in_enc.get_uniform_sample(self.c_w)
            else:
                a_value = in_enc.get_normal_sample(self.a_w, self.tb_sigma)
                b_value = in_enc.get_normal_sample(self.b_w, self.tb_sigma)
                c_value = in_enc.get_normal_sample(self.c_w, self.tb_sigma)

            y_value = (a_value * b_value) + c_value
            vectors.append(
                (
                    f"{a_value}*{b_value}+{c_value}",
                    {
                        "a": in_enc.encode_value(a_value, self.a_w),
                        "b": in_enc.encode_value(b_value, self.b_w),
                        "c": in_enc.encode_value(c_value, self.c_w),
                    },
                    {
                        "y": out_enc.encode_value(y_value, self.y_w),
                    },
                )
            )

        return vectors
