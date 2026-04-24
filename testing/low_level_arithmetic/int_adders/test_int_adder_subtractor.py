import pytest

from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import (
    AdderTestVectors,
    Encoding,
    SubtractorTestVectors,
    is_signed,
)
from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import RippleCarryFinalAdder
from spirehdl.arithmetic.int_multipliers.stages.fsa_stages import (
    BrentKungPrefixFinalStage,
    KoggeStonePrefixFinalStage,
    SklanskyPrefixFinalStage,
)
from spirehdl.arithmetic.prefix_adders.adders import StageBasedPrefixAdder, StageBasedSubtractor
from spirehdl.helpers import run_vectors_on_simulator
from spirehdl.spirehdl_simulator import Simulator


FSA_OPTIONS = [
    RippleCarryFinalAdder,
    BrentKungPrefixFinalStage,
    KoggeStonePrefixFinalStage,
    SklanskyPrefixFinalStage,
]

ENCODINGS = [
    Encoding.unsigned,
    Encoding.twos_complement,
]


@pytest.mark.parametrize("fsa_cls", FSA_OPTIONS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("enc", ENCODINGS, ids=lambda e: e.name)
def test_adder(fsa_cls, enc):
    n_bits = 8
    full_output_bit = True
    signed = is_signed(enc)

    adder = StageBasedPrefixAdder(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
        fsa_cls=fsa_cls,
        full_output_bit=full_output_bit,
    )
    module = adder.to_module(f"Adder_{fsa_cls.__name__}_{enc.name}", with_clock=True, with_reset=True)

    out_enc = Encoding.twos_complement if signed else Encoding.unsigned
    vecs = AdderTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        y_w=adder.io.y.typ.width,
        num_vectors=64,
        tb_sigma=None,
        a_encoding=enc,
        b_encoding=enc,
        y_encoding=out_enc,
    ).generate()

    sim = Simulator(module)
    run_vectors_on_simulator(sim, vecs, use_signed=False, with_clk=False)


@pytest.mark.parametrize("fsa_cls", FSA_OPTIONS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("enc", ENCODINGS, ids=lambda e: e.name)
def test_subtractor(fsa_cls, enc):
    n_bits = 8
    full_output_bit = True
    signed = is_signed(enc)

    sub = StageBasedSubtractor(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
        fsa_cls=fsa_cls,
        full_output_bit=full_output_bit,
    )
    module = sub.to_module(f"Sub_{fsa_cls.__name__}_{enc.name}", with_clock=True, with_reset=True)

    # subtraction result is always signed (a - b can be negative)
    vecs = SubtractorTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        y_w=sub.io.y.typ.width,
        num_vectors=64,
        tb_sigma=None,
        a_encoding=enc,
        b_encoding=enc,
        y_encoding=Encoding.twos_complement,
    ).generate()

    sim = Simulator(module)
    run_vectors_on_simulator(sim, vecs, use_signed=False, with_clk=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
