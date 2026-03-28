from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import SubtractorTestVectors, Encoding, is_signed
from sprouthdl.arithmetic.prefix_adders.adders import RippleCarryFinalAdder, StageBasedSubtractor
from sprouthdl.arithmetic.int_multipliers.stages.fsa_stages import BrentKungPrefixFinalStage
from sprouthdl.helpers import run_vectors_on_simulator
from sprouthdl.sprouthdl_simulator import Simulator


def int_subtractor_tb_sim():

    n_bits = 8
    full_output_bit = True
    enc = Encoding.twos_complement if full_output_bit else Encoding.twos_complement_overflow
    signed = is_signed(enc)

    for fsa_cls, fsa_name in [
        (RippleCarryFinalAdder, "RippleCarry"),
        (BrentKungPrefixFinalStage, "BrentKung"),
    ]:
        sub = StageBasedSubtractor(
            a_w=n_bits,
            b_w=n_bits,
            signed_a=signed,
            signed_b=signed,
            optim_type="area",
            fsa_cls=fsa_cls,
            full_output_bit=full_output_bit,
        )
        module = sub.to_module(f"Subtractor{n_bits}_{fsa_name}", with_clock=True, with_reset=True)

        vecs = SubtractorTestVectors(
            a_w=n_bits,
            b_w=n_bits,
            y_w=sub.io.y.typ.width,
            num_vectors=64,
            tb_sigma=None,
            a_encoding=enc,
            b_encoding=enc,
            y_encoding=enc,
        ).generate()

        sim = Simulator(module)
        run_vectors_on_simulator(sim, vecs, use_signed=False, print_on_pass=True, with_clk=False, test_name=f"Subtractor Test ({fsa_name})")


if __name__ == "__main__":
    int_subtractor_tb_sim()
