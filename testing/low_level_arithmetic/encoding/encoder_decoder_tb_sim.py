from spire.arithmetic.encoding.sign_magnitude import (
    SignMagnitudeToTwosComplementDecoder,
    TwosComplementToSignMagnitudeEncoder,
)
from spire.arithmetic.int_multipliers.eval.testvector_generation import (
    Encoding,
    EncoderDecoderTestVectors,
)
from spire.helpers import run_vectors, run_vectors_on_simulator
from spire.component import Netlist
from spire.simulator import Simulator
from spire.verilog_testbench import TestbenchGenSimulator
from spire.various.vcd_writer import write_vcd


def _run_component(module: Netlist, vecs, base_filename: str, *, test_name: str, with_clk: bool = False) -> None:

    run_vectors(module, vecs, print_on_pass=True)

    sim = Simulator(module)
    sim.trace_enabled = True
    run_vectors_on_simulator(
        sim,
        vecs,
        print_on_pass=True,
        with_clk=with_clk,
        test_name=test_name,
    )

    trace_by_names = sim.get_trace_by_names()
    write_vcd(
        trace_by_names=trace_by_names,
        filename=f"{base_filename}.vcd",
        top_module=module.name,
        timescale="1ns",
    )

    sim_tb = TestbenchGenSimulator(module)
    run_vectors_on_simulator(
        sim_tb, vecs, print_on_pass=False, with_clk=with_clk, test_name=test_name
    )

    data_filename = f"{base_filename}_vectors.dat"

    sim_tb.to_testbench_file(f"{base_filename}.v", tb_module_name=module.name + "_tb")
    sim_tb.to_data_driver_testbench_file_incl_dat(
        f"{base_filename}_data.v",
        vectors=vecs,
        data_file=data_filename,
        with_clk=with_clk,
    )
    module.to_verilog_file(f"{base_filename}_module.v")


def sign_magnitude_encoder_decoder_tb_sim():
    width = 4
    num_vectors = 16

    for clip_most_negative in (False, True):
        suffix = "clip" if clip_most_negative else "noclip"

        encoder_input_encoding = Encoding.twos_complement if clip_most_negative else Encoding.twos_complement_symmetric
        encoder_vecs = EncoderDecoderTestVectors(
            width=width,
            num_vectors=num_vectors,
            input_encoding=encoder_input_encoding,
            output_encoding=Encoding.sign_magnitude,
        ).generate()

        encoder = TwosComplementToSignMagnitudeEncoder(width=width, clip_most_negative=clip_most_negative)
        encoder_module = encoder.to_module(
            f"TwosComplementToSignMagnitudeEncoder{width}_{suffix}", with_clock=False, with_reset=False
        )
        _run_component(
            encoder_module,
            encoder_vecs,
            base_filename=f"sign_mag_encoder_tb_sim_{suffix}",
            test_name=f"Encoder ({'clip' if clip_most_negative else 'no clip'})",
            with_clk=False,
        )

        decoder_input_encoding = Encoding.sign_magnitude_ext if clip_most_negative else Encoding.sign_magnitude
        decoder_output_encoding = Encoding.twos_complement_symmetric
        decoder_vecs = EncoderDecoderTestVectors(
            width=width,
            num_vectors=num_vectors,
            input_encoding=decoder_input_encoding,
            output_encoding=decoder_output_encoding,
        ).generate()

        decoder = SignMagnitudeToTwosComplementDecoder(width=width, clip_most_negative=clip_most_negative)
        decoder_module = decoder.to_module(
            f"SignMagnitudeToTwosComplementDecoder{width}_{suffix}", with_clock=False, with_reset=False
        )
        _run_component(
            decoder_module,
            decoder_vecs,
            base_filename=f"sign_mag_decoder_tb_sim_{suffix}",
            test_name=f"Decoder ({'clip' if clip_most_negative else 'no clip'})",
            with_clk=False,
        )


if __name__ == "__main__":
    sign_magnitude_encoder_decoder_tb_sim()
