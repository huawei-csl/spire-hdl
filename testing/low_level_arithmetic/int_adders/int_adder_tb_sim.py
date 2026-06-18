from spire.arithmetic.int_multipliers.eval.testvector_generation import AdderTestVectors, Encoding, is_signed
from spire.arithmetic.prefix_adders.adders import RippleCarryFinalAdder, StageBasedPrefixAdder
from spire.helpers import run_vectors, run_vectors_on_simulator
from spire.simulator import Simulator
from spire.verilog_testbench import TestbenchGenSimulator, write_vector_data_file
from spire.various.vcd_writer import write_vcd


def int_adders_tb_sim():

    n_bits = 8
    full_output_bit = True
    #enc = Encoding.unsigned if full_output_bit else Encoding.unsigned_overflow
    enc = Encoding.twos_complement if full_output_bit else Encoding.twos_complement_overflow
    signed = is_signed(enc)
    
    
    adder = StageBasedPrefixAdder(
        a_w=n_bits,
        b_w=n_bits,
        signed_a = signed,
        signed_b = signed,
        optim_type="area",
        fsa_cls=RippleCarryFinalAdder,
        full_output_bit=full_output_bit,
    )
    module = adder.to_module(f"PrefixAdder{n_bits}", with_clock=True, with_reset=True)


    vecs = AdderTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        y_w=adder.io.y.typ.width,
        num_vectors=16,
        tb_sigma=None,
        a_encoding=enc,
        b_encoding=enc,
        y_encoding=enc,
    ).generate()

    use_signed = False

    run_vectors(module, vecs, print_on_pass=True, use_signed=use_signed)

    sim = Simulator(module)
    sim.trace_enabled = True
    run_vectors_on_simulator(sim, vecs, use_signed=use_signed, print_on_pass=True, with_clk=False, test_name="SpireHDL Simulation      , Int Adder Test")

    trace_by_names = sim.get_trace_by_names()

    vcd_filename = "int_adder_tb_sim.vcd"
    write_vcd(
        trace_by_names=trace_by_names,
        filename=vcd_filename,
        top_module=module.name,
        timescale="1ns",
    )

    sim_tb = TestbenchGenSimulator(module)
    run_vectors_on_simulator(sim_tb, vecs, use_signed=use_signed, print_on_pass=False, with_clk=False, test_name="TestbenchGen Simulation, Int Adder Test")

    # print("\n".join(sim_tb.to_testbench_lines()))

    tb_filename = "int_adder_tb_sim.v"
    data_tb_filename = "int_adder_tb_data.v"
    data_filename = "int_adder_vectors.dat"

    write_vector_data_file(vecs, data_filename)

    verilog_filename = "int_adder.v"
    sim_tb.to_testbench_file(tb_filename, tb_module_name=module.name + "_tb")
    sim_tb.to_data_driver_testbench_file(
        data_tb_filename,
        data_file=data_filename,
        with_clk=False,
    )
    module.to_verilog_file(verilog_filename)

    return


if __name__ == "__main__":
    int_adders_tb_sim()
