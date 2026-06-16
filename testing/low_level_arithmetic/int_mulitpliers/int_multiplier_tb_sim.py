from enum import Enum
from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, TwoInputAritEncodings, PPAOption, PPGOption
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, MultiplierTestVectors
from spirehdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import MultiplierTestVectorsInt, StageBasedMultiplierBasic
from spirehdl.arithmetic.int_multipliers.multipliers.multipliers_ext_optimized import OptimizedMultiplier, OptimizedMultiplierFrom4BitBlocksStrong
from spirehdl.arithmetic.int_multipliers.multipliers.multipliers_ext import StarMultiplier
from spirehdl.helpers import extract_yosys_metrics_from_verilog_file, get_yosys_metrics, run_vectors, run_vectors_on_simulator
from spirehdl.spirehdl_simulator import Simulator
from spirehdl.spirehdl_verilog_testbench import TestbenchGenSimulator, write_vector_data_file
from spirehdl.various.vcd_writer import write_vcd


def int_tb_sim():
    n_bits = 4
    signed = False  

    # enum selection for which multiplier to test
    class MultiplierType(Enum):
        STAR = 0
        STAGE_BASED = 1
        OPTIMIZED = 2

    sel = MultiplierType.OPTIMIZED

    if sel == MultiplierType.STAR:

        mult = StarMultiplier(
            a_w=n_bits,
            b_w=n_bits,
            a_encoding=Encoding.unsigned if not signed else Encoding.twos_complement,
            b_encoding=Encoding.unsigned if not signed else Encoding.twos_complement,
        )
        
    elif sel == MultiplierType.STAGE_BASED:

        mult = StageBasedMultiplierBasic(
            a_w=n_bits,
            b_w=n_bits,
            signed_a=signed,
            signed_b=signed,
            optim_type="area",
            ppg_cls=PPGOption.BOOTH_OPTIMISED.value,  # PPGOption.AND.value,
            ppa_cls=PPAOption.CARRY_SAVE_TREE.value, #PPAOption.WALLACE_TREE.value,
            fsa_cls=FSAOption.PLUS_OPERATOR.value #FSAOption.PREFIX_ZCG.value,  # FSAOption.RIPPLE.value,
        )

    elif sel == MultiplierType.OPTIMIZED:

        #mult = OptimizedMultiplierFrom4BitBlocksStrong(
        mult = OptimizedMultiplier(
            a_w=n_bits,
            b_w=n_bits,
            optim_type="area",
            ppg_cls=PPGOption.NONE.value,
        )

    with_clk = True
    module = mult.to_module(f"Mul{n_bits}", with_clock=with_clk)


    # get yosys transistor count
    yosys_metrics = get_yosys_metrics(module)
    print(f"Yosys metrics: {yosys_metrics}")

    file_name = f"temp.v"
    module.to_verilog_file(file_name)
    yosys_metrics_verilog = extract_yosys_metrics_from_verilog_file(file_name)
    print(f"Yosys metrics from verilog: {yosys_metrics_verilog}")

    decoder = None

    encodings = TwoInputAritEncodings.with_enc(Encoding.unsigned if not signed else Encoding.twos_complement,)
    vecs = MultiplierTestVectors(
                a_w=n_bits,
                b_w=n_bits,
                y_w=mult.io.y.typ.width,
                num_vectors=16,
                tb_sigma=None,
                a_encoding=encodings.a,
                b_encoding=encodings.b,
                y_encoding=encodings.y,
            ).generate()

    use_signed = False

    run_vectors(module, vecs, print_on_pass=True, use_signed=use_signed)

    sim = Simulator(module)
    sim.trace_enabled = True
    run_vectors_on_simulator(sim, vecs, decoder=decoder, use_signed=use_signed, print_on_pass=True, with_clk=with_clk)

    trace_by_names = sim.get_trace_by_names()
    # print(f"Traced signals: {list(trace_by_names.keys())}")

    vcd_filename = "int_multiplier_tb_sim.vcd"
    write_vcd(trace_by_names=trace_by_names, 
              filename=vcd_filename,
              top_module=module.name, 
              timescale="1ns")

    sim_tb = TestbenchGenSimulator(module)
    run_vectors_on_simulator(sim_tb, vecs, decoder=decoder, use_signed=use_signed, print_on_pass=False, with_clk=with_clk)

    print("\n".join(sim_tb.to_testbench_lines()))

    tb_filename = "int_multiplier_tb_sim.v"
    data_tb_filename = "int_multiplier_tb_data.v"
    data_filename = "int_multiplier_vectors.dat"

    write_vector_data_file(vecs, data_filename)

    verilog_filename = "int_multiplier.v"
    sim_tb.to_testbench_file(tb_filename, tb_module_name=module.name+"_tb")
    sim_tb.to_data_driver_testbench_file(
        data_tb_filename,
        data_file=data_filename,
        input_stimuli=["a", "b"],
        outputs_expected=["y"],
        with_clk=with_clk
    )
    module.to_verilog_file(verilog_filename)

    return

if __name__ == "__main__":
    int_tb_sim()
