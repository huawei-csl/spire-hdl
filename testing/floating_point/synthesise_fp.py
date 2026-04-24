import os
from testing.low_level_arithmetic.compressor_tree.compressor_tree_spire_hdl import gen_compressor_tree_graph_and_sprout_module
from spirehdl.arithmetic.floating_point.spire_hdl_float_mult_sn import build_fp_mul_sn

from flowy.flows.reinforce.data_collection.lib.definitions import Encodings, Policy, RecipeSelection, SelectionMetric, SimMode, StrategyName, Operations
from flowy.flows.reinforce.run.statistical.run_flow import run_flow
from flowy.flows.reinforce.analysis.visualize_run import visualize_run
from flowy.data_structures.database import RunDatabase, RunIdentifier

def main():

    output_dir = "testing/floating_point/generated"
    
    source_design = "fp_multiplier"
    # source_design = "int_multiplier"

    if source_design == "fp_multiplier":
        #floating point multiplier
        #Configuration
        exponent_width = 4
        fraction_width = 3 # M
        subnormals = True

        total_bits = 1 + exponent_width + fraction_width
        filename = f"fxmul_E{exponent_width}_M{fraction_width}_subn{subnormals}.v"

        # Build multiplier
        fp_multiplier = build_fp_mul_sn(
            "mydesign_comb",
            EW=exponent_width,
            FW=fraction_width,
            subnormals=subnormals,
        )
        verilog_code = fp_multiplier.to_verilog()

    elif source_design == "int_multiplier":
        n_bits = 4
        total_bits = n_bits
        g, m = gen_compressor_tree_graph_and_sprout_module(n_bits, policy="wallace")
        m.name = "mydesign_comb"
        verilog_code = m.to_verilog()
        filename = f"compressor_tree_{n_bits}bits.v"

    # Export to file
    verilog_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(verilog_path, "w") as f:
        f.write(verilog_code)
    print(f"Wrote {verilog_path} ({total_bits} bits)")


    # run flowy optimization
    db_identifier = run_flow(
        iterations=10,
        chains=10,
        chain_len=10,
        chain_workers=10,
        recipe_selection=RecipeSelection.PERFORMANCE_SAMPLING,
        experiment="exp_test_3",
        env_option="auto",
        strategy_name="equal",
        debug=False,
        selection_metric=SelectionMetric.nb_transistors,
        input_encoding=Encodings.twos_complement,
        output_encoding=Encodings.twos_complement,
        verilog_file=verilog_path,
        compression_scripts_per_step=1,
        simulation_tb=False,
    )

    visualize_run(db_identifier)
    
    # extract number of transistors
    run_db = RunDatabase(db_identifier)
    # load final best design
    run_data = run_db.load('final_mockturtle_design')

    print(f"Final best design, nb_transistors: {run_data['nb_transistors'].value}")


if __name__ == "__main__":
    main()