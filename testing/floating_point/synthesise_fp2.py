from testing.low_level_arithmetic.compressor_tree.compressor_tree_spire_hdl import gen_compressor_tree_graph_and_spire_module
from spire.arithmetic.floating_point.spire_hdl_float_mult_sn import build_fp_mul_sn

from spire.optimize import flowy_optimize

from spire.helpers import get_aig_stats, get_yosys_metrics, get_yosys_transistor_count
from spire.component import Component, Module

def main():

    output_dir = "testing/floating_point/generated"

    source_design = "fp_multiplier"
    # source_design = "int_multiplier"

    if source_design == "fp_multiplier":
        # floating point multiplier
        # Configuration
        exponent_width = 4
        fraction_width = 3 # M
        subnormals = True

        total_bits = 1 + exponent_width + fraction_width
        filename = f"fxmul_E{exponent_width}_M{fraction_width}_subn{subnormals}.v"

        # Build multiplier
        m = build_fp_mul_sn(
            "mydesign_comb",
            EW=exponent_width,
            FW=fraction_width,
            subnormals=subnormals,
        )
        verilog_code = m.to_verilog()

    elif source_design == "int_multiplier":
        n_bits = 4
        total_bits = n_bits
        g, m = gen_compressor_tree_graph_and_spire_module(n_bits, policy="wallace")
        m.name = "mydesign_comb"
        verilog_code = m.to_verilog()
        filename = f"compressor_tree_{n_bits}bits.v"

  


    m_optimized = flowy_optimize(m, nb_runs=2)

    # c = module.to_component()
    
    def get_module_stats(module: Module):
        transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
        yosys_metrics = get_yosys_metrics(module)
        aig_gates = get_aig_stats(module)

        print(f"Design stats: transistor_count={transistor_count}, yosys_metrics={yosys_metrics}, aig_gates={aig_gates}")
        
        return {
            "transistor_count": transistor_count,
            "yosys_metrics": yosys_metrics,
            "aig_gates": aig_gates,
        }

    print("Original design stats:")
    m_orig_stats = get_module_stats(m)
    print("Optimized design stats:")
    m_opt_stats = get_module_stats(m_optimized)

    print(f"Original vs Optimized transistor count: {m_orig_stats['transistor_count']} vs {m_opt_stats['transistor_count']}")
    print(f"Original vs Optimized AIG gates: {m_orig_stats['aig_gates']} vs {m_opt_stats['aig_gates']}")


if __name__ == "__main__":
    main()