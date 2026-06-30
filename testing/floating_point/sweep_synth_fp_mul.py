from spire.expr import reset_shared_cache
from testing.floating_point.synthesise_fp2 import flowy_optimize
from aigverse import Aig, DepthAig
from spire.arithmetic.floating_point.float_mult import build_f16_mul, build_fp_mul
from spire.arithmetic.floating_point.float_mult_sn import build_fp_mul_sn
from spire.arithmetic.floating_point.hif8 import build_hif8_mul_logic
from spire.helpers import get_yosys_transistor_count
from spire.aiger import AigerExporter, export_module_to_aiger
from spire.aig.aig_aigerverse import conv_aag_into_aig, read_aag_into_aig

from aigverse import aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing

from spire.component import Netlist


def get_result(m: Netlist, optim_steps = 200) -> tuple:
    agg_lines = AigerExporter(m).get_aag()
    aig = conv_aag_into_aig(agg_lines)
    print("nodes:", aig.size(), "PIs:", aig.num_pis(), "POs:", aig.num_pos())

    # Clone the AIG network for size comparison
    aig_clone = aig.clone()

    # Optimize the AIG with several optimization algorithms
    for i in range(optim_steps):
        for optimization in [aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing]:
            optimization(aig)

    nb_transistors = get_yosys_transistor_count(m)
    
    # Print the size of the unoptimized and optimized AIGs
    print(f"Original AIG Size:  {aig_clone.size()}")
    print(f"Optimized AIG Size: {aig.size()}")
    print(f"Number of Transistors: {nb_transistors}")    

    return aig.size(), DepthAig(aig).num_levels(), nb_transistors

def optimize_m(m: Netlist) -> Netlist:
    #return m
    return flowy_optimize(m)

def get_size_mult(n_bits: int, ew: int, subnormals=False, optim_steps = 200) -> int:

    reset_shared_cache()

    fw = n_bits - ew - 1

    m = build_fp_mul_sn(f"F{n_bits}Mul", EW=ew, FW=fw, subnormals=subnormals)

    m = optimize_m(m)

    return get_result(m, optim_steps=optim_steps)


def get_hifloat8_results():

    reset_shared_cache()

    m = build_hif8_mul_logic("HiFP8Mul_Logic_Ref")

    m = optimize_m(m)

    return get_result(m)   


def main():

    print("starting sweep for floating point multipliers")

    ew_start = 1
    n_bits = 8

    # sweep and plot

    import matplotlib.pyplot as plt
    import numpy as np
    ew_values = list(range(ew_start, n_bits - 2))
    res= [get_size_mult(n_bits, ew) for ew in ew_values]
    sizes = [size for size, _, _ in res]
    depths = [depth for _, depth, _ in res]
    nb_transistors = [nt for _, _, nt in res]
    # print("EW sizes:", ew_values)
    # print("Sizes:", sizes)

    res_sn = [get_size_mult(n_bits, ew, subnormals=True) for ew in ew_values]
    sizes_sn = [size for size, _, _ in res_sn]
    depths_sn = [depth for _, depth, _ in res_sn]
    nb_transistors_sn = [nt for _, _, nt in res_sn]

    size_hif8, depth_hif8, nb_transistors_hif8 = get_hifloat8_results()

    # plot with two subplots, one below the other
    _, ax = plt.subplots(3, 1, figsize=(7, 10)) #, sharex=True)
    ax[0].plot(ew_values, sizes, marker='o', label=f'F{n_bits}Mul (No Subnormals)')
    ax[0].plot(ew_values, sizes_sn, marker='x', label=f'F{n_bits}Mul (With Subnormals)')
    ax[0].axhline(y=size_hif8, linestyle='--', color='red', label='HIFP8Mul')
    ax[0].legend()
    ax[0].set_title(f'AIG Size vs Exponent Width (EW) for F{n_bits}Mul')
    ax[0].set_xlabel('Exponent Width (EW)')
    ax[0].set_ylabel('AIG Size')
    ax[0].set_xticks(ew_values)
    ax[0].grid()

    ax[1].plot(ew_values, depths, marker='o', label=f'F{n_bits}Mul (No Subnormals)')
    ax[1].plot(ew_values, depths_sn, marker='x', label=f'F{n_bits}Mul (With Subnormals)')
    ax[1].axhline(y=depth_hif8, linestyle='--', color='red', label='HIFP8Mul')
    ax[1].legend()
    ax[1].set_title(f'AIG Depth vs Exponent Width (EW) for F{n_bits}Mul')
    ax[1].set_xlabel('Exponent Width (EW)')
    ax[1].set_ylabel('AIG Depth')
    ax[1].set_xticks(ew_values)
    ax[1].grid()

    ax[2].plot(ew_values, nb_transistors, marker="o", label=f"F{n_bits}Mul (No Subnormals)")
    ax[2].plot(ew_values, nb_transistors_sn, marker="x", label=f"F{n_bits}Mul (With Subnormals)")
    ax[2].axhline(y=nb_transistors_hif8, linestyle='--', color='red', label='HIFP8Mul')
    ax[2].legend()
    ax[2].set_title(f"Number of Transistors vs Exponent Width (EW) for F{n_bits}Mul")
    ax[2].set_xlabel("Exponent Width (EW)")
    ax[2].set_ylabel("Number of Transistors")
    ax[2].set_xticks(ew_values)
    ax[2].grid()

    plt.tight_layout()

    plt.savefig(f'aig_size_vs_ew_{n_bits}.png')
    print(f'Saved plot to aig_size_vs_ew_{n_bits}.png')

if __name__ == "__main__":
    main()