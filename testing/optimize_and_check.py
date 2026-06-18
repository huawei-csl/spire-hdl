from aigverse import Aig, DepthAig
from spire.expr import UInt
from spire.arithmetic.floating_point.spire_hdl_float_mult import build_f16_mul, build_fp_mul, run_vectors_aby
from spire.arithmetic.floating_point.spire_hdl_float_mult_sn import build_fp_mul_sn
from spire.aiger import AigerExporter, AigerImporter, export_module_to_aiger
from spire.aig.aig_aigerverse import _get_aag_sym, conv_aag_into_aig, conv_aig_into_aag, read_aag_into_aig

from aigverse import aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing

from spire.component import IOCollector
from spire.arithmetic.floating_point.fp_encoding import fp_decode
from spire.arithmetic.floating_point.fp_mul_testvectors import build_f16_subnormal_ext_vectors, build_f16_subnormal_vectors, build_f16_vectors


def get_size_mult(ew: int, subnormals=False) -> int:

    ew=5
    subnormals = True

    fw = 16 - ew - 1
    m = build_fp_mul_sn("F16Mul", EW=ew, FW=fw, subnormals=subnormals)

    aag = AigerExporter(m).get_aag()
    aag_sym = _get_aag_sym(aag)
    m_right_back = AigerImporter(aag[:-2]+aag_sym).get_spire_module()

    collector = IOCollector()
    spire_collected = collector.group(
        m_right_back,
        {
            "a": UInt(16),
            "b": UInt(16),
            "y": UInt(16),
        },
    )

    aig = conv_aag_into_aig(aag, Aig())

    aag_bak = conv_aig_into_aag(aig)
    aig_bak = conv_aag_into_aig(aag_bak, Aig())

    # assert equivalence
    from aigverse import equivalence_checking
    assert equivalence_checking(aig, aig_bak), "AIGs are not equivalent after conversion!"

    # aig = read_aiger_into_aig("i10.aig")  # or any AIGER file
    print("nodes:", aig.size(), "PIs:", aig.num_pis(), "POs:", aig.num_pos())

    # Clone the AIG network for size comparison
    aig_clone = aig.clone()

    # Optimize the AIG with several optimization algorithms
    # n_iter_optimizations = 100
    # for i in range(n_iter_optimizations):
    #     for optimization in [aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing]:
    #         optimization(aig)

    # Print the size of the unoptimized and optimized AIGs
    print(f"Original AIG Size:  {aig_clone.size()}")
    print(f"Optimized AIG Size: {aig.size()}")

    aag = conv_aig_into_aag(aig)

    spire = AigerImporter(aag[:-2]+aag_sym).get_spire_module()

    collector = IOCollector()
    spire_collected = collector.group(spire, {
        "a": UInt(16),
        "b": UInt(16),
        "y": UInt(16),
    })

    aag2 = AigerExporter(spire).get_aag()
    aig2 = conv_aag_into_aig(aag2, Aig())

    from aigverse import equivalence_checking
    assert equivalence_checking(aig_clone, aig2), "AIGs are not equivalent after conversion!"

    run_vectors_aby(m, build_f16_vectors(), label="float16 default cases", decoder=lambda b: fp_decode(b, ew, fw))
    run_vectors_aby(m, build_f16_subnormal_vectors(), label="float16 subnormal cases", decoder=lambda b: fp_decode(b, ew, fw))

    run_vectors_aby(spire, build_f16_vectors(), label="float16 default cases", decoder=lambda b: fp_decode(b, ew, fw))
    run_vectors_aby(spire, build_f16_subnormal_vectors(), label="float16 subnormal cases", decoder=lambda b: fp_decode(b, ew, fw))
    run_vectors_aby(spire, build_f16_subnormal_ext_vectors(), label="float16 subnormal ext cases", decoder=lambda b: fp_decode(b, ew, fw))

    return aig.size(), DepthAig(aig).num_levels()


def main():
    
    # sweep and plot
    
    import matplotlib.pyplot as plt
    import numpy as np
    ew_values = list(range(1, 14))
    size_depth= [get_size_mult(ew) for ew in ew_values]
    sizes = [size for size, _ in size_depth]
    depths = [depth for _, depth in size_depth]
    #print("EW sizes:", ew_values)
    #print("Sizes:", sizes)
    
    #sizes_sn = [get_size_mult(ew, subnormals=True) for ew in ew_values]
    #print("EW sizes (subnormals):", ew_values)
    #print("Sizes (subnormals):", sizes_sn)
    
    plt.figure(figsize=(10, 6))
    plt.plot(ew_values, sizes, marker='o', label='F16Mul (No Subnormals)')
    #plt.plot(ew_values, sizes_sn, marker='x', label='F16Mul (With Subnormals)')
    plt.legend()
    plt.title('AIG Size vs Exponent Width (EW) for F16Mul')
    plt.xlabel('Exponent Width (EW)')
    plt.ylabel('AIG Size')
    plt.xticks(ew_values)
    plt.grid()
    plt.savefig('aig_size_vs_ew.png')
    print('Saved plot to aig_size_vs_ew.png')

if __name__ == "__main__":
    main()