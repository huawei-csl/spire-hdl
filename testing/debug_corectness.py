from aigverse import equivalence_checking
from spire.aig.aig_aigerverse import _get_aag_sym, conv_aag_into_aig
from spire.expr import UInt
from spire.aiger import AigerExporter, AigerImporter
from spire.arithmetic.floating_point.spire_hdl_float_mult import run_vectors_aby
from spire.arithmetic.floating_point.spire_hdl_float_mult_sn import build_fp_mul_sn
from spire.arithmetic.floating_point.fp_encoding import fp_decode
from spire.arithmetic.floating_point.fp_mul_testvectors import build_f16_subnormal_ext_vectors, build_f16_subnormal_vectors
from spire.component import IOCollector


def main():
    ew=5
    subnormals = True

    fw = 16 - ew - 1
    m = build_fp_mul_sn("F16Mul", EW=ew, FW=fw, subnormals=subnormals)
    # m3 = build_fp_mul_sn("F16Mul3", EW=ew+1, FW=fw-1, subnormals=subnormals)

    aag = AigerExporter(m).get_aag()
    aig = conv_aag_into_aig(aag)

    aag_sym = _get_aag_sym(aag)
    m2 = AigerImporter(aag[:-2]+aag_sym).get_spire_module()

    aag2 = AigerExporter(m2).get_aag()
    aig2 = conv_aag_into_aig(aag2)

    assert equivalence_checking(aig, aig2), "AIGs are not equivalent after conversion!"

    collector = IOCollector()
    collector.group(m2, {
        "a": UInt(16),
        "b": UInt(16),
        "y": UInt(16),
    })

    run_vectors_aby(m, build_f16_subnormal_vectors(), label="float16 subnormal cases", decoder=lambda b: fp_decode(b, 5, 10))
    run_vectors_aby(m, build_f16_subnormal_ext_vectors(), label="float16 subnormal ext cases", decoder=lambda b: fp_decode(b, 5, 10))

    run_vectors_aby(m2, build_f16_subnormal_vectors(), label="float16 subnormal cases", decoder=lambda b: fp_decode(b, 5, 10))
    run_vectors_aby(m2, build_f16_subnormal_ext_vectors(), label="float16 subnormal ext cases", decoder=lambda b: fp_decode(b, 5, 10))


if __name__ == "__main__":
    main()