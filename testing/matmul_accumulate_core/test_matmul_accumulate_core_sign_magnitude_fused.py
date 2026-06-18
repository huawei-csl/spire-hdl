from __future__ import annotations

import numpy as np

from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, EncodingModel, is_signed
from spire.arithmetic.int_arithmetic_config import AdderConfig
from spire.cores.matmul_accumulate.matmul_accumulate_core import MMAcDims, MMAcWidths, max_y_width_unsigned
from spire.cores.matmul_accumulate.matmul_accumulate_core_sign_magnitude_fused import (
    MMAcEncodedCfg,
    MatmulAccumulateComponent,
    MultiplierConfig,
    SignMagnitudeEncoderConfig,
)
from spire.helpers import get_yosys_metrics
from spire.simulator import Simulator


def test_mmac_core_sign_magnitude_pipeline():
    dim = 4
    dim_k = dim
    a_width = 4
    b_width = 4
    c_width = max_y_width_unsigned(a_width, b_width, dim_k, include_carry_from_add=False)
    no_encoding_decoding = False # just for testing

    encoding = Encoding.twos_complement_symmetric  # Encoding.twos_complement_symmetric or Encoding.twos_complement
    signed_io_type = False

    # optional: disable encoders when using StageBasedSignMagnitudeToTwosComplementMultiplier
    encoding_cfg = SignMagnitudeEncoderConfig(
        encoder_clip_most_negative=True if encoding == Encoding.twos_complement else False,
        decoder_clip_most_negative=False
    )

    mult_cfg = MultiplierConfig(
        use_operator=False,
        multiplier_opt=MultiplierOption.STAGE_BASED_SIGN_MAGNITUDE_MULTIPLIER if encoding == Encoding.twos_complement_symmetric else MultiplierOption.STAGE_BASED_SIGN_MAGNITUDE_EXT_MULTIPLIER,
        encodings=TwoInputAritEncodings.with_enc(Encoding.sign_magnitude if encoding == Encoding.twos_complement_symmetric else Encoding.sign_magnitude_ext),
        ppg_opt=PPGOption.AND,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
    )

    # for testing - remove encoder/decoders
    if no_encoding_decoding:
        encoding = Encoding.twos_complement
        encoding_cfg.encoder_cls = None
        encoding_cfg.decoder_cls = None
        mult_cfg.multiplier_opt = MultiplierOption.STAGE_BASED_MULTIPLIER
        mult_cfg.encodings = TwoInputAritEncodings.with_enc(encoding)
        mult_cfg.ppg_opt = PPGOption.BAUGH_WOOLEY if is_signed(encoding) else PPGOption.AND
        

    add_cfg = AdderConfig(use_operator=False, fsa_opt=FSAOption.RIPPLE_CARRY, full_output_bit=True, encoding=encoding) # what is the encoding in add_cfg

    dims = MMAcDims(dim_m=dim, dim_n=dim, dim_k=dim_k)
    widths = MMAcWidths(a_width=a_width, b_width=b_width, c_width=c_width)
    cfg = MMAcEncodedCfg(dims=dims, widths=widths, mult_cfg=mult_cfg, add_cfg=add_cfg, encoding_cfg=encoding_cfg)

    core = MatmulAccumulateComponent(cfg, signed_io_type=signed_io_type)

    print(
        f"Output matrix Y has shape: ({dim}, {dim}) with element width {core.io.Y[0,0].typ.width} bits"
    )

    module = core.to_module("matmul_accumulate_core_sign_mag")
    sim = Simulator(module)

    a_vals = EncodingModel(encoding).get_uniform_sample_np(a_width, size=(dim, dim))
    b_vals = EncodingModel(encoding).get_uniform_sample_np(b_width, size=(dim, dim))
    c_vals = EncodingModel(encoding).get_uniform_sample_np(c_width, size=(dim, dim))

    for i in range(dim):
        for j in range(dim):
            sim.set(core.io.A[i, j], int(a_vals[i, j]))
            sim.set(core.io.B[i, j], int(b_vals[i, j]))
            sim.set(core.io.C[i, j], int(c_vals[i, j]))

    sim.eval()

    y_hw = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            y_hw[i, j] = sim.get(core.io.Y[i, j])

    y_np = a_vals @ b_vals + c_vals
    if signed_io_type:
        assert np.array_equal(y_hw, y_np), "Simulation mismatch for matmul accumulate core"
    else:
        sim.peek(core.io.Y[0, 1][0])
        # encode each element according to the encoding
        y_np_encoded = np.vectorize(
            lambda x: EncodingModel(encoding).encode_value(int(x), core.io.Y[0, 0].typ.width)
        )(y_np)
        assert np.array_equal(y_hw, y_np_encoded), "Simulation mismatch for matmul accumulate core"

    yosys_metrics = get_yosys_metrics(module)
    print(f"Yosys metrics: {yosys_metrics}")


if __name__ == "__main__":
    test_mmac_core_sign_magnitude_pipeline()
