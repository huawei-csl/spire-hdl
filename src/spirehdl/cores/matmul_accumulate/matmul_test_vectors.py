from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from spirehdl.aggregate.aggregate_array import Array
from spirehdl.arithmetic.floating_point.fp_encoding import fp_decode, fp_encode
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, EncodingModel
from spirehdl.cores.matmul_accumulate.matmul_accumulate_core import MatmulAccumulateCore
from spirehdl.cores.matmul_accumulate.matmul_accumulate_core_float import FpMatmulAccumulateComponent


def _shape2d(arr: Array) -> Tuple[int, int]:
    return len(arr), len(arr[0])


def generate_matmul_vectors(
    core: MatmulAccumulateCore,
    encoding: Encoding,
    num_vectors: int,
    sigma: Optional[float] = None,
    encoding_ab_inputs: Optional[Encoding] = None,
) -> List[Tuple[str, Dict[str, int], Dict[str, int]]]:
    """Build flat-dict test vectors for a matmul-accumulate core.

    Args:
        core: Component whose io.A/B/C/Y ports define the shapes and widths.
        encoding: Encoding used for C and Y (and A/B when encoding_ab_inputs is None).
        num_vectors: Number of random test vectors to generate.
        sigma: If given, use normal sampling with this sigma; otherwise uniform.
        encoding_ab_inputs: Override encoding for A and B (e.g. for mixed-precision).
    """
    a_width = core.io.A[0, 0].typ.width
    b_width = core.io.B[0, 0].typ.width
    c_width = core.io.C[0, 0].typ.width
    y_width = core.io.Y[0, 0].typ.width

    a_rows, a_cols = _shape2d(core.io.A)  # A: (m, k)
    b_rows, b_cols = _shape2d(core.io.B)  # B: (k, n)
    c_rows, c_cols = _shape2d(core.io.C)  # C/Y: (m, n)

    enc_model = EncodingModel(encoding)
    enc_model_inputs = EncodingModel(encoding_ab_inputs) if encoding_ab_inputs is not None else enc_model

    vectors: List[Tuple[str, Dict[str, int], Dict[str, int]]] = []

    for idx in range(num_vectors):
        if sigma is None:
            a_vals = enc_model_inputs.get_uniform_sample_np(a_width, size=(a_rows, a_cols))
            b_vals = enc_model_inputs.get_uniform_sample_np(b_width, size=(b_rows, b_cols))
            c_vals = enc_model.get_uniform_sample_np(c_width, size=(c_rows, c_cols))
        else:
            a_vals = enc_model_inputs.get_normal_sample_np(a_width, sigma, size=(a_rows, a_cols))
            b_vals = enc_model_inputs.get_normal_sample_np(b_width, sigma, size=(b_rows, b_cols))
            c_vals = enc_model.get_normal_sample_np(c_width, sigma, size=(c_rows, c_cols))

        y_vals = a_vals @ b_vals + c_vals

        a_vals = np.vectorize(lambda x: enc_model_inputs.encode_value(int(x), a_width))(a_vals)
        b_vals = np.vectorize(lambda x: enc_model_inputs.encode_value(int(x), b_width))(b_vals)
        c_vals = np.vectorize(lambda x: enc_model.encode_value(int(x), c_width))(c_vals)
        y_vals = np.vectorize(lambda x: enc_model.encode_value(int(x), y_width))(y_vals)

        ins: Dict[str, int] = {}
        outs: Dict[str, int] = {}

        for i in range(a_rows):
            for k in range(a_cols):
                ins[core.io.A[i, k].name] = int(a_vals[i, k])

        for k in range(b_rows):
            for j in range(b_cols):
                ins[core.io.B[k, j].name] = int(b_vals[k, j])

        for i in range(c_rows):
            for j in range(c_cols):
                ins[core.io.C[i, j].name] = int(c_vals[i, j])
                outs[core.io.Y[i, j].name] = int(y_vals[i, j])

        vectors.append((f"vec_{idx}", ins, outs))

    return vectors


def generate_fp_matmul_vectors(
    core: FpMatmulAccumulateComponent,
    num_vectors: int,
    *,
    seed: int = 42,
) -> List[Tuple[str, Dict[str, int], Dict[str, int]]]:
    """Generate bit-exact test vectors for an FpMatmulAccumulateComponent.

    The reference computation follows the same sequential left-to-right
    accumulation order as elaborate():
        dot = A[i,0]*B[0,j]
        dot = dot + A[i,1]*B[1,j]  ...
        acc = dot + C[i,j]
    Both multiplications and additions use fp_encode (round-to-nearest-even),
    matching the IEEE-compliant hardware.
    """
    ft = core.cfg.ftype
    ew, fw = ft.exponent_width, ft.fraction_width
    sn = ft.subnormal_support
    dims = core.cfg.dims
    m, n, k = dims.dim_m, dims.dim_n, dims.dim_k
    rng = np.random.default_rng(seed=seed)

    vectors: List[Tuple[str, Dict[str, int], Dict[str, int]]] = []

    for idx in range(num_vectors):
        a_f = rng.uniform(0.1, 2.0, size=(m, k))
        b_f = rng.uniform(0.1, 2.0, size=(k, n))
        c_f = rng.uniform(0.1, 2.0, size=(m, n))

        a_bits = np.vectorize(lambda x: fp_encode(float(x), ew, fw, subnormals=sn))(a_f)
        b_bits = np.vectorize(lambda x: fp_encode(float(x), ew, fw, subnormals=sn))(b_f)
        c_bits = np.vectorize(lambda x: fp_encode(float(x), ew, fw, subnormals=sn))(c_f)

        a_q = np.vectorize(lambda b: fp_decode(int(b), ew, fw))(a_bits)
        b_q = np.vectorize(lambda b: fp_decode(int(b), ew, fw))(b_bits)

        y_bits = np.zeros((m, n), dtype=int)
        for i in range(m):
            for j in range(n):
                dot = fp_decode(fp_encode(a_q[i, 0] * b_q[0, j], ew, fw, subnormals=sn), ew, fw)
                for kk in range(1, k):
                    prod = fp_decode(fp_encode(a_q[i, kk] * b_q[kk, j], ew, fw, subnormals=sn), ew, fw)
                    dot = fp_decode(fp_encode(dot + prod, ew, fw, subnormals=sn), ew, fw)
                y_bits[i, j] = fp_encode(dot + fp_decode(int(c_bits[i, j]), ew, fw), ew, fw, subnormals=sn)

        ins: Dict[str, int] = {}
        outs: Dict[str, int] = {}
        for i in range(m):
            for kk in range(k):
                ins[core.io.A[i, kk].bits.name] = int(a_bits[i, kk])
        for kk in range(k):
            for j in range(n):
                ins[core.io.B[kk, j].bits.name] = int(b_bits[kk, j])
        for i in range(m):
            for j in range(n):
                ins[core.io.C[i, j].bits.name] = int(c_bits[i, j])
                outs[core.io.Y[i, j].bits.name] = int(y_bits[i, j])

        vectors.append((f"vec_{idx}", ins, outs))

    return vectors
