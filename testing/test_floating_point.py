from __future__ import annotations
import math
import pathlib
import sys

import numpy as np

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "src"))
sys.path.append(str(repo_root))

from spirehdl.arithmetic.floating_point.fp_encoding import fp_decode
from spirehdl.arithmetic.floating_point.fp_mul_testvectors import (
    build_f16_subnormal_vectors,
    build_f16_vectors,
)

from spirehdl.aggregate.aggregate_floating_point import FloatingPoint, FloatingPointType
from spirehdl.aggregate.aggregate_register import AggregateRegister
from spirehdl.arithmetic.floating_point.spire_hdl_float_mult import build_f16_mul
from spirehdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding
from spirehdl.arithmetic.int_arithmetic_config import AdderConfig, MultiplierConfig
from spirehdl.spirehdl import Const, Expr, UInt, as_expr, reset_shared_cache
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator


def _eval_expr(e: Expr) -> int:
    m = Module("Eval", with_clock=False, with_reset=False)
    sim = Simulator(m)
    return sim.peek(e)


def _find_signal(mod: Module, name: str):
    return next(sig for sig in mod._signals if sig.name == name)


def _decode_half(bits: int) -> float:
    return float(np.frombuffer(np.uint16(bits).tobytes(), dtype=np.float16)[0])


def _encode_half(val: float) -> int:
    return int(np.float16(val).view(np.uint16))


def _assert_fp_match(got_bits: int, expected_bits: int, *, require_bit_exact: bool = True):
    got_val = fp_decode(got_bits, 5, 10)
    expected_val = fp_decode(expected_bits, 5, 10)

    if math.isnan(expected_val):
        assert math.isnan(got_val)
        return

    if math.isinf(expected_val):
        assert math.isinf(got_val)
        assert math.copysign(1.0, got_val) == math.copysign(1.0, expected_val)
        return

    assert math.isclose(got_val, expected_val, rel_tol=1e-3, abs_tol=1e-3)
    if require_bit_exact:
        assert got_bits == expected_bits


def test_sim_aggregate_register_floating_point():
    ft = FloatingPointType(exponent_width=5, fraction_width=10)
    m = Module("FpAggReg", with_clock=True, with_reset=False)

    x = m.input(UInt(ft.width_total), "x")

    acc = AggregateRegister(FloatingPoint, ft, name="fp_reg", init=0)
    m._signals.append(acc.bits)

    x_view = FloatingPoint(ft, bits=as_expr(x))
    acc <<= x_view

    y = m.output(UInt(ft.width_total), "y")
    y <<= acc.bits

    sim = Simulator(m)

    sim.eval()
    assert sim.get("y") == 0
    assert sim.get("fp_reg") == 0

    one_raw = 0x3C00  # 1.0 in binary16
    sim.set("x", one_raw)
    sim.step(1)

    assert sim.get("fp_reg") == one_raw
    assert sim.get("y") == one_raw


def test_floating_point_field_access_and_mul():
    reset_shared_cache()

    ft = FloatingPointType(exponent_width=5, fraction_width=10)

    raw = 0x3C00  # +1.0 in binary16
    fp = FloatingPoint(ft, bits=Const(raw, ft.to_hdl_type()))

    assert _eval_expr(fp.sign) == 0
    assert _eval_expr(fp.exponent) == 0x0F
    assert _eval_expr(fp.fraction) == 0

    mod = build_f16_mul()
    sim = Simulator(mod)
    sim.set("a", raw).set("b", raw).eval()

    prod_bits = sim.get("y")
    y_sig = _find_signal(mod, "y")
    y_view = FloatingPoint(ft, bits=y_sig)

    assert prod_bits == sim.peek(y_view.bits)


def test_floating_point_mul_matches_numpy_random():
    reset_shared_cache()

    ft = FloatingPointType(exponent_width=5, fraction_width=10)
    mod = build_f16_mul()
    sim = Simulator(mod)

    rng = np.random.default_rng(2024)

    for _ in range(50):
        a_f = np.float16(rng.uniform(-8.0, 8.0))
        b_f = np.float16(rng.uniform(-8.0, 8.0))

        a_bits = int(a_f.view(np.uint16))
        b_bits = int(b_f.view(np.uint16))

        sim.set("a", a_bits).set("b", b_bits).eval()
        got_bits = sim.get("y")

        expected_bits = int((a_f * b_f).view(np.uint16))

        got_val = _decode_half(got_bits)
        expected_val = _decode_half(expected_bits)

        if math.isnan(expected_val):
            assert math.isnan(got_val)
            continue

        if math.isinf(expected_val):
            assert math.isinf(got_val) and (math.copysign(1.0, got_val) == math.copysign(1.0, expected_val))
            continue

        assert math.isclose(got_val, expected_val, rel_tol=1e-3, abs_tol=1e-3)
        assert got_bits == expected_bits


def test_floating_point_mul_matches_general_vectors():
    reset_shared_cache()

    mod = build_f16_mul()
    sim = Simulator(mod)

    for name, a_bits, b_bits, expected_bits in build_f16_vectors():
        sim.set("a", a_bits).set("b", b_bits).eval()
        got_bits = sim.get("y")

        try:
            _assert_fp_match(got_bits, expected_bits)
        except AssertionError as exc:  # pragma: no cover - aids debugging
            raise AssertionError(f"Vector '{name}' failed: got 0x{got_bits:04x}, expected 0x{expected_bits:04x}") from exc


def test_floating_point_mul_matches_subnormal_vectors():
    reset_shared_cache()

    mod = build_f16_mul()
    sim = Simulator(mod)

    for name, a_bits, b_bits, expected_bits in build_f16_subnormal_vectors():
        sim.set("a", a_bits).set("b", b_bits).eval()
        got_bits = sim.get("y")

        try:
            _assert_fp_match(got_bits, expected_bits, require_bit_exact=False)
        except AssertionError as exc:  # pragma: no cover - aids debugging
            raise AssertionError(f"Vector '{name}' failed: got 0x{got_bits:04x}, expected 0x{expected_bits:04x}") from exc


def _build_fp_binop_module(op_name: str) -> Module:
    ft = FloatingPointType(exponent_width=5, fraction_width=10)
    m = Module(f"FpAgg{op_name.title()}", with_clock=False, with_reset=False)

    a_bits = m.input(UInt(ft.width_total), "a")
    b_bits = m.input(UInt(ft.width_total), "b")

    a_fp = FloatingPoint(ft, bits=as_expr(a_bits))
    b_fp = FloatingPoint(ft, bits=as_expr(b_bits))

    if op_name == "add":
        res_fp = a_fp + b_fp
    elif op_name == "mul":
        res_fp = a_fp * b_fp
    else:
        raise ValueError(f"Unsupported op '{op_name}'")

    y_bits = m.output(UInt(ft.width_total), "y")
    y_bits <<= res_fp.bits

    return m


def test_aggregate_floating_point_add():
    reset_shared_cache()
    mod = _build_fp_binop_module("add")
    sim = Simulator(mod)

    vectors = [
        (1.0, 2.0),
        (0.5, 0.5),
        (1.5, -1.25),
        (-2.0, 3.0),
    ]

    for a_f, b_f in vectors:
        sim.set("a", _encode_half(a_f)).set("b", _encode_half(b_f)).eval()
        got_bits = sim.get("y")
        expected_bits = _encode_half(np.float16(a_f) + np.float16(b_f))
        try:
            _assert_fp_match(got_bits, expected_bits)
        except AssertionError as exc:
            raise AssertionError(
                f"add failed for a={a_f}, b={b_f}: got 0x{got_bits:04x}, expected 0x{expected_bits:04x}"
            ) from exc


def test_aggregate_floating_point_mul():
    reset_shared_cache()
    mod = _build_fp_binop_module("mul")
    sim = Simulator(mod)

    vectors = [
        (1.5, 2.0),
        (1.25, -0.5),
        (-2.0, -2.0),
        (0.5, 0.5),
    ]

    for a_f, b_f in vectors:
        sim.set("a", _encode_half(a_f)).set("b", _encode_half(b_f)).eval()
        got_bits = sim.get("y")
        expected_bits = _encode_half(np.float16(a_f) * np.float16(b_f))
        try:
            _assert_fp_match(got_bits, expected_bits)
        except AssertionError as exc:
            raise AssertionError(
                f"mul failed for a={a_f}, b={b_f}: got 0x{got_bits:04x}, expected 0x{expected_bits:04x}"
            ) from exc


# ---- Tests with adder / multiplier configs ----

_FP16_ADD_VECTORS = [(1.0, 2.0), (0.5, 0.5), (1.5, -1.25), (-2.0, 3.0)]
_FP16_MUL_VECTORS = [(1.5, 2.0), (1.25, -0.5), (-2.0, -2.0), (0.5, 0.5)]


def _build_fp_binop_module_with_cfg(op_name: str, adder_cfg=None, mult_cfg=None) -> Module:
    ft = FloatingPointType(exponent_width=5, fraction_width=10)
    m = Module(f"FpAgg{op_name.title()}Cfg", with_clock=False, with_reset=False)

    a_bits = m.input(UInt(ft.width_total), "a")
    b_bits = m.input(UInt(ft.width_total), "b")

    a_fp = FloatingPoint(ft, bits=as_expr(a_bits), adder_cfg=adder_cfg, mult_cfg=mult_cfg)
    b_fp = FloatingPoint(ft, bits=as_expr(b_bits))

    res_fp = a_fp + b_fp if op_name == "add" else a_fp * b_fp

    y_bits = m.output(UInt(ft.width_total), "y")
    y_bits <<= res_fp.bits
    return m


def test_fp_add_with_adder_cfg_use_operator():
    """FpAdd with AdderConfig(use_operator=True) gives identical results to the default path."""
    reset_shared_cache()
    mod = _build_fp_binop_module_with_cfg("add", adder_cfg=AdderConfig(use_operator=True))
    sim = Simulator(mod)
    for a_f, b_f in _FP16_ADD_VECTORS:
        sim.set("a", _encode_half(a_f)).set("b", _encode_half(b_f)).eval()
        _assert_fp_match(sim.get("y"), _encode_half(np.float16(a_f) + np.float16(b_f)))


def test_fp_add_with_explicit_adder_cfg():
    """FpAdd with an explicit stage-based (ripple-carry) adder gives correct results."""
    reset_shared_cache()
    adder_cfg = AdderConfig(use_operator=False, fsa_opt=FSAOption.RIPPLE_CARRY, encoding=Encoding.unsigned)
    mod = _build_fp_binop_module_with_cfg("add", adder_cfg=adder_cfg)
    sim = Simulator(mod)
    for a_f, b_f in _FP16_ADD_VECTORS:
        sim.set("a", _encode_half(a_f)).set("b", _encode_half(b_f)).eval()
        _assert_fp_match(sim.get("y"), _encode_half(np.float16(a_f) + np.float16(b_f)))


def test_fp_mul_with_mult_cfg_use_operator():
    """FpMul with MultiplierConfig(use_operator=True) gives identical results to the default path."""
    reset_shared_cache()
    mod = _build_fp_binop_module_with_cfg("mul", mult_cfg=MultiplierConfig(use_operator=True))
    sim = Simulator(mod)
    for a_f, b_f in _FP16_MUL_VECTORS:
        sim.set("a", _encode_half(a_f)).set("b", _encode_half(b_f)).eval()
        _assert_fp_match(sim.get("y"), _encode_half(np.float16(a_f) * np.float16(b_f)))


def test_fp_mul_with_explicit_mult_cfg():
    """FpMul with a stage-based (Wallace tree + ripple carry) multiplier gives correct results."""
    reset_shared_cache()
    mult_cfg = MultiplierConfig(
        use_operator=False,
        multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
        encodings=TwoInputAritEncodings.with_enc(Encoding.unsigned),
        ppg_opt=PPGOption.AND,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
    )
    mod = _build_fp_binop_module_with_cfg("mul", mult_cfg=mult_cfg)
    sim = Simulator(mod)
    for a_f, b_f in _FP16_MUL_VECTORS:
        sim.set("a", _encode_half(a_f)).set("b", _encode_half(b_f)).eval()
        _assert_fp_match(sim.get("y"), _encode_half(np.float16(a_f) * np.float16(b_f)))
