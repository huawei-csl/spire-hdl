import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pytest

from spire.arithmetic.floating_point.spire_hdl_hif8 import build_hif8_mul_logic, build_hif8_mul_lut, catalogue_summary, float_to_hif8, hif8_to_float, multiply_hif8
from spire.arithmetic.floating_point.spire_hdl_hif8_via_fp8 import build_hif8_mul_via_fp8_module
from spire.simulator import Simulator


@pytest.mark.parametrize(
    "encoding,expected",
    [
        (0x6F, math.inf),
        (0xEF, -math.inf),
        (0x33, 0.34375),
        (0x6E, 32768.0),
        (0x7E, 3.0517578125e-05),
        (0x01, 2 ** -22),
        (0x07, 2 ** -16),
    ],
)
def test_hif8_to_float(encoding, expected):
    decoded = hif8_to_float(encoding)
    if math.isinf(expected):
        assert math.isinf(decoded) and math.copysign(1.0, decoded) == math.copysign(1.0, expected)
    else:
        assert decoded == pytest.approx(expected, rel=0, abs=1e-9)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, 0x00),
        (math.inf, 0x6F),
        (-math.inf, 0xEF),
        (2 ** 15, 0x6E),
        (2 ** -15, 0x7E),
        (2 ** -16, 0x07),
        (2 ** -22, 0x01),
        (0.34375, 0x33),
        (-0.34375, 0xB3),
    ],
)
def test_float_to_hif8(value, expected):
    assert float_to_hif8(value) == expected


def test_catalogue_basic_counts():
    summary = catalogue_summary()
    assert summary["total"] > 0
    assert summary["normals"] > 0
    assert summary["denormals"] == 7  # M = 1..7


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (0x6F, 0x33, 0x6F),  # inf * finite
        (0x33, 0x6F, 0x6F),  # finite * inf
        (0x00, 0x6F, 0x80),  # zero * inf -> NaN
        (0x01, 0x6C, float_to_hif8(hif8_to_float(0x01) * hif8_to_float(0x6C))),
        (0x33, 0x33, float_to_hif8(hif8_to_float(0x33) * hif8_to_float(0x33))),
    ],
)
def test_multiply_hif8(a, b, expected):
    assert multiply_hif8(a, b) == expected

# pytest ignore for no
@pytest.mark.parametrize(
    "builder,name",
    [
        #(build_hif8_mul_logic, "HiF8Mul_Logic"), # skip for now
        (build_hif8_mul_lut, "HiF8Mul_LUT"),
    ],
)
def test_hif8_module_matches_reference(builder, name):
    dut = builder(name)
    sim = Simulator(dut)
    samples = sorted(set(range(0, 256, 17)) | {255})
    for aval in range(256):
        for bval in samples:
            sim.set("a", aval).set("b", bval).eval()
            got = sim.get("y")
            exp = multiply_hif8(aval, bval)
            assert got == exp, f"a=0x{aval:02X} b=0x{bval:02X}"


def test_hif8_via_fp8_component_reasonable_accuracy():
    dut = build_hif8_mul_via_fp8_module()
    sim = Simulator(dut)

    for aval in range(0, 256, 7):
        for bval in range(0, 256, 13):
            sim.set("a", aval).set("b", bval).eval()
            got_bits = sim.get("y")
            ref_bits = multiply_hif8(aval, bval)

            got_val = hif8_to_float(got_bits)
            ref_val = hif8_to_float(ref_bits)

            if math.isnan(ref_val):
                if math.isnan(got_val):
                    continue
                assert got_val == pytest.approx(0.0, abs=1e-3)
                continue
            if math.isinf(ref_val):
                assert math.isinf(got_val)
                assert math.copysign(1.0, got_val) == math.copysign(1.0, ref_val)
                continue

            if math.isinf(got_val):
                assert abs(ref_val) >= 2 ** 14
                continue

            if ref_val == 0.0:
                assert got_val == pytest.approx(0.0, abs=1e-3)
            else:
                assert got_val == pytest.approx(ref_val, rel=1.0, abs=1e-3)