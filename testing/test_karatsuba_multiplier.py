"""Regression tests for KaratsubaMultiplier correctness.

The Karatsuba wrapper had two circuit-corrupting bugs that no test covered:
- odd operand widths silently truncated the (lo + hi) sub-multiplier input (8.1)
- deep recursion (karatsuba_only_at_first_level=False) returned over-wide partial
  products, misplacing p2's weight at every level (8.2)
"""
import pytest

from spire.arithmetic.int_multipliers.multipliers.multipliers_ext_karatsuba import KaratsubaMultiplier
from spire.simulator import Simulator


def _vectors(width):
    mx = (1 << width) - 1
    half = 1 << (width - 1)
    return [(mx, 1), (1, mx), (mx, mx), (0, mx), (3, 5), (mx >> 1, 2), (half, half - 1),
            (0x55 & mx, 0x33 & mx), (mx - 1, mx - 2)]


def _check(width, *, deep, compressor=False):
    m = KaratsubaMultiplier(a_w=width, b_w=width, karatsuba_only_at_first_level=not deep, use_compressor=compressor)
    sim = Simulator(m.to_netlist(name=f"kara_{width}_{'deep' if deep else 'single'}_{int(compressor)}"))
    for a, b in _vectors(width):
        sim.set(m.io.a, a)
        sim.set(m.io.b, b)
        sim.eval()
        got = sim.get(m.io.y)
        assert got == a * b, f"w={width} deep={deep}: {a}*{b} -> got {got:#x}, expected {a * b:#x}"


@pytest.mark.parametrize("width", [7, 9, 8, 16])
def test_karatsuba_single_level(width):
    _check(width, deep=False)


@pytest.mark.parametrize("width", [16, 17])
def test_karatsuba_deep_recursion(width):
    _check(width, deep=True)


@pytest.mark.parametrize("width,deep", [(8, False), (16, True)])
def test_karatsuba_compressor(width, deep):
    _check(width, deep=deep, compressor=True)
