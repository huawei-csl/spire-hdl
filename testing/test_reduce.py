"""Tests for spire.reduce — balanced reduction trees."""

import functools
import itertools
import random

import pytest

from spire import Simulator
from spire.component import Netlist
from spire.expr import UInt, mux
from spire.reduce import (argmax_, argmin_, clamp_, max_, min_, prefix_scan,
                          prod_, reduce_tree, sum_)
from spire.selection_analysis import collect_chain


def _netlist(n, w):
    m = Netlist("R", with_clock=False, with_reset=False)
    xs = [m.input(UInt(w), f"x{i}") for i in range(n)]
    return m, xs


def _run(sim, vals):
    for i, v in enumerate(vals):
        sim.set(f"x{i}", v)
    sim.eval()


@pytest.mark.parametrize("n", [1, 2, 3, 5, 16, 20])
def test_max_min_equivalence(n):
    m, xs = _netlist(n, 8)
    mx = m.output(UInt(8), "mx")
    mn = m.output(UInt(8), "mn")
    mx <<= max_(xs)
    mn <<= min_(xs)
    sim = Simulator(m)
    rng = random.Random(1)
    for _ in range(200):
        vals = [rng.getrandbits(8) for _ in range(n)]
        _run(sim, vals)
        assert sim.get("mx") == max(vals)
        assert sim.get("mn") == min(vals)


def test_max_tree_is_log_depth():
    m, xs = _netlist(16, 8)
    y = m.output(UInt(8), "y")
    y <<= max_(xs)
    pairs, _ = collect_chain(y._driver)
    assert len(pairs) <= 5  # right spine of a 16-leaf tree, not a 15-deep chain


@pytest.mark.parametrize("topology", ["tree", "chain", "huffman"])
def test_sum_equivalence(topology):
    m, xs = _netlist(16, 8)
    y = m.output(UInt(12), "y")  # 16 x 8-bit fits in 12 bits
    y <<= sum_(xs, topology)[0:12]
    sim = Simulator(m)
    rng = random.Random(2)
    for _ in range(200):
        vals = [rng.getrandbits(8) for _ in range(16)]
        _run(sim, vals)
        assert sim.get("y") == sum(vals)


def test_prod_equivalence():
    m, xs = _netlist(4, 4)
    y = m.output(UInt(16), "y")
    y <<= prod_(xs)
    sim = Simulator(m)
    rng = random.Random(3)
    for _ in range(200):
        vals = [rng.getrandbits(4) for _ in range(4)]
        _run(sim, vals)
        assert sim.get("y") == vals[0] * vals[1] * vals[2] * vals[3]


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_argmax_argmin_leftmost_ties(n):
    m, xs = _netlist(n, 2)
    iw = max(1, (n - 1).bit_length())
    v, i = argmax_(xs)
    _, j = argmin_(xs)
    m.output(UInt(2), "v").assign(v)
    m.output(UInt(iw), "i").assign(i)
    m.output(UInt(iw), "j").assign(j)
    sim = Simulator(m)
    for vals in itertools.product(range(4), repeat=n):  # exhaustive incl. ties
        _run(sim, vals)
        assert sim.get("v") == max(vals)
        assert sim.get("i") == vals.index(max(vals))  # leftmost wins
        assert sim.get("j") == vals.index(min(vals))


def test_clamp():
    m, xs = _netlist(3, 3)
    y = m.output(UInt(3), "y")
    y <<= clamp_(xs[0], xs[1], xs[2])
    sim = Simulator(m)
    for x, lo, hi in itertools.product(range(8), repeat=3):
        if lo > hi:
            continue
        _run(sim, [x, lo, hi])
        assert sim.get("y") == min(max(x, lo), hi), (x, lo, hi)


def test_reduce_tree_generic_xor():
    m, xs = _netlist(16, 8)
    y = m.output(UInt(8), "y")
    y <<= reduce_tree(lambda a, b: a ^ b, xs)
    sim = Simulator(m)
    rng = random.Random(4)
    for _ in range(100):
        vals = [rng.getrandbits(8) for _ in range(16)]
        _run(sim, vals)
        assert sim.get("y") == functools.reduce(lambda a, b: a ^ b, vals)


@pytest.mark.parametrize("topology", ["tree", "chain", "huffman", "matrix"])
def test_max_min_topologies(topology):
    m, xs = _netlist(9, 8)  # odd count exercises leftover elements
    m.output(UInt(8), "mx").assign(max_(xs, topology))
    m.output(UInt(8), "mn").assign(min_(xs, topology))
    sim = Simulator(m)
    rng = random.Random(5)
    for _ in range(150):
        vals = [rng.getrandbits(8) for _ in range(9)]
        _run(sim, vals)
        assert sim.get("mx") == max(vals)
        assert sim.get("mn") == min(vals)


@pytest.mark.parametrize("topology", ["tree", "chain", "matrix"])
def test_argmax_topologies_leftmost_ties(topology):
    m, xs = _netlist(5, 2)
    v, i = argmax_(xs, topology)
    m.output(UInt(2), "v").assign(v)
    m.output(UInt(3), "i").assign(i)
    sim = Simulator(m)
    for vals in itertools.product(range(4), repeat=5):
        _run(sim, vals)
        assert sim.get("v") == max(vals)
        assert sim.get("i") == vals.index(max(vals)), (topology, vals)


def test_huffman_shallower_on_skewed_inputs():
    from spire.reduce import _expr_depth
    m, xs = _netlist(9, 8)
    deep = xs[0]
    for _ in range(8):
        deep = (deep + 1)[0:8]  # artificially deep input cone
    items = [deep] + xs[1:]
    t = max_(items, "tree")
    h = max_(items, "huffman")
    assert _expr_depth(h) <= _expr_depth(t)
    m.output(UInt(8), "yt").assign(t)
    m.output(UInt(8), "yh").assign(h)
    sim = Simulator(m)
    rng = random.Random(6)
    for _ in range(100):
        vals = [rng.getrandbits(8) for _ in range(9)]
        _run(sim, vals)
        expect = max([(vals[0] + 8) & 0xFF] + vals[1:])
        assert sim.get("yt") == sim.get("yh") == expect


@pytest.mark.parametrize("topology", ["sklansky", "brentkung", "koggestone"])
@pytest.mark.parametrize("n", [1, 5, 8])
def test_prefix_scan_running_max(topology, n):
    m, xs = _netlist(n, 8)
    for i, o in enumerate(prefix_scan(lambda a, b: mux(a >= b, a, b), xs, topology)):
        m.output(UInt(8), f"y{i}").assign(o)
    sim = Simulator(m)
    rng = random.Random(7)
    for _ in range(100):
        vals = [rng.getrandbits(8) for _ in range(n)]
        _run(sim, vals)
        for i, expect in enumerate(itertools.accumulate(vals, max)):
            assert sim.get(f"y{i}") == expect


@pytest.mark.parametrize("topology", ["sklansky", "brentkung", "koggestone"])
def test_prefix_scan_preserves_operand_order(topology):
    # fn(a, b) = b is associative but not commutative: prefix i must equal x_i,
    # so any swapped combine inside a scan variant shows up immediately
    m, xs = _netlist(6, 4)
    for i, o in enumerate(prefix_scan(lambda a, b: b, xs, topology)):
        m.output(UInt(4), f"y{i}").assign(o)
    sim = Simulator(m)
    vals = [3, 7, 1, 9, 12, 5]
    _run(sim, vals)
    for i in range(6):
        assert sim.get(f"y{i}") == vals[i]


def test_bad_topology_raises():
    xs = [1, 2]
    with pytest.raises(ValueError, match="topology"):
        reduce_tree(lambda a, b: a | b, xs, topology="magic")
    with pytest.raises(ValueError, match="topology"):
        prefix_scan(lambda a, b: a | b, xs, topology="magic")
    with pytest.raises(ValueError, match="huffman"):
        argmax_(xs, topology="huffman")


def test_empty_raises():
    with pytest.raises(ValueError):
        reduce_tree(lambda a, b: a | b, [])
    with pytest.raises(ValueError):
        argmax_([])
    with pytest.raises(ValueError):
        prefix_scan(lambda a, b: a | b, [])
