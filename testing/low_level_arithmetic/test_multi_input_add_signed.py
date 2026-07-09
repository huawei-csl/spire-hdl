"""Signed multi-input add: the compressor tree must serve signed and mixed chains too.

build_multi_input_add used to fall back to chained `+` for any signed operand; now signed
operands enter the tree in sign-extension-compression form (inverted MSB plus one folded
constant row, the identity the fused inner product uses for its C term). Exhaustive checks
against Python integers, plus the end-to-end chain replacement through replace_arithmetic_ops.
"""
import pytest

from spire import Component, IORecord, Input, Output, SInt, Simulator, UInt
from spire.arithmetic.int_arithmetic_config import ArithmeticAutoConfig, build_multi_input_add, replace_arithmetic_ops
from spire.expr import Op2, reset_shared_cache


def _sim_mia(in_types, out_signed):
    """Component whose output is build_multi_input_add over its inputs; returns (sim, out_width)."""

    class Mia(Component):
        def __init__(self):
            fields = {f"i{k}": Input(t) for k, t in enumerate(in_types)}
            result = build_multi_input_add([sig for sig in fields.values()], ArithmeticAutoConfig())
            fields["y"] = Output(result.typ)
            self.io = IORecord(**fields)
            self._result = result
            self.elaborate()

        def elaborate(self):
            self.io.y <<= self._result

    comp = Mia()
    assert comp.io.y.typ.signed == out_signed
    return Simulator(comp.to_netlist("mia", with_clock=False, with_reset=False)), comp.io.y.typ.width


def _ranges(t):
    if t.signed:
        return range(-(1 << (t.width - 1)), 1 << (t.width - 1))
    return range(0, 1 << t.width)


@pytest.mark.parametrize("in_types", [
    [SInt(3), SInt(3), SInt(3)],                    # all signed
    [SInt(2), SInt(2), SInt(2), SInt(2)],           # all signed, four operands
    [SInt(3), UInt(3), SInt(3)],                    # mixed signedness
    [SInt(2), UInt(4), SInt(3)],                    # mixed signedness and widths
])
def test_signed_mia_exhaustive(in_types):
    reset_shared_cache()
    sim, _ = _sim_mia(in_types, out_signed=True)
    import itertools
    for vals in itertools.product(*[_ranges(t) for t in in_types]):
        for k, v in enumerate(vals):
            sim.set(f"i{k}", v)
        sim.eval()
        assert sim.get("y", signed=True) == sum(vals), f"{vals}"


def test_signed_mia_five_operands_spot():
    reset_shared_cache()
    types = [SInt(4)] * 5
    sim, _ = _sim_mia(types, out_signed=True)
    for vals in [(-8, -8, -8, -8, -8), (7, 7, 7, 7, 7), (-8, 7, -1, 0, 3), (5, -6, 7, -8, 1)]:
        for k, v in enumerate(vals):
            sim.set(f"i{k}", v)
        sim.eval()
        assert sim.get("y", signed=True) == sum(vals), f"{vals}"


def test_signed_mia_builds_a_tree_not_a_chain():
    """The whole point: signed operands must reach the compressor tree, not the chained `+`."""
    reset_shared_cache()
    ops = [Input(SInt(4), name=f"t{k}") for k in range(4)]
    result = build_multi_input_add(ops, ArithmeticAutoConfig())
    assert result.typ.signed and not isinstance(result, Op2), "signed operands fell back to chained +"


def test_unsigned_mia_unchanged():
    reset_shared_cache()
    sim, _ = _sim_mia([UInt(3)] * 3, out_signed=False)
    import itertools
    for vals in itertools.product(range(8), repeat=3):
        for k, v in enumerate(vals):
            sim.set(f"i{k}", v)
        sim.eval()
        assert sim.get("y") == sum(vals), f"{vals}"


def test_signed_add_chain_replaced_end_to_end():
    """a+b+c+d over SInt inputs: replace_arithmetic_ops must swap the chain for the tree
    and the netlist must still compute the exact sum."""

    class Chain(Component):
        def __init__(self):
            self.io = IORecord(a=Input(SInt(3)), b=Input(SInt(3)), c=Input(SInt(3)), d=Input(SInt(3)),
                               y=Output(SInt(6)))
            self.elaborate()

        def elaborate(self):
            self.io.y <<= self.io.a + self.io.b + self.io.c + self.io.d

    reset_shared_cache()
    comp = Chain()
    replace_arithmetic_ops(comp, ArithmeticAutoConfig())
    sim = Simulator(comp.to_netlist("chain", with_clock=False, with_reset=False))
    import itertools
    for vals in itertools.product(range(-4, 4), repeat=4):
        sim.set("a", vals[0]).set("b", vals[1]).set("c", vals[2]).set("d", vals[3])
        sim.eval()
        assert sim.get("y", signed=True) == sum(vals), f"{vals}"


def test_mia_db_carries_rows_for_both_signs():
    """The shipped DB must serve signed and unsigned rows across the whole swept grid; without
    signed rows every signed or mixed chain would silently ride the stale-DB fallback."""
    from spire.arithmetic.eval.auto_config import lookup_best_mia_config
    for n in (3, 4, 5, 8):
        for w in (2, 4, 8, 16, 32):
            for signed in (False, True):
                assert lookup_best_mia_config(n_inputs=n, n_bits=w, signed=signed) is not None, (n, w, signed)
