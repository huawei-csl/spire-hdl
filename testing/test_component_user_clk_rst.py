"""Clock/reset are framework-provided: IO leaves named clk/rst are rejected, the with_* flags are the mechanism."""
import pytest

from spire import Component, IORecord, Input, Output
from spire.expr import Register, UInt, reset_shared_cache
from spire.simulator import Simulator


class _Counter(Component):
    def __init__(self):
        self.io = IORecord(en=Input(UInt(1)), q=Output(UInt(4)))
        self.elaborate()

    def elaborate(self):
        r = Register(UInt(4), init=0)
        r <<= r + self.io.en
        self.io.q <<= r


def test_clocked_component_via_flags():
    reset_shared_cache()
    m = _Counter().to_netlist(with_clock=True, with_reset=True)
    text = m.to_verilog()
    assert "posedge clk" in text

    sim = Simulator(m)
    sim.set("en", 1)
    sim.reset(True)
    sim.deassert_reset()
    assert sim.get("q") == 0
    sim.step()
    assert sim.get("q") == 1


@pytest.mark.parametrize("leaf", ["clk", "rst"])
def test_io_leaf_named_clk_rst_rejected(leaf):
    reset_shared_cache()

    class _Bad(Component):
        def __init__(self):
            self.io = IORecord(**{leaf: Input(UInt(1))}, q=Output(UInt(1)))
            self.elaborate()

        def elaborate(self):
            self.io.q <<= 0

    with pytest.raises(ValueError, match="framework|with_clock"):
        _Bad().to_netlist()


def test_dict_io_keeps_field_names():
    reset_shared_cache()

    class _DictIO(Component):
        def __init__(self):
            self.io = {"foo": Input(UInt(4)), "bar": Output(UInt(4))}
            self.elaborate()

        def elaborate(self):
            self.io["bar"] <<= self.io["foo"]

    m = _DictIO().to_netlist()
    assert [p.name for p in m._ports] == ["foo", "bar"]
    # and the generated record class must not carry a Signal-comparing dataclass __eq__
    rec = _DictIO().get_ios()
    assert "__eq__" not in type(rec).__dict__
