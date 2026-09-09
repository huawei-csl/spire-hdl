"""``pipeline(value, cycles)``: registered delay lines for leaves and composites.

Covers the delay itself (scalar / aggregate / expression inputs), the stage options
(``init``, ``enable``, ``name``), type and structure preservation, and the error paths.
"""
import pytest

from spire import Component, IORecord, Input, Output, Wire, pipeline
from spire.composite.array import Array
from spire.composite.fixed_point import FixedPoint, FixedPointType
from spire.composite.record import CompositeRecord
from spire.composite.register import CompositeRegister
from spire.expr import Bool, SInt, UInt, cat, reset_shared_cache
from spire.interfaces.stream import Stream
from spire.ir import Netlist
from spire.simulator import Simulator

Q8_8 = FixedPointType(width_total=16, width_frac=8, signed=True)


def _sim(m: Netlist) -> Simulator:
    sim = Simulator(m)
    sim.reset(True)
    sim.deassert_reset()
    return sim


# ---------------------------------------------------------------------------
# Scalar signals
# ---------------------------------------------------------------------------

def test_default_is_one_cycle():
    reset_shared_cache()
    m = Netlist("pipe1", with_clock=True)
    x = m.input(UInt(8), "x")
    y = m.output(UInt(8), "y")
    y <<= pipeline(x)

    assert [s.name for s in m._signals if s.kind == "reg"] == []  # not collected yet
    text = m.to_verilog()
    assert "reg [7:0] x_d1;" in text  # stage named after its source signal
    assert "x_d1 <= x;" in text

    sim = _sim(m)
    for sent, seen in [(5, 0), (6, 5), (7, 6), (8, 7)]:
        sim.set("x", sent)
        sim.eval()
        assert sim.get("y") == seen
        sim.step()


def test_multiple_cycles_chain():
    reset_shared_cache()
    m = Netlist("pipe3", with_clock=True)
    x = m.input(UInt(8), "x")
    y = m.output(UInt(8), "y")
    y <<= pipeline(x, 3)

    text = m.to_verilog()
    assert [f"reg [7:0] x_d{i};" in text for i in (1, 2, 3)] == [True] * 3
    assert "x_d1 <= x;" in text and "x_d2 <= x_d1;" in text and "x_d3 <= x_d2;" in text

    sim = _sim(m)
    stream = [1, 2, 3, 4, 5, 6]
    for i, sent in enumerate(stream):
        sim.set("x", sent)
        sim.eval()
        assert sim.get("y") == (stream[i - 3] if i >= 3 else 0)
        sim.step()


def test_cycles_zero_is_identity():
    reset_shared_cache()
    m = Netlist("pipe0", with_clock=True)
    x = m.input(UInt(8), "x")
    assert pipeline(x, 0) is x
    assert "reg" not in m.to_verilog().split("// Registers")[1].split("//")[0]


def test_expression_input_is_registered():
    reset_shared_cache()
    m = Netlist("pipe_expr", with_clock=True)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(9), "y")
    y <<= pipeline(a + b, 2)

    sim = _sim(m)
    sim.set("a", 20)
    sim.set("b", 30)
    sim.eval()
    assert sim.get("y") == 0
    sim.step()
    sim.step()
    sim.eval()
    assert sim.get("y") == 50  # full 9-bit sum, not truncated


def test_signed_type_is_preserved():
    reset_shared_cache()
    m = Netlist("pipe_signed", with_clock=True)
    a = m.input(SInt(8), "a")
    y = m.output(SInt(8), "y")
    d = pipeline(a, 2)
    assert d.typ.signed and d.typ.width == 8
    y <<= d
    assert "reg signed [7:0] a_d2;" in m.to_verilog()

    sim = _sim(m)
    sim.set("a", -7)
    sim.step()
    sim.step()
    sim.eval()
    assert sim.get("y") == -7


def test_init_resets_every_stage():
    reset_shared_cache()
    m = Netlist("pipe_init", with_clock=True, with_reset=True)
    x = m.input(UInt(8), "x")
    y = m.output(UInt(8), "y")
    y <<= pipeline(x, 2, init=9)

    text = m.to_verilog()
    assert text.count("<= 8'd9;") == 2  # both stages carry the reset arm

    sim = _sim(m)
    sim.set("x", 4)
    sim.eval()
    assert sim.get("y") == 9
    sim.step()
    assert sim.get("y") == 9
    sim.step()
    assert sim.get("y") == 4


def test_enable_holds_the_line():
    reset_shared_cache()
    m = Netlist("pipe_en", with_clock=True)
    x = m.input(UInt(8), "x")
    en = m.input(Bool(), "en")
    y = m.output(UInt(8), "y")
    y <<= pipeline(x, 2, enable=en)

    sim = _sim(m)
    sim.set("en", 1)
    sim.set("x", 3)
    sim.step()  # stage1 = 3
    sim.set("x", 4)
    sim.step()  # stage1 = 4, stage2 = 3
    sim.eval()
    assert sim.get("y") == 3

    sim.set("en", 0)  # stalled: further edges must not move the data
    sim.set("x", 5)
    sim.step()
    sim.step()
    sim.eval()
    assert sim.get("y") == 3

    sim.set("en", 1)
    sim.step()
    sim.eval()
    assert sim.get("y") == 4


def test_name_overrides_the_inferred_base():
    reset_shared_cache()
    m = Netlist("pipe_name", with_clock=True)
    x = m.input(UInt(4), "x")
    y = m.output(UInt(4), "y")
    y <<= pipeline(x, 2, name="stage")
    text = m.to_verilog()
    assert "reg [3:0] stage_d1;" in text and "reg [3:0] stage_d2;" in text


# ---------------------------------------------------------------------------
# Composites
# ---------------------------------------------------------------------------

class _Packet(CompositeRecord):
    """Record with a constructor argument and a non-HDL attribute."""

    def __init__(self, width: int = 8):
        self.width_arg = width
        super().__init__(flag=Wire(Bool()), data=Wire(UInt(width)))


def test_array_pipeline_keeps_structure_and_delays():
    reset_shared_cache()

    class Dut(Component):
        def __init__(self):
            self.io = IORecord(a=Input(UInt(8)), b=Input(UInt(8)), y=Output(UInt(16)))
            self.elaborate()

        def elaborate(self):
            vec = Array([self.io.a, self.io.b])
            vec_d = pipeline(vec, 2, name="vec")
            assert isinstance(vec_d, Array) and len(vec_d) == 2
            assert vec_d[0] is not self.io.a  # a fresh view, not the input
            self.io.y <<= cat(vec_d[0], vec_d[1])

    m = Dut().to_netlist("arr", with_clock=True)
    text = m.to_verilog()
    assert "reg [15:0] vec_d1;" in text  # one packed register per stage
    assert "reg [15:0] vec_d2;" in text
    assert "wire [7:0] vec_d2_0;" in text and "wire [7:0] vec_d2_1;" in text

    sim = _sim(m)
    sim.set("a", 0x11)
    sim.set("b", 0x22)
    sim.step()
    sim.set("a", 0x33)
    sim.set("b", 0x44)
    sim.step()
    sim.eval()
    assert sim.peek("y") == 0x2211
    sim.step()
    sim.eval()
    assert sim.peek("y") == 0x4433


def test_record_pipeline_exposes_fields_and_extra_attrs():
    reset_shared_cache()

    class Dut(Component):
        def __init__(self):
            self.io = IORecord(v=Input(Bool()), d=Input(UInt(8)), y=Output(UInt(9)))
            self.elaborate()

        def elaborate(self):
            pkt = _Packet(8)
            pkt.flag <<= self.io.v
            pkt.data <<= self.io.d
            pkt_d = pipeline(pkt, name="pkt")
            assert isinstance(pkt_d, _Packet)
            assert pkt_d.width_arg == 8          # non-HDL attributes survive the clone
            assert pkt_d.data is not pkt.data    # fresh leaves
            self.io.y <<= cat(pkt_d.data, pkt_d.flag)

    m = Dut().to_netlist("rec", with_clock=True)
    text = m.to_verilog()
    assert "reg [8:0] pkt_d1;" in text
    assert "wire  pkt_d1_flag;" in text and "wire [7:0] pkt_d1_data;" in text

    sim = _sim(m)
    sim.set("v", 1)
    sim.set("d", 0x5A)
    sim.step()
    sim.eval()
    assert sim.peek("y") == (1 << 8) | 0x5A  # cat() is LSB-first: data low, flag high


def test_interface_pipeline_keeps_class_and_methods():
    reset_shared_cache()

    class Dut(Component):
        def __init__(self):
            self.io = IORecord(d=Input(UInt(8)), y=Output(UInt(8)))
            self.elaborate()

        def elaborate(self):
            st = Stream(UInt(8))
            st.valid <<= 1
            st.ready <<= 1
            st.data <<= self.io.d
            st_d = pipeline(st)
            assert isinstance(st_d, Stream)
            assert st_d.fire().typ.width == 1  # interface behaviour still available
            self.io.y <<= st_d.data

    sim = _sim(Dut().to_netlist("iface", with_clock=True))
    sim.set("d", 0x77)
    sim.step()
    sim.eval()
    assert sim.peek("y") == 0x77


def test_fixed_point_pipeline_round_trips_value():
    reset_shared_cache()

    class Dut(Component):
        def __init__(self):
            self.io = IORecord(x=Input(SInt(16)), y=Output(SInt(16)))
            self.elaborate()

        def elaborate(self):
            fp = FixedPoint(Q8_8, bits=self.io.x)
            fp_d = pipeline(fp, 2, name="fp")
            assert isinstance(fp_d, FixedPoint) and fp_d.ftype == Q8_8
            self.io.y <<= fp_d.bits

    m = Dut().to_netlist("fp", with_clock=True)
    text = m.to_verilog()
    assert "reg signed [15:0] fp_d1;" in text  # one leaf: the reg keeps its type
    assert "wire signed [15:0] fp_d2_q;" in text  # single-leaf view is <stage>_q

    sim = _sim(m)
    sim.set("x", -256)  # -1.0 in Q8.8
    sim.step()
    sim.step()
    sim.eval()
    assert sim.get("y") == -256


def test_nested_composite_leaf_names_follow_the_path():
    reset_shared_cache()

    class Dut(Component):
        def __init__(self):
            self.io = IORecord(a=Input(UInt(4)), b=Input(UInt(4)), y=Output(UInt(16)))
            self.elaborate()

        def elaborate(self):
            nest = Array([Array([self.io.a, self.io.b]), Array([self.io.b, self.io.a])])
            nest_d = pipeline(nest, name="nest")
            assert [leaf.name for leaf in nest_d.to_list()] == [
                "nest_d1_0_0", "nest_d1_0_1", "nest_d1_1_0", "nest_d1_1_1"]
            self.io.y <<= nest_d.to_bits()

    sim = _sim(Dut().to_netlist("nest", with_clock=True))
    sim.set("a", 1)
    sim.set("b", 2)
    sim.step()
    sim.eval()
    assert sim.peek("y") == 0x1221


def test_composite_cycles_zero_is_identity():
    reset_shared_cache()
    vec = Array([Wire(UInt(4)), Wire(UInt(4))])
    assert pipeline(vec, 0) is vec


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_negative_cycles_rejected():
    reset_shared_cache()
    with pytest.raises(ValueError, match="cycles must be >= 0"):
        pipeline(Wire(UInt(4)), -1)


@pytest.mark.parametrize("cycles", [1.0, "2", True])
def test_non_int_cycles_rejected(cycles):
    reset_shared_cache()
    with pytest.raises(TypeError, match="cycles must be an int"):
        pipeline(Wire(UInt(4)), cycles)


def test_dynamic_init_rejected():
    reset_shared_cache()
    x = Wire(UInt(4))
    with pytest.raises(ValueError, match="init must be a constant"):
        pipeline(x, 1, init=x + 1)


def test_composite_register_is_delayed_as_its_value_type():
    reset_shared_cache()

    class Dut(Component):
        def __init__(self):
            self.io = IORecord(x=Input(SInt(16)), y=Output(SInt(16)))
            self.elaborate()

        def elaborate(self):
            acc = CompositeRegister(FixedPoint, Q8_8, name="acc", init=0)
            acc <<= self.io.x
            acc_d = pipeline(acc, 2)              # the register, not its .value
            assert isinstance(acc_d, FixedPoint)  # comes back as what the register holds
            assert acc_d.ftype == Q8_8
            self.io.y <<= acc_d.bits

    m = Dut().to_netlist("creg", with_clock=True, with_reset=True)
    text = m.to_verilog()
    assert "reg signed [15:0] acc_d1;" in text  # stages named after the register
    assert "wire signed [15:0] acc_d2_q;" in text

    sim = _sim(m)
    sim.set("x", -512)
    sim.step()  # acc <- x
    sim.step()  # acc_d1 <- acc
    sim.step()  # acc_d2 <- acc_d1
    sim.eval()
    assert sim.get("y") == -512
