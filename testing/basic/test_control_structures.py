import pytest

from spire.expr import Bool, UInt
from spire.control_structures import case_, default, else_, elif_, if_, switch_
from spire.component import Module
from spire.simulator import Simulator, _sid


def simulate_outputs(module: Module, vectors: list[tuple[dict[str, int], dict[str, int]]]) -> None:
    sim = Simulator(module)
    for inputs, expected in vectors:
        for name, value in inputs.items():
            sim.set(name, value)
        sim.eval()
        for out_name, out_value in expected.items():
            assert sim.get(out_name) == out_value


def test_if_elif_else_priority():
    m = Module("IfElse", with_clock=False, with_reset=False)
    sel_a = m.input(Bool(), "sel_a")
    sel_b = m.input(Bool(), "sel_b")
    out = m.output(UInt(2), "out")

    out <<= 0
    with if_(sel_a):
        out <<= 1
    with elif_(sel_b):
        out <<= 2
    with else_():
        out <<= 3

    simulate_outputs(
        m,
        [
            ({"sel_a": 0, "sel_b": 0}, {"out": 3}),
            ({"sel_a": 0, "sel_b": 1}, {"out": 2}),
            ({"sel_a": 1, "sel_b": 0}, {"out": 1}),
            ({"sel_a": 1, "sel_b": 1}, {"out": 1}),
        ],
    )


def test_register_conditional_assignment():
    m = Module("RegIf", with_clock=True, with_reset=False)
    en = m.input(Bool(), "en")
    value = m.input(UInt(4), "value")
    reg = m.reg(UInt(4), "reg")
    reg.set_init(0)

    with if_(en):
        reg <<= value

    sim = Simulator(m)
    sim.eval()

    assert sim.get("reg") == 0

    sim.set("en", 1)
    sim.set("value", 5)
    next_state = sim._compute_next_state()
    assert next_state[_sid(reg)] == 5

    sim.set("reg", 5)
    sim.set("en", 0)
    sim.set("value", 9)
    next_state = sim._compute_next_state()
    assert next_state[_sid(reg)] == 5


def test_switch_cases():
    m = Module("Switch", with_clock=False, with_reset=False)
    sel = m.input(UInt(2), "sel")
    out = m.output(UInt(4), "out")

    out <<= 7
    with switch_(sel):
        with case_(0):
            out <<= 1
        with case_(1, 2):
            out <<= 2
        with default():
            out <<= 3

    simulate_outputs(
        m,
        [
            ({"sel": 0}, {"out": 1}),
            ({"sel": 1}, {"out": 2}),
            ({"sel": 2}, {"out": 2}),
            ({"sel": 3}, {"out": 3}),
        ],
    )


def test_nested_switch_without_binding():
    m = Module("NestedSwitch", with_clock=False, with_reset=False)
    outer_sel = m.input(UInt(2), "outer_sel")
    inner_sel = m.input(UInt(2), "inner_sel")
    out = m.output(UInt(4), "out")

    out <<= 9
    with switch_(outer_sel):
        with case_(0):
            with switch_(inner_sel):
                with case_(1):
                    out <<= 1
                with default():
                    out <<= 2
        with case_(1):
            out <<= 3
        with default():
            out <<= 4

    simulate_outputs(
        m,
        [
            ({"outer_sel": 0, "inner_sel": 1}, {"out": 1}),
            ({"outer_sel": 0, "inner_sel": 0}, {"out": 2}),
            ({"outer_sel": 1, "inner_sel": 2}, {"out": 3}),
            ({"outer_sel": 2, "inner_sel": 1}, {"out": 4}),
        ],
    )


def test_missing_default_driver_raises():
    m = Module("MissingDriver", with_clock=False, with_reset=False)
    cond = m.input(Bool(), "cond")
    out = m.output(UInt(2), "out")

    with pytest.raises(RuntimeError):
        with if_(cond):
            out <<= 1
