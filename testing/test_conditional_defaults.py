"""A combinational signal assigned only inside if_/switch_ blocks needs no prior driver: it is 0 outside the
covered arms, an explicit prior assignment still wins, and registers keep holding their value."""
from spire import Component, IORecord, Input, Output, Register, Simulator, UInt
from spire.control_structures import case_, default, else_, if_, switch_


class IfElse(Component):
    def __init__(self):
        self.io = IORecord(sel=Input(UInt(1)), a=Input(UInt(4)), b=Input(UInt(4)), y=Output(UInt(4)))
        self.elaborate()

    def elaborate(self):
        with if_(self.io.sel):
            self.io.y <<= self.io.a
        with else_():
            self.io.y <<= self.io.b


class IfOnly(Component):
    def __init__(self):
        self.io = IORecord(sel=Input(UInt(1)), a=Input(UInt(4)), y=Output(UInt(4)), z=Output(UInt(4)))
        self.elaborate()

    def elaborate(self):
        self.io.z <<= 7                         # explicit prior driver: the fallback when sel is 0
        with if_(self.io.sel):
            self.io.y <<= self.io.a             # no prior driver: 0 when sel is 0
            self.io.z <<= self.io.a


class Switch(Component):
    def __init__(self):
        self.io = IORecord(op=Input(UInt(2)), a=Input(UInt(4)), b=Input(UInt(4)), y=Output(UInt(4)),
                           n=Output(UInt(4)))
        self.elaborate()

    def elaborate(self):
        with switch_(self.io.op):
            with case_(0):
                self.io.y <<= self.io.a
                self.io.n <<= self.io.a
            with case_(1):
                self.io.y <<= self.io.b
                self.io.n <<= self.io.b
            with default():
                self.io.y <<= 0xA                # n has no default arm: 0 for op 2 and 3


class Hold(Component):
    def __init__(self):
        self.io = IORecord(en=Input(UInt(1)), d=Input(UInt(4)), q=Output(UInt(4)))
        self.elaborate()

    def elaborate(self):
        r = Register(UInt(4), init=0)
        with if_(self.io.en):
            r <<= self.io.d                     # a register without a prior driver holds, as before
        self.io.q <<= r


def test_if_else_without_prior_driver():
    sim = Simulator(IfElse())
    sim.set("a", 5).set("b", 9)
    assert sim.set("sel", 1).step().get("y") == 5
    assert sim.set("sel", 0).step().get("y") == 9


def test_if_without_else_falls_back_to_zero_unless_driven_before():
    sim = Simulator(IfOnly())
    sim.set("a", 5)
    assert (sim.set("sel", 1).step().get("y"), sim.get("z")) == (5, 5)
    assert (sim.set("sel", 0).step().get("y"), sim.get("z")) == (0, 7)


def test_switch_without_prior_driver():
    sim = Simulator(Switch())
    sim.set("a", 3).set("b", 6)
    assert [(sim.set("op", op).step().get("y"), sim.get("n")) for op in range(4)] == \
        [(3, 3), (6, 6), (0xA, 0), (0xA, 0)]


def test_register_without_prior_driver_still_holds():
    sim = Simulator(Hold())
    sim.deassert_reset()
    assert sim.set("en", 1).set("d", 4).step().get("q") == 4
    assert sim.set("en", 0).set("d", 9).step().get("q") == 4
