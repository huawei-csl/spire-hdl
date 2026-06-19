"""
Composing Components — build a bigger Component from smaller ones.

A top-level `Component` instantiates sub-`Component`s and flattens their logic into itself with
`.inline()`, producing one design — all in the Component world, never touching the IR. This is the
idiomatic way to build hierarchy in Spire.
"""

from spire import Component, IORecord, Input, Output, UInt, Simulator


class Adder(Component):
    """A simple width-parameterized adder."""

    def __init__(self, width: int = 8):
        self.width = width
        self.io = IORecord(
            a=Input(UInt(width)),
            b=Input(UInt(width)),
            sum=Output(UInt(width + 1)),
        )
        self.elaborate()

    def elaborate(self):
        self.io.sum <<= self.io.a + self.io.b


class Sum3(Component):
    """``result = (x + y) + z``, built from two `Adder` sub-components."""

    def __init__(self):
        self.io = IORecord(
            x=Input(UInt(8)),
            y=Input(UInt(8)),
            z=Input(UInt(8)),
            result=Output(UInt(10)),
        )
        self.elaborate()

    def elaborate(self):
        # inline() flattens each sub-component's logic into this component.
        add_xy = Adder(width=8).inline()
        add_xy.io.a <<= self.io.x
        add_xy.io.b <<= self.io.y

        add_xyz = Adder(width=9).inline()
        add_xyz.io.a <<= add_xy.io.sum
        add_xyz.io.b <<= self.io.z

        self.io.result <<= add_xyz.io.sum


if __name__ == "__main__":
    print("=" * 60)
    print("Composing Components: Sum3 = (x + y) + z from two Adders")
    print("=" * 60)
    print(Sum3().to_verilog(name="Sum3"))

    # Simulate the composed component directly — no IR in sight.
    sim = Simulator(Sum3())
    sim.set("x", 10)
    sim.set("y", 20)
    sim.set("z", 12)
    sim.eval()
    print("result(x=10, y=20, z=12) =", sim.peek_outputs()["result"], "(expected 42)")
