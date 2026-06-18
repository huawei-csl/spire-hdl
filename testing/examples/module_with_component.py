"""
Netlist with Component Example (power-user / IR level)

Most designs only need `Component` (see simple_component.py / component_example.py).
This example shows the lower-level IR, `Netlist` (the flat, lowered netlist that every
backend consumes — formerly named `Module`), and how to inline `Component` logic into a
hand-built netlist. `Netlist` lives in `spire.ir`; the quick start never needs it.
"""

from spire import Component, IORecord, Input, Output, UInt
from spire.ir import Netlist


# Define a reusable Adder Component
class Adder(Component):
    """A simple adder component."""

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


def create_netlist_with_component():
    """Build a flat Netlist and inline two Adder components into it."""
    m = Netlist("TopLevel", with_clock=False, with_reset=False)

    x = m.input(UInt(8), "x")
    y = m.input(UInt(8), "y")
    z = m.input(UInt(8), "z")
    result = m.output(UInt(10), "result")

    # inline() flattens the component's logic into the surrounding netlist
    # (its IO ports become internal wires).
    adder1 = Adder(width=8).inline()
    adder1.io.a <<= x
    adder1.io.b <<= y

    adder2 = Adder(width=9).inline()
    adder2.io.a <<= adder1.io.sum
    adder2.io.b <<= z

    result <<= adder2.io.sum
    return m


if __name__ == "__main__":
    print("=" * 60)
    print("Netlist with inlined Components")
    print("=" * 60)
    print("A single flat netlist with the component logic inlined.\n")

    netlist = create_netlist_with_component()
    print(netlist.to_verilog())
    print()

    print("=" * 60)
    print("Key Points")
    print("=" * 60)
    print(
        """
1. Prefer `Component` for authoring (see simple_component.py). `Netlist` is the IR.
2. Use .inline() to flatten a Component's logic into a netlist
   - Result: a single flat netlist with all logic.
3. For hierarchy, compose Components within Components
   - See component_example.py for details.
"""
    )
