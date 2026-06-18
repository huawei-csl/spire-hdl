"""
Component Example - Defining and Instantiating Components

This example demonstrates:
1. How to define a custom Component with IO ports (IORecord + Input/Output)
2. How to instantiate and use components
3. How to compose components hierarchically with inline()
4. How to generate Verilog and simulate directly from a Component
"""

from spire import Component, IORecord, Input, Output, Bool, UInt, Simulator


# Example 1: Simple Adder Component
# ==================================
class Adder(Component):
    """A simple adder component that adds two numbers."""

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


# Example 2: Comparator Component
# ================================
class Comparator(Component):
    """A comparator that checks if a > b."""

    def __init__(self, width: int = 8):
        self.width = width
        self.io = IORecord(
            a=Input(UInt(width)),
            b=Input(UInt(width)),
            greater=Output(Bool()),
        )
        self.elaborate()

    def elaborate(self):
        self.io.greater <<= self.io.a > self.io.b


# Example 3: Hierarchical Component - Adder with Compare
# =======================================================
class AdderWithCompare(Component):
    """A hierarchical component that instantiates other components."""

    def __init__(self, width: int = 8):
        self.width = width
        self.io = IORecord(
            a=Input(UInt(width)),
            b=Input(UInt(width)),
            sum=Output(UInt(width + 1)),
            sum_greater_than_a=Output(Bool()),
        )
        self.elaborate()

    def elaborate(self):
        # Instantiate an adder and inline it (flatten its logic into this component).
        adder = Adder(width=self.width).inline()
        adder.io.a <<= self.io.a
        adder.io.b <<= self.io.b
        self.io.sum <<= adder.io.sum

        # Instantiate a comparator to check if sum > a.
        comparator = Comparator(width=self.width + 1).inline()
        comparator.io.a <<= adder.io.sum
        comparator.io.b <<= self.io.a
        self.io.sum_greater_than_a <<= comparator.io.greater


# Example Usage and Testing
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Simple Adder Component")
    print("=" * 60)
    adder = Adder(width=8)
    print(adder.to_verilog(name="Adder8bit"))
    print()

    print("=" * 60)
    print("Example 2: Comparator Component")
    print("=" * 60)
    comparator = Comparator(width=8)
    print(comparator.to_verilog(name="Comparator8bit"))
    print()

    print("=" * 60)
    print("Example 3: Hierarchical Component")
    print("=" * 60)
    adder_compare = AdderWithCompare(width=8)
    print(adder_compare.to_verilog(name="AdderWithCompare8bit"))
    print()

    print("=" * 60)
    print("Example 4: Simulation")
    print("=" * 60)
    # Simulate the adder directly — Simulator lowers the Component internally.
    sim = Simulator(Adder(width=8))
    sim.set("a", 10)
    sim.set("b", 20)
    sim.eval()

    outputs = sim.peek_outputs()
    print("Input a=10, b=20")
    print(f"Output sum={outputs['sum']}")
    assert outputs["sum"] == 30, "Adder simulation failed!"
    print("✓ Simulation passed!")
    print()
