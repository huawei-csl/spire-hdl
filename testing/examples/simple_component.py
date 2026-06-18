"""
Simple Component Example - Minimal Working Example

This is the simplest possible example showing how to:
1. Define a Component with IO ports (IORecord + Input/Output)
2. Implement the elaborate() method to define behavior
3. Generate Verilog directly from the Component
"""

from spirehdl import Component, IORecord, Input, Output, UInt


class SimpleAdder(Component):
    """A simple adder that adds two 8-bit numbers."""

    def __init__(self):
        # IO: field names become signal names; direction is explicit (Input/Output).
        self.io = IORecord(
            a=Input(UInt(8)),
            b=Input(UInt(8)),
            sum=Output(UInt(9)),  # 9 bits to hold the carry-out
        )
        self.elaborate()

    def elaborate(self):
        """Define the component's behavior by connecting signals."""
        self.io.sum <<= self.io.a + self.io.b


if __name__ == "__main__":
    adder = SimpleAdder()

    # Generate Verilog straight from the Component — no need to touch the IR.
    print("Generated Verilog:")
    print("=" * 50)
    print(adder.to_verilog(name="SimpleAdder"))
