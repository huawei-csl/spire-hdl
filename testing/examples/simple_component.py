"""
Simple Component Example - Minimal Working Example

This is the simplest possible example showing how to:
1. Define a Component with IO ports
2. Implement the elaborate() method to define behavior
3. Convert to a Module
4. Generate Verilog
"""

from dataclasses import dataclass
from spirehdl.spirehdl_module import Component
from spirehdl.spirehdl import UInt, Signal


class SimpleAdder(Component):
    """A simple adder that adds two 8-bit numbers."""
    
    def __init__(self):
        # Step 1: Define IO structure using a dataclass
        @dataclass
        class IO:
            a: Signal      # input
            b: Signal      # input
            sum: Signal    # output
        
        # Step 2: Create Signal instances for each IO port
        self.io = IO(
            a=Signal(name="a", typ=UInt(8), kind="input"),
            b=Signal(name="b", typ=UInt(8), kind="input"),
            sum=Signal(name="sum", typ=UInt(9), kind="output"),  # 9 bits to hold sum
        )
        
        # Step 3: Build the component logic
        self.elaborate()
    
    def elaborate(self):
        """Define the component's behavior by connecting signals."""
        # Connect output to the sum of inputs
        self.io.sum <<= self.io.a + self.io.b


if __name__ == "__main__":
    # Step 4: Create an instance of the component
    adder = SimpleAdder()
    
    # Step 5: Convert to a Module
    module = adder.to_module(name="SimpleAdder")
    
    # Step 6: Generate and print Verilog
    verilog = module.to_verilog()
    print("Generated Verilog:")
    print("=" * 50)
    print(verilog)
