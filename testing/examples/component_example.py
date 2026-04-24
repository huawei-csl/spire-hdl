"""
Component Example - Defining and Instantiating Components

This example demonstrates:
1. How to define a custom Component with IO ports
2. How to instantiate and use components
3. How to connect IO ports between components
4. How to convert to Module and generate Verilog
"""

import abc
from dataclasses import dataclass
from spirehdl.spirehdl_module import Component, Module
from spirehdl.spirehdl import Bool, UInt, Signal


# Example 1: Simple Adder Component
# ==================================
class Adder(Component):
    """A simple adder component that adds two numbers."""
    
    def __init__(self, width: int = 8):
        self.width = width
        
        # Define IO ports using a dataclass
        @dataclass
        class IO:
            a: Signal      # input a
            b: Signal      # input b
            sum: Signal    # output sum
        
        # Create the IO structure with Signal instances
        self.io = IO(
            a=Signal(name="a", typ=UInt(width), kind="input"),
            b=Signal(name="b", typ=UInt(width), kind="input"),
            sum=Signal(name="sum", typ=UInt(width + 1), kind="output"),
        )
        
        # Build the internal logic
        self.elaborate()
    
    def elaborate(self):
        """Define the component's behavior."""
        # Connect output to the sum of inputs
        self.io.sum <<= self.io.a + self.io.b


# Example 2: Comparator Component
# ================================
class Comparator(Component):
    """A comparator that checks if a > b."""
    
    def __init__(self, width: int = 8):
        self.width = width
        
        @dataclass
        class IO:
            a: Signal
            b: Signal
            greater: Signal
        
        self.io = IO(
            a=Signal(name="a", typ=UInt(width), kind="input"),
            b=Signal(name="b", typ=UInt(width), kind="input"),
            greater=Signal(name="greater", typ=Bool(), kind="output"),
        )
        
        self.elaborate()
    
    def elaborate(self):
        """Define the comparator logic."""
        self.io.greater <<= self.io.a > self.io.b


# Example 3: Hierarchical Component - Adder with Compare
# =======================================================
class AdderWithCompare(Component):
    """A hierarchical component that instantiates other components."""
    
    def __init__(self, width: int = 8):
        self.width = width
        
        @dataclass
        class IO:
            a: Signal
            b: Signal
            sum: Signal
            sum_greater_than_a: Signal
        
        self.io = IO(
            a=Signal(name="a", typ=UInt(width), kind="input"),
            b=Signal(name="b", typ=UInt(width), kind="input"),
            sum=Signal(name="sum", typ=UInt(width + 1), kind="output"),
            sum_greater_than_a=Signal(name="sum_greater_than_a", typ=Bool(), kind="output"),
        )
        
        self.elaborate()
    
    def elaborate(self):
        """Instantiate and connect sub-components."""
        # Instantiate an adder
        adder = Adder(width=self.width).make_internal()
        
        # Connect adder inputs to our inputs
        adder.io.a <<= self.io.a
        adder.io.b <<= self.io.b
        
        # Connect adder output to our output
        self.io.sum <<= adder.io.sum
        
        # Instantiate a comparator to check if sum > a
        comparator = Comparator(width=self.width + 1).make_internal()
        
        # Connect comparator inputs
        comparator.io.a <<= adder.io.sum
        comparator.io.b <<= self.io.a
        
        # Connect comparator output
        self.io.sum_greater_than_a <<= comparator.io.greater


# Example Usage and Testing
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Simple Adder Component")
    print("=" * 60)
    
    # Create an adder component
    adder = Adder(width=8)
    
    # Convert to module and generate Verilog
    adder_module = adder.to_module(name="Adder8bit")
    print(adder_module.to_verilog())
    print()
    
    print("=" * 60)
    print("Example 2: Comparator Component")
    print("=" * 60)
    
    # Create a comparator
    comparator = Comparator(width=8)
    
    # Convert to module and generate Verilog
    comparator_module = comparator.to_module(name="Comparator8bit")
    print(comparator_module.to_verilog())
    print()
    
    print("=" * 60)
    print("Example 3: Hierarchical Component")
    print("=" * 60)
    
    # Create the hierarchical component
    adder_compare = AdderWithCompare(width=8)
    
    # Convert to module and generate Verilog
    adder_compare_module = adder_compare.to_module(name="AdderWithCompare8bit")
    print(adder_compare_module.to_verilog())
    print()
    
    print("=" * 60)
    print("Example 4: Simulation")
    print("=" * 60)
    
    # Simulate the adder
    from spirehdl.spirehdl_simulator import Simulator
    
    # Create a fresh adder for simulation
    sim_adder = Adder(width=8)
    sim_adder_module = sim_adder.to_module(name="SimAdder")
    
    # Create simulator
    sim = Simulator(sim_adder_module)
    
    # Set inputs
    sim.set("a", 10)
    sim.set("b", 20)
    
    # Run simulation
    sim.eval()
    
    # Check outputs
    outputs = sim.peek_outputs()
    print(f"Input a=10, b=20")
    print(f"Output sum={outputs['sum']}")
    print(f"Expected: 30, Got: {outputs['sum']}")
    assert outputs['sum'] == 30, "Adder simulation failed!"
    print("✓ Simulation passed!")
    print()
