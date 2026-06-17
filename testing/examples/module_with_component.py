"""
Module with Component Example

This example demonstrates how to use a Component inside a Module,
showing the integration between Module-based and Component-based design.
"""

from dataclasses import dataclass
from spirehdl.spirehdl_module import Module, Component
from spirehdl.spirehdl import UInt, Signal


# Define a reusable Adder Component
class Adder(Component):
    """A simple adder component."""
    
    def __init__(self, width: int = 8):
        self.width = width
        
        @dataclass
        class IO:
            a: Signal
            b: Signal
            sum: Signal
        
        self.io = IO(
            a=Signal(typ=UInt(width), kind="input"),
            b=Signal(typ=UInt(width), kind="input"),
            sum=Signal(typ=UInt(width + 1), kind="output"),
        )
        
        self.elaborate()
    
    def elaborate(self):
        self.io.sum <<= self.io.a + self.io.b


def create_module_with_component():
    """Create a Module that uses a Component."""
    
    # Create a Module
    m = Module("TopLevel", with_clock=False, with_reset=False)
    
    # Add module inputs
    x = m.input(UInt(8), "x")
    y = m.input(UInt(8), "y")
    z = m.input(UInt(8), "z")
    
    # Add module output
    result = m.output(UInt(10), "result")
    
    # Create an Adder component and make it internal
    # (internal means its IO ports become internal wires in the module)
    adder1 = Adder(width=8).make_internal()
    
    # Connect the first adder
    adder1.io.a <<= x
    adder1.io.b <<= y
    
    # Create a second adder to add the result of the first adder with z
    adder2 = Adder(width=9).make_internal()
    
    # Connect the second adder
    adder2.io.a <<= adder1.io.sum
    adder2.io.b <<= z
    
    # Connect to module output
    result <<= adder2.io.sum
    
    return m


def create_module_with_component_alternative():
    """Alternative approach: Convert component to module and wire it up."""
    
    # Create a Module
    m = Module("TopLevelAlt", with_clock=False, with_reset=False)
    
    # Add module inputs
    x = m.input(UInt(8), "x")
    y = m.input(UInt(8), "y")
    
    # Add module output
    result = m.output(UInt(9), "result")
    
    # Create an Adder component
    adder = Adder(width=8)
    
    # Convert component to module (this creates a separate module definition)
    adder_module = adder.to_module(name="AdderSubModule")
    
    # Note: This approach creates a module definition but doesn't instantiate it
    # within the parent module. For true hierarchical instantiation, use the
    # make_internal() approach shown in create_module_with_component()
    
    # Instead, we'll directly wire the logic
    result <<= x + y
    
    return m, adder_module


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Module with Components (Recommended)")
    print("=" * 60)
    print("This creates a single flat module with the component logic inline.\n")
    
    module = create_module_with_component()
    print(module.to_verilog())
    print()
    
    print("=" * 60)
    print("Example 2: Component to Separate Module")
    print("=" * 60)
    print("This creates separate module definitions.\n")
    
    main_module, sub_module = create_module_with_component_alternative()
    
    print("Main Module:")
    print(main_module.to_verilog())
    print()
    
    print("Sub-module (Component as Module):")
    print(sub_module.to_verilog())
    print()
    
    print("=" * 60)
    print("Key Points")
    print("=" * 60)
    print("""
1. Use .make_internal() to inline component logic into a module
   - Best for: Creating complex logic from reusable components
   - Result: Single flat module with all logic
   
2. Use .to_module() to convert a component to a module definition
   - Best for: Generating separate module definitions
   - Result: Multiple module definitions (currently not hierarchical)
   
3. For hierarchical designs, compose components within components
   - See component_example.py for details
""")
