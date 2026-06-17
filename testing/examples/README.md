# SpireHDL Examples

This directory contains examples demonstrating various features of SpireHDL.

## Basic Examples

### component_example.py

A comprehensive guide showing how to:
- Define custom Components with IO ports
- Instantiate components
- Connect IO ports between components  
- Create hierarchical designs by instantiating sub-components
- Convert components to modules and generate Verilog
- Simulate component behavior

Run the example:
```bash
python examples/component_example.py
```

### simple_component.py

A minimal example showing the essential steps for creating and using a component:
- Defining a component with IO ports
- Implementing the elaborate() method
- Converting to a module
- Generating Verilog

Run the example:
```bash
python examples/simple_component.py
```

### module_with_component.py

Shows how to integrate Components within a Module-based design:
- Using components inside modules with `make_internal()`
- Converting components to separate module definitions
- Understanding the difference between flat and hierarchical designs

Run the example:
```bash
python examples/module_with_component.py
```

### direct_expression_basics.py

Minimal arithmetic expression example for newcomers:
- Uses direct expression building (`y = a + b`) with no intermediate wires
- Uses `+`, `-`, unary `-`, and `Const(..., Int(...))`
- Shows both `Const(False, Bool())` and plain `False`
- Includes a recursive Horner-form polynomial expression example
- No wires, no slicing, and no boolean operations
- Starts with `y = a + b` and then extends with constants/unary minus
- Prints expressions in Verilog form (`assign y = ...`)

Run the example:
```bash
PYTHONPATH=src python testing/examples/direct_expression_basics.py
```

## Key Concepts

### Defining IO Ports

IO ports are defined using Signal objects with specific attributes:
- `name`: The signal name
- `typ`: The type (UInt, SInt, Bool, etc.)
- `kind`: One of "input", "output", "wire", or "reg"

Example:
```python
from dataclasses import dataclass
from spirehdl.spirehdl import Signal, UInt

@dataclass
class IO:
    a: Signal
    b: Signal
    sum: Signal

self.io = IO(
    a=Signal(typ=UInt(8), kind="input"),
    b=Signal(typ=UInt(8), kind="input"),
    sum=Signal(typ=UInt(9), kind="output"),
)
```

### Connecting Signals

Use the `<<=` operator to connect signals:
```python
self.io.sum <<= self.io.a + self.io.b
```

### Hierarchical Design

To use a component as an internal building block:
1. Instantiate the sub-component
2. Call `.make_internal()` to convert its IO ports to internal wires
3. Connect the sub-component's IO to your component's signals

Example:
```python
sub_component = MyComponent(width=8).make_internal()
sub_component.io.input <<= self.io.my_input
self.io.my_output <<= sub_component.io.output
```

### Generating Verilog

Convert a component to a module and generate Verilog:
```python
component = MyComponent()
module = component.to_module(name="MyModule")
verilog_code = module.to_verilog()
print(verilog_code)
```
