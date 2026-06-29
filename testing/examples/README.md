# Spire Examples

This directory contains examples demonstrating various features of Spire.

## Basic Examples

### component_example.py

A comprehensive guide showing how to:
- Define custom Components with IO ports
- Instantiate components
- Connect IO ports between components  
- Create hierarchical designs by instantiating sub-components
- Lower components to a netlist and generate Verilog
- Simulate component behavior

Run the example:
```bash
python examples/component_example.py
```

### simple_component.py

A minimal example showing the essential steps for creating and using a component:
- Defining a component with IO ports
- Implementing the elaborate() method
- Lowering to a netlist
- Generating Verilog

Run the example:
```bash
python examples/simple_component.py
```

### composing_components.py

Shows how to compose a larger Component from smaller ones — all in the
Component world, never touching the IR:
- Instantiating sub-Components and wiring their IO
- Building hierarchy without leaving Python
- One flat design emitted from several reusable blocks

Run the example:
```bash
python examples/composing_components.py
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

Declare IO with `IORecord` plus `Input`/`Output`: the field names become the
signal names, and `Input`/`Output` set the direction and bit-precise type
(`UInt`, `SInt`, `Bool`, …).

Example:
```python
from spire import IORecord, Input, Output, UInt

self.io = IORecord(
    a=Input(UInt(8)),
    b=Input(UInt(8)),
    sum=Output(UInt(9)),
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
2. Connect the sub-component's IO to your component's signals
3. Emission embeds the reached sub-component logic automatically

Example:
```python
sub_component = MyComponent(width=8)
sub_component.io.input <<= self.io.my_input
self.io.my_output <<= sub_component.io.output
```

### Generating Verilog

Generate Verilog straight from a component:
```python
component = MyComponent()
print(component.to_verilog(name="MyModule"))
```
