from spirehdl.spirehdl import Bool, UInt
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator  # your simulator with peek/watch

m = Module("VecBits", with_clock=False, with_reset=False)
v = m.input(UInt(8), "v")
low4 = m.output(UInt(4), "low4"); low4 <<= v[3:0]
sim = Simulator(m)

sim.set("v", 0b1010_0110).eval()

# Peek individual bits as expressions
print("v[0] =", sim.peek(v[0]))   # 0
print("v[5] =", sim.peek(v[5]))   # 1 --> only works when putting a breakpoint before this statement

# Inventory of names (inputs/outputs/wires/regs)
print("signals:", sim.list_signals())
