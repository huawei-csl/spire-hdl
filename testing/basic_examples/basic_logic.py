from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl import Bool, UInt, SInt, mux, cat

# declare Moudule
m = Module("LogicDemo", with_clock=False, with_reset=False)

# declare inputs
x = m.input(UInt(8), "x")
y = m.input(UInt(8), "y")
sg = m.input(SInt(8), "sg")

# declare outputs
f = m.input(Bool(), "f")
z = m.output(UInt(9), "z")  # 8+8 -> 9 bits
eq = m.output(Bool(), "eq")
hi = m.output(UInt(4), "hi")

sum_ = x + y  # sum is 9-bit
z <<= sum_  # auto-fit/truncate handled
eq <<= x == y  # Bool
hi <<= cat(y[6:8], x[6:8])  # concat 2+2 = 4 bits

# Mux on Bool
w = m.output(UInt(8), "w")
w <<= mux(f, x & y, x | y)

print(m.to_verilog())
