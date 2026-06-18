from spire.component import Module
from spire.expr import UInt

m = Module("MulAddComb", with_clock=False, with_reset=False)
a = m.input(UInt(16), "a")
b = m.input(UInt(16), "b")
c = m.input(UInt(32), "c")
y = m.output(UInt(32), "y")

prod = m.wire(UInt(32), "prod")
prod <<= a * b  # 16x16 -> 32
y <<= prod + c  # 32 + 32 -> 33, auto-truncated to 32 on connect

print(m.to_verilog())