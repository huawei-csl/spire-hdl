from spire.component import Module
from spire.expr import UInt
from spire.simulator import Simulator


m = Module("MulAddComb", with_clock=False, with_reset=False)
a = m.input(UInt(16), "a")
b = m.input(UInt(16), "b")
c = m.input(UInt(32), "c")
y = m.output(UInt(32), "y")

prod = m.wire(UInt(32), "prod")
prod <<= a * b
y <<= prod + c  # (32+32)->33, output truncates to 32

sim = Simulator(m)
sim.set("a", 3).set("b", 5).set("c", 7).eval()
print("y =", sim.get("y"))  # y = 22 (3*5 + 7)