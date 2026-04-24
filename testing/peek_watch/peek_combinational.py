from spirehdl.spirehdl import  Bool
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator  # your simulator with peek/watch

# y = (a & b) | c, with an internal 'mid' wire for demo
m = Module("CombDemo", with_clock=False, with_reset=False)
a = m.input(Bool(), "a")
b = m.input(Bool(), "b")
c = m.input(Bool(), "c")
mid = m.wire(Bool(), "mid")
y = m.output(Bool(), "y")

mid <<= a & b
y <<= mid | c

sim = Simulator(m)

# Set inputs and evaluate once
sim.set("a", 1).set("b", 0).set("c", 0).eval()

print("y =", sim.get("y"))           # 0
print("mid (peek by name) =", sim.peek("mid"))  # 0

# Watch 'mid' and 'y'
sim.watch("mid").watch("y", alias="Y")
sim.eval()  # capture current values into watch table
print("watch mid =", sim.get_watch("mid"))      # 0
print("watch Y   =", sim.get_watch("Y"))        # 0

# Change inputs and re-eval
sim.set("b", 1).eval()
print("mid now =", sim.peek("mid"))   # 1
print("y now   =", sim.get("y"))      # 1
