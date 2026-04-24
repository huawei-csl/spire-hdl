from spirehdl.spirehdl import  Bool
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator  # your simulator with peek/watch


m = Module("ExprDemo", with_clock=False, with_reset=False)
a = m.input(Bool(), "a")
b = m.input(Bool(), "b")
c = m.input(Bool(), "c")
y = m.output(Bool(), "y")
expr = (a ^ b) & ~c    # arbitrary Expr
y <<= expr

sim = Simulator(m)
sim.set("a",1).set("b",1).set("c",0).eval()

# Peek the expression directly
print("expr value =", sim.peek(expr))  # 0 (1^1=0; 0 & ~0 -> 0)
print("y =", sim.get("y"))             # 0

# Watch the Expr too
sim.watch(expr, alias="EX").eval()
print("watch EX =", sim.get_watch("EX"))
