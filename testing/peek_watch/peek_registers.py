from spirehdl.spirehdl import UInt
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator

# 4-bit accumulator: acc <<= acc + din
m = Module("Accu", with_clock=True, with_reset=True)
din = m.input(UInt(4), "din")
acc = m.reg(UInt(4), "acc", init=0)
y = m.output(UInt(4), "y")

acc <<= acc + din
y <<= acc

sim = Simulator(m)

# reset to init value
sim.reset().deassert_reset()
print("after reset, acc =", sim.peek("acc"))  # 0

# set din and check next-state before tick
sim.set("din", 3).eval()
print("peek_next(acc) =", sim.peek_next("acc"))  # 3 (0 + 3)
print("y (current acc) =", sim.get("y"))  # 0

# take a clock step
sim.step()
print("after 1st tick, acc =", sim.peek("acc"))  # 3
print("y =", sim.get("y"))  # 3

# watch both acc and next-state through time
sim.watch("acc").watch(din, alias="DIN")
sim.set("din", 5).eval()
print("watch acc:", sim.get_watch("acc"))  # 3 (current)
print("peek_next(acc):", sim.peek_next("acc"))  # (3 + 5) mod 16 = 8
sim.step()
print("after 2nd tick, acc =", sim.peek("acc"))  # 8
