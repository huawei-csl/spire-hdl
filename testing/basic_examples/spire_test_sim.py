from spire import Component, IORecord, Input, Output, UInt, Simulator
from spire.expr import Wire


class MulAddComb(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(16)), b=Input(UInt(16)), c=Input(UInt(32)), y=Output(UInt(32)))
        self.elaborate()

    def elaborate(self):
        prod = Wire(UInt(32))
        prod <<= self.io.a * self.io.b
        self.io.y <<= prod + self.io.c   # (32+32)->33, output truncates to 32


sim = Simulator(MulAddComb())
sim.set("a", 3).set("b", 5).set("c", 7).eval()
print("y =", sim.get("y"))  # y = 22 (3*5 + 7)
