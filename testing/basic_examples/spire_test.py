from spire import Component, IORecord, Input, Output, UInt
from spire.expr import Wire


class MulAddComb(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(16)), b=Input(UInt(16)), c=Input(UInt(32)), y=Output(UInt(32)))
        self.elaborate()

    def elaborate(self):
        prod = Wire(UInt(32))
        prod <<= self.io.a * self.io.b   # 16x16 -> 32
        self.io.y <<= prod + self.io.c   # 32 + 32 -> 33, auto-truncated to 32 on connect


print(MulAddComb().to_verilog(name="MulAddComb"))
