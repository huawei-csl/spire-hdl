from spire import Component, IORecord, Input, Output, Bool, UInt, SInt
from spire.expr import mux, cat


class LogicDemo(Component):
    def __init__(self):
        self.io = IORecord(
            x=Input(UInt(8)), y=Input(UInt(8)), sg=Input(SInt(8)), f=Input(Bool()),
            z=Output(UInt(9)), eq=Output(Bool()), hi=Output(UInt(4)), w=Output(UInt(8)),
        )
        self.elaborate()

    def elaborate(self):
        io = self.io
        io.z  <<= io.x + io.y                 # 8+8 -> 9 bits, auto-fit/truncate handled
        io.eq <<= io.x == io.y                # Bool
        io.hi <<= cat(io.y[6:8], io.x[6:8])   # concat 2+2 = 4 bits
        io.w  <<= mux(io.f, io.x & io.y, io.x | io.y)   # mux on Bool


print(LogicDemo().to_verilog(name="LogicDemo"))
