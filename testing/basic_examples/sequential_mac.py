from spire import Component, IORecord, Input, Output, UInt
from spire.expr import Register


class Mac32(Component):
    def __init__(self):
        self.io = IORecord(a=Input(UInt(16)), b=Input(UInt(16)), acc_out=Output(UInt(32)))
        self.elaborate()

    def elaborate(self):
        acc = Register(UInt(32), init=0, name="acc")
        acc <<= acc + (self.io.a * self.io.b)
        self.io.acc_out <<= acc


print(Mac32().to_verilog(name="Mac32", with_clock=True, with_reset=True))
