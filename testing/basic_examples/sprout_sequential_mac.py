from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl import Register, UInt

mac = Module("Mac32", with_clock=True, with_reset=True)
a = mac.input(UInt(16), "a")
b = mac.input(UInt(16), "b")
acc_out = mac.output(UInt(32), "acc_out")
acc = Register(UInt(32))  # or acc = mac.reg(UInt(32), "acc", init=0)
acc <<= acc + (a * b)
acc_out <<= acc
print(mac.to_verilog())
