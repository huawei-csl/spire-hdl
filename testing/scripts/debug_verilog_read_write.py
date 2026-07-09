from aigverse import Aig, write_aiger, write_verilog, write_dot, read_aiger_into_aig, read_verilog_into_aig, read_pla_into_aig

# Create a sample AIG
aig = Aig()
a = aig.create_pi()
b = aig.create_pi()
f = aig.create_and(a, b)
aig.create_po(f)

# Write to AIGER format
write_aiger(aig, "example.aig")

# Write to Verilog format
write_verilog(aig, "example.v")

# Write to DOT format
write_dot(aig, "example.dot")

# Read from AIGER format
read_aig = read_aiger_into_aig("example.aig")

# Read from Verilog format
read_verilog_aig = read_verilog_into_aig("example.v")

# Read from PLA format
#read_pla_aig = read_pla_into_aig("example.pla")

print(f"Original AIG size: {aig.size()}")
print(f"Read AIGER AIG size: {read_aig.size()}")
print(f"Read Verilog AIG size: {read_verilog_aig.size()}")
#print(f"Read PLA AIG size: {read_pla_aig.size()}")
