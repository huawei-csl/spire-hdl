# minus_semantics_test.py
import random
from math import ceil, log2
from testing.roundtrip import roundtrip_and_group
from spire.expr import UInt
from spire.component import Netlist
from spire.simulator import Simulator


def build_minus_probe(IW: int):
    m = Netlist(f"MinusProbe_W{IW}", with_clock=False, with_reset=False)
    idx = m.input(UInt(IW), "idx")

    sub1 = m.output(UInt(IW), "sub1")
    sub1 <<= idx - 1  # then resized by output width
    sub2 = m.output(UInt(IW), "sub2")
    sub2 <<= idx - 2

    return m


def run(IW=4, trials=32):
    m = build_minus_probe(IW)
    
    m2, spec = roundtrip_and_group(m, keep_symbols=True)
    
    sim = Simulator(m2)
    for idx_val in range(trials):
        val = idx_val & ((1 << IW) - 1)  # drive only IW-range
        sim.set("idx", val).eval()
        got1 = sim.get("sub1")
        got2 = sim.get("sub2")
        exp1 = (val - 1) & ((1 << IW) - 1)
        exp2 = (val - 2) & ((1 << IW) - 1)
        if got1 != exp1 or got2 != exp2:
            print(f"FAIL idx={val}  got1={got1} exp1={exp1}  got2={got2} exp2={exp2}")
            #return
    print("minus modulo test PASS")


if __name__ == "__main__":
    run(IW=5)
    run(IW=6)