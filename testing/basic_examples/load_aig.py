from attr import dataclass


from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, MultiplierTestVectorsExhaustive, to_encoding
from spirehdl.helpers import get_aig_stats, get_yosys_metrics, run_vectors
from spirehdl.spirehdl import Signal, UInt
from spirehdl.spirehdl_module import Component
from spirehdl.spirehdl_simulator import Simulator
from spirehdl.arithmetic.int_multipliers.eval.testvector_generation import MultiplierTestVectors

# Example 1: Simple Adder Component
# ==================================
class Multiplier(Component):
    """A simple adder component that adds two numbers."""

    def __init__(self, width: int = 8):
        self.width = width

        # Define IO ports using a dataclass
        @dataclass
        class IO:
            a: Signal  # input a
            b: Signal  # input b
            y: Signal  # output product

        # Create the IO structure with Signal instances
        self.io = IO(
            a=Signal(name="a", typ=UInt(width), kind="input"),
            b=Signal(name="b", typ=UInt(width), kind="input"),
            y=Signal(name="y", typ=UInt(width * 2), kind="output"),
        )

        # Build the internal logic
        #self.elaborate()


if __name__ == "__main__":

    width = 4

    mult = Multiplier(width=width)
    print(mult)
    #mult.from_aig_file("../gate_net/circuit.aig", make_internal=False)
    mult.from_aig_file("../AI4EDA_TNet/out.aag", make_internal=False)

    # from_aig_file(aig_file_path, aiger_map_file_path, make_internal=False)

    print(mult)

    m_mult = mult.to_module("multiplier_from_aig")

    stats= get_aig_stats(m_mult, n_iter_optimizations=10)
    print(f"AIG stats: {stats}")

    yosys_metrics = get_yosys_metrics(m_mult, deepsyn=False)
    print(f"Yosys metrics: {yosys_metrics}")

    sim = Simulator(m_mult)
    sim.set("a", 3).set("b", 15)
    sim.eval()
    print(f"Inputs: {sim.peek_inputs()}")
    print(f"Outputs: {sim.peek_outputs()}")

    vecs = MultiplierTestVectorsExhaustive(
        a_w=width,
        b_w=width,
        a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned,
        y_encoding=Encoding.unsigned,
    ).generate()

    run_vectors(m_mult, vecs, print_on_pass=True)
