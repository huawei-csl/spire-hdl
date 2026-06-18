from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, MultiplierTestVectorsExhaustive, to_encoding
from spire.helpers import get_aig_stats, get_yosys_metrics, run_vectors
from spire.expr import UInt
from spire.component import ImportedComponent
from spire.io_record import IORecord, Input, Output
from spire.simulator import Simulator
from spire.arithmetic.int_multipliers.eval.testvector_generation import MultiplierTestVectors

# Import shell: logic is loaded from an AIG via from_aig_file() (no elaborate()).
class Multiplier(ImportedComponent):
    """A multiplier whose logic is imported from an AIG netlist."""

    def __init__(self, width: int = 8):
        self.width = width
        self.io = IORecord(
            a=Input(UInt(width)),
            b=Input(UInt(width)),
            y=Output(UInt(width * 2)),
        )


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
