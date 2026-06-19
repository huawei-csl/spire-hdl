from spire.expr import Bool, UInt
from spire.state import Encoding, State, state
from spire.control_structures import case_, default, if_, switch_
from spire.component import Netlist
from spire.simulator import Simulator, _sid


class BinaryFSM(State, encoding=Encoding.BINARY):
    IDLE = state()
    RUN = state()
    DONE = state()


class OnehotFSM(State, encoding=Encoding.ONEHOT):
    A = state()
    B = state()
    C = state()
    D = state()


class GrayFSM(State, encoding=Encoding.GRAY):
    S0 = state()
    S1 = state()
    S2 = state()
    S3 = state()


def test_state_binary_encoding():
    assert BinaryFSM._width == 2
    assert BinaryFSM.IDLE.value == 0
    assert BinaryFSM.RUN.value == 1
    assert BinaryFSM.DONE.value == 2
    assert len(BinaryFSM.names) == 3


def test_state_onehot_encoding():
    assert OnehotFSM._width == 4
    assert OnehotFSM.A.value == 0b0001
    assert OnehotFSM.B.value == 0b0010
    assert OnehotFSM.C.value == 0b0100
    assert OnehotFSM.D.value == 0b1000


def test_state_gray_encoding():
    assert GrayFSM._width == 2
    assert GrayFSM.S0.value == 0b00
    assert GrayFSM.S1.value == 0b01
    assert GrayFSM.S2.value == 0b11
    assert GrayFSM.S3.value == 0b10


def test_state_machine_simulation():
    """Simple 3-state FSM: IDLE -> RUN -> DONE -> IDLE, controlled by 'go' input."""
    m = Netlist("FSM", with_clock=True, with_reset=False)
    go = m.input(Bool(), "go")
    reg = m.reg(BinaryFSM.typ, "state", init=BinaryFSM.IDLE)
    out = m.output(UInt(2), "out")

    out <<= 0

    with switch_(reg):
        with case_(BinaryFSM.IDLE):
            out <<= 0
            with if_(go):
                reg <<= BinaryFSM.RUN
        with case_(BinaryFSM.RUN):
            out <<= 1
            reg <<= BinaryFSM.DONE
        with case_(BinaryFSM.DONE):
            out <<= 2
            reg <<= BinaryFSM.IDLE
        with default():
            reg <<= BinaryFSM.IDLE

    sim = Simulator(m)
    sim.eval()

    # Initial state: IDLE, out=0
    assert sim.get(reg) == BinaryFSM.IDLE.value
    assert sim.get(out) == 0

    # No go signal: stays in IDLE
    sim.set(go, 0)
    sim.step()
    assert sim.get(reg) == BinaryFSM.IDLE.value

    # Assert go: next state is RUN
    sim.set(go, 1)
    sim.step()
    assert sim.get(reg) == BinaryFSM.RUN.value

    # Advance to RUN
    sim.set(reg, BinaryFSM.RUN.value)
    sim.eval()
    assert sim.get(out) == 1
    sim.step()
    assert sim.get(reg) == BinaryFSM.DONE.value

    # Advance to DONE
    sim.set(reg, BinaryFSM.DONE.value)
    sim.eval()
    assert sim.get(out) == 2
    sim.step()
    assert sim.get(reg) == BinaryFSM.IDLE.value

if __name__ == "__main__":
    print("Running tests...")
    test_state_binary_encoding()
    test_state_onehot_encoding()
    test_state_gray_encoding()
    test_state_machine_simulation()
