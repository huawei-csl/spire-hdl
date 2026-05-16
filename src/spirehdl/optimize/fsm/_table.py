"""Extract an FSM transition table from a register's driver tree.

Given the FSM's state register ``reg`` (whose ``_driver`` is the full next-state mux tree the user built inside the
``with`` block) and optionally a list of Moore output signals, enumerate every ``(state_value × input_combination)``
pair and evaluate the driver to recover ``next_state_value`` and per-output values.

This is the "symbolic eval over an Expr DAG" use of ``_evaluator``. The inputs are auto-discovered via
``_walker.find_input_signals`` (every Signal leaf reachable from the driver, minus ``reg`` itself and minus
auto-shared CSE wires).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Sequence, TYPE_CHECKING

from spirehdl.optimize.fsm._evaluator import eval_with
from spirehdl.optimize.fsm._walker import find_input_signals

if TYPE_CHECKING:
    from spirehdl.spirehdl import Signal
    from spirehdl.spirehdl_state import State


class TooLargeForExhaustiveExtraction(RuntimeError):
    """Raised when the input domain exceeds ``MAX_INPUT_COMBINATIONS``."""


# Conservative cap on the per-FSM input domain. 65 536 = 2^16 input bits, which covers any FSM with a few 1-bit /
# few-bit inputs. Larger needs the future symbolic-input variant — for now we skip minimisation on those.
MAX_INPUT_COMBINATIONS = 1 << 16


@dataclass
class TransitionTable:
    """Holds the enumerated next-state-and-output table for one FSM.

    ``transitions[state_value][input_tuple] = next_state_value``.
    ``outputs[state_value][input_tuple] = tuple(out_value_per_signal)``.
    ``input_signals`` records the (deduped) list of free input Signals in the order their values appear in
    ``input_tuple``. ``input_combos`` is the explicit list of input tuples enumerated — derived from
    ``input_signals`` by the extractor but can be set directly by tests / synthetic callers.
    """
    state_values: list[int]
    input_signals: list["Signal"]
    transitions: dict[int, dict[tuple[int, ...], int]] = field(default_factory=dict)
    outputs:     dict[int, dict[tuple[int, ...], tuple[int, ...]]] = field(default_factory=dict)
    output_signals: list["Signal"] = field(default_factory=list)
    input_combos: list[tuple[int, ...]] = field(default_factory=list)

    def __post_init__(self):
        if not self.input_combos:
            self.input_combos = self._enumerate_combos()

    def _enumerate_combos(self) -> list[tuple[int, ...]]:
        if not self.input_signals:
            return [()]
        return list(product(*[range(1 << s.typ.width) for s in self.input_signals]))

    def all_input_combos(self) -> list[tuple[int, ...]]:
        return list(self.input_combos)


def extract_transition_table(
    reg: "Signal",
    state_cls: "type[State]",
    outputs: Sequence["Signal"] = (),
    *,
    max_input_combinations: int = MAX_INPUT_COMBINATIONS,
) -> TransitionTable:
    """Enumerate ``(state_value, input_tuple) -> (next_state_value, output_tuple)``.

    Parameters
    ----------
    reg : Signal
        The FSM's state register. Must have a driver that's the next-state expression. ``reg.typ`` must equal
        ``state_cls.typ``.
    state_cls : subclass of State
        The state set. Used to enumerate ``state_value``s over.
    outputs : sequence of Signals
        Moore output signals. Their drivers are evaluated alongside the next-state expression.
    max_input_combinations : int
        Defensive cap. Raises ``TooLargeForExhaustiveExtraction`` if the product of input domain sizes exceeds this.

    Returns
    -------
    TransitionTable
    """
    if reg._driver is None:
        raise ValueError(f"register {reg.name!r} has no driver; nothing to extract")

    # Auto-discover input signals from reg._driver + every output's driver.
    roots: list = [reg._driver]
    for o in outputs:
        if o._driver is not None:
            roots.append(o._driver)
    inputs = find_input_signals(roots, state_cls, exclude=[reg])

    # Check input domain size up front.
    domain = 1
    for s in inputs:
        domain *= 1 << s.typ.width
        if domain > max_input_combinations:
            raise TooLargeForExhaustiveExtraction(f"input domain {domain} exceeds cap {max_input_combinations}")

    table = TransitionTable(
        state_values=list(state_cls._values.values()),
        input_signals=list(inputs),
        output_signals=list(outputs),
    )

    for sv in table.state_values:
        table.transitions[sv] = {}
        table.outputs[sv] = {}
        for ic in table.all_input_combos():
            bindings = [(reg, sv)] + list(zip(inputs, ic))
            try:
                next_sv = eval_with(reg._driver, bindings) & ((1 << reg.typ.width) - 1)
            except ValueError as e:
                raise ValueError(f"extract_transition_table: failed at state={sv}, inputs={ic}: {e}")
            table.transitions[sv][ic] = next_sv

            out_tup = []
            for o in outputs:
                if o._driver is None:
                    out_tup.append(0)
                else:
                    ov = eval_with(o._driver, bindings) & ((1 << o.typ.width) - 1)
                    out_tup.append(ov)
            table.outputs[sv][ic] = tuple(out_tup)

    return table
