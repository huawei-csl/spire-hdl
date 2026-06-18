"""``spire.optimize.fsm`` — FSM minimisation + state-encoding-search utilities.

Public surface — both context managers and the three underlying passes exported as standalone callables for tooling
reuse and future decorators::

    from spire.optimize.fsm import (
        optimized_fsm,        # Hopcroft state minimisation wrapper
        optimized_encoding,   # bit-assignment search wrapper
        minimize_fsm,         # standalone Hopcroft
        search_encoding,      # standalone bit-assignment search
        apply_encoding,       # rewrite Const values for a chosen assignment
    )

Conventional import path is ``from spire.state import ...`` — ``spire_state.py`` re-exports the same
names so users get a single import site for the State + FSM-optimisation API.
"""
from spire.optimize.fsm._emit import apply_encoding, restore_encoding, snapshot_encoding
from spire.optimize.fsm._encoding_search import search_encoding
from spire.optimize.fsm._hopcroft import equivalence_classes, minimize_fsm
from spire.optimize.fsm.optimized_encoding import optimized_encoding
from spire.optimize.fsm._adjacency import adjacency_search
from spire.optimize.fsm._minimize_emit import build_comb_cone, minimize_and_rewrite
from spire.optimize.fsm.optimized_fsm import optimized_fsm

__all__ = [
    "optimized_fsm",
    "optimized_encoding",
    "minimize_fsm",
    "minimize_and_rewrite",
    "adjacency_search",
    "build_comb_cone",
    "search_encoding",
    "apply_encoding",
    "snapshot_encoding",
    "restore_encoding",
    "equivalence_classes",
]
