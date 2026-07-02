"""Spire design DB — a content-addressed store of verification-gated implementations.

For each subcircuit (a *slot*, keyed by its golden spec) the DB holds correct implementations,
each with metric vectors and provenance. Producers (optimization tools, agents, humans) insert
through the verification gate; consumers select deterministically — most ergonomically via the
``@from_design_db`` decorator, which splices the selected implementation in place of the decorated
function (miss ⇒ the original logic).

S1 shipped registration, the verification seam (Tier-0 CEC, bounded budget, fail-and-choose) and
the insert gate; S2 adds selection, the decorator, and the ``spire db`` CLI. Heavy spire imports
(pyosys/aigverse) happen lazily inside the functions that need them, so importing this package
stays dependency-light.
"""
from spire.design_db.decorator import from_design_db  # noqa: F401
from spire.design_db.insert import InsertResult, insert_design, seed_original  # noqa: F401
from spire.design_db.keys import CANONICAL_TOP, golden_and_key, normalize, port_spec, spec_key  # noqa: F401
from spire.design_db.select import (SelectionResult, constrained, lexicographic,  # noqa: F401
                                    metric_value, pareto_front, resolve_metric, select_design,
                                    weighted)
from spire.design_db.store import (DB_DIRNAME, DB_ENV, DesignDB, DesignDBError,  # noqa: F401
                                   register_slot, resolve_db_root)
from spire.design_db.verify import (DEFAULT_CEC_BUDGET_S, CECInapplicable, CECTimeout,  # noqa: F401
                                    SlotUnverified, VerificationError, VerificationFailed,
                                    cec_check, default_verification, detect_class)
from spire.design_db.verify_sim import (SimTimeout, freeze_sim_verification,  # noqa: F401
                                        run_frozen_tb)
