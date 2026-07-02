"""Spire design DB — a content-addressed store of verification-gated implementations.

For each subcircuit (a *slot*, keyed by its golden spec) the DB holds correct implementations,
each with metric vectors and provenance. Producers (campaigns, agents, humans) insert through the
verification gate; consumers select deterministically. S1 ships registration, the verification
seam (Tier-0 CEC, bounded budget, fail-and-choose), and the insert gate.

Keeping this package importable without pyosys/aigverse: heavy spire imports happen lazily inside
the functions that need them.
"""
from spire.design_db.insert import InsertResult, insert_design  # noqa: F401
from spire.design_db.keys import CANONICAL_TOP, golden_and_key, normalize, port_spec, spec_key  # noqa: F401
from spire.design_db.store import (DB_DIRNAME, DB_ENV, DesignDB, DesignDBError,  # noqa: F401
                                   register_slot, resolve_db_root)
from spire.design_db.verify import (DEFAULT_CEC_BUDGET_S, CECInapplicable, CECTimeout,  # noqa: F401
                                    SlotUnverified, VerificationError, VerificationFailed,
                                    cec_check, default_verification, detect_class)
