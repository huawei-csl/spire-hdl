"""Lean 4 bridge for the design_db Lean tier: lowered Spire netlist -> Lean circuit model -> proof.

Handwritten once (shipped here):
    SpireSemantics.lean   BitVec semantics of Spire's signed add / mul / resize
    bridge.py             netlist -> Lean, Interface generation, Lake scaffolding, contract-free proof
                          template, `lake build`, proof checker (statement + type-based axiom audit)

Everything under a generated Lake project is derived from netlists and must not be edited by hand,
except `<Id>Proof.lean` (see spire.design_db.verify_lean).
"""
