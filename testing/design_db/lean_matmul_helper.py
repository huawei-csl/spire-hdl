"""Test helpers that WRITE Lean proofs for the matmul-accumulate candidates. The package never does;
in production this is the agent's job (a skill may carry code like this as a first attempt).

Pieces: exact-width lemmas per arithmetic shape, a spec layer `Mmac` (Y = C + A·B over ℤ) with its
`equiv` proof, a candidate proof through the layer, a delta proof against an admitted design, and a
second layer `Dot` proven via `Mmac`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List

from spire.design_db.lean.bridge import Port


def exact_lemma(op, wa, wb, wo) -> str:
    fn, sym = ('mulSS', '*') if op == '*' else ('addSS', '+')
    ovf, api = ('smulOverflow', 'toInt_mul_of_not_smulOverflow') if op == '*' else ('saddOverflow', 'toInt_add_of_not_saddOverflow')
    return f"""@[simp] theorem {fn[:3]}_{wa}_{wb}_{wo}_exact (a : BitVec {wa}) (b : BitVec {wb}) :
    ({fn} {wo} a b).toInt = a.toInt {sym} b.toInt := by
  unfold {fn}
  have h : ¬ ((a.signExtend {wo}).{ovf} (b.signExtend {wo}) = true) := by
    bv_decide
  calc
    ((a.signExtend {wo}) {sym} (b.signExtend {wo})).toInt
        = (a.signExtend {wo}).toInt {sym} (b.signExtend {wo}).toInt := by
            exact BitVec.{api} (x := a.signExtend {wo}) (y := b.signExtend {wo}) h
    _ = a.toInt {sym} b.toInt := by
          rw [BitVec.toInt_signExtend_of_le (x := a) (by omega)]
          rw [BitVec.toInt_signExtend_of_le (x := b) (by omega)]
"""


def lemmas(shapes: Iterable) -> str:
    return '\n'.join(exact_lemma(*s) for s in sorted(shapes, key=lambda x: (x[0], x[3], x[1], x[2])))


@dataclass(frozen=True)
class MatmulContract:
    """Y[i][j] = C[i][j] + sum_k A[i][k]*B[k][j], all signed two's complement."""
    m: int; n: int; k: int
    a_width: int; b_width: int; c_width: int; y_width: int

    def ports(self) -> List[Port]:
        P = lambda name, w, d: {"name": name, "width": w, "signed": True, "dir": d}
        return ([P(f'a_{i}_{k}', self.a_width, 'input') for i in range(self.m) for k in range(self.k)]
                + [P(f'b_{k}_{j}', self.b_width, 'input') for k in range(self.k) for j in range(self.n)]
                + [P(f'c_{i}_{j}', self.c_width, 'input') for i in range(self.m) for j in range(self.n)]
                + [P(f'y_{i}_{j}', self.y_width, 'output') for i in range(self.m) for j in range(self.n)])

    def spec_lean(self, name: str = 'Mmac', *, reverse_terms: bool = False) -> str:
        """A spec layer: every output equals C + A·B over the integers (terms reversed for a 2nd layer)."""
        def rhs(i, j):
            terms = [f'i.c_{i}_{j}.toInt'] + [f'i.a_{i}_{k}.toInt * i.b_{k}_{j}.toInt' for k in range(self.k)]
            return ' + '.join(terms[::-1] if reverse_terms else terms)
        ents = [(i, j) for i in range(self.m) for j in range(self.n)]
        body = [f'    (eval i).y_{i}_{j}.toInt = {rhs(i, j)}' + (' ∧' if ix + 1 < len(ents) else '') for ix, (i, j) in enumerate(ents)]
        return '\n'.join([f'-- Spec layer {name}: Y = C + A·B over the integers.', 'import Interface', 'open Interface', '',
                          f'namespace {name}', 'def Correct (eval : Inputs → Outputs) : Prop :=', '  ∀ i : Inputs,', *body,
                          f'end {name}', ''])


def layer_proof_lean(name: str, golden_shapes: Iterable) -> str:
    """`name.equiv : ∀ e, name.Correct e ↔ Spec.Correct e`, via `name.golden : name.Correct GoldenCircuit.eval`."""
    return f"""import {name}
import Spec
import Std.Tactic
open SpireSemantics Interface
namespace {name}
{lemmas(golden_shapes)}
theorem golden : {name}.Correct GoldenCircuit.eval := by
  intro i; and_intros <;> (simp [GoldenCircuit.eval]; ac_rfl)
theorem equiv (e : Inputs → Outputs) : {name}.Correct e ↔ Spec.Correct e := by
  constructor
  · intro h i
    have a := h i; have b := golden i
    apply Outputs.ext <;> apply BitVec.toInt_inj.mp <;> simp_all
  · intro h i
    rw [h i]; exact golden i
end {name}
"""


def second_layer_proof_lean(name: str, via: str) -> str:
    """A layer proven equivalent to Spec through another layer (statements differ only by reordering)."""
    return f"""import {name}
import {via}Proof
open Interface
namespace {name}
theorem via_{via} (e : Inputs → Outputs) : {name}.Correct e ↔ {via}.Correct e := by
  constructor <;> intro h i <;> have hi := h i <;> and_intros <;> (simp_all; try ac_rfl)
theorem equiv (e : Inputs → Outputs) : {name}.Correct e ↔ Spec.Correct e := (via_{via} e).trans ({via}.equiv e)
end {name}
"""


def candidate_proof_lean(shapes: Iterable, *, circuit_module: str, proof_module: str, layer: str = 'Mmac') -> str:
    """`implements : Spec.Correct <circuit>.eval` through a spec layer: prove the integer statement, convert."""
    return f"""import {circuit_module}
import {layer}Proof
import Std.Tactic
open SpireSemantics Interface
namespace {proof_module}
{lemmas(shapes)}
theorem implements : Spec.Correct {circuit_module}.eval :=
  ({layer}.equiv _).mp (by intro i; and_intros <;> (simp [{circuit_module}.eval]; ac_rfl))
end {proof_module}
"""


def delta_proof_lean(*, circuit_module: str, proof_module: str, base_circuit: str, base_proof: str) -> str:
    """`implements` for a design that differs from an admitted one only by commuted multiplier operands:
    prove the step to the admitted design, chain with its `implements`. Spec stays the target."""
    return f"""import {circuit_module}
import {base_circuit}
import {base_proof}
open SpireSemantics Interface
namespace {proof_module}
theorem step (i : Inputs) : {circuit_module}.eval i = {base_circuit}.eval i := by
  simp only [{circuit_module}.eval, {base_circuit}.eval, mulSS, BitVec.mul_comm]
theorem implements : Spec.Correct {circuit_module}.eval := fun i => (step i).trans ({base_proof}.implements i)
end {proof_module}
"""
