"""Test helpers that WRITE Lean proofs for the matmul-accumulate candidates. The package never does;
in production this is the agent's job (a skill may carry code like this as a first attempt).
Also `OpZoo`/`Bits*` components that exercise the wider operator set for the differential and bv_decide tests.

Pieces: exact-width lemmas per arithmetic shape, a spec layer `Mmac` (Y = C + A·B over ℤ) with its
`equiv` proof, a candidate proof through the layer, a delta proof against an admitted design, and a
second layer `Dot` proven via `Mmac`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List

from spire.design_db.lean.bridge import Port


def exact_lemma(op, wa, wb, wo) -> str:
    fn, sym, ovf, api = {'+': ('addSS', '+', 'saddOverflow', 'toInt_add_of_not_saddOverflow'),
                         '-': ('subSS', '-', 'ssubOverflow', 'toInt_sub_of_not_ssubOverflow'),
                         '*': ('mulSS', '*', 'smulOverflow', 'toInt_mul_of_not_smulOverflow')}[op]
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


# --- operator-coverage components (for the differential test and whole-design bv_decide proofs) ---------

def opzoo_component():
    """Every supported combinational operator, with signed / unsigned / mixed operands."""
    from spire import Bool, Component, IORecord, Input, Output, SInt, UInt
    from spire.expr import cat, mux

    class OpZoo(Component):
        def __init__(self):
            self.io = IORecord(a=Input(SInt(8)), b=Input(SInt(8)), u=Input(UInt(8)), v=Input(UInt(6)), s=Input(Bool()),
                               y_and=Output(UInt(8)), y_or=Output(UInt(8)), y_xor=Output(UInt(8)), y_not=Output(UInt(8)),
                               y_sub=Output(SInt(9)), y_usub=Output(UInt(9)), y_uadd=Output(UInt(9)), y_umul=Output(UInt(14)),
                               y_mixadd=Output(SInt(9)), y_mixmul=Output(SInt(16)), y_shl=Output(SInt(10)),
                               y_shr=Output(UInt(8)), y_shlv=Output(UInt(8)), y_lt=Output(Bool()), y_eq=Output(Bool()),
                               y_mixlt=Output(Bool()), y_ge=Output(Bool()), y_mux=Output(SInt(8)), y_mux2=Output(SInt(9)),
                               y_cat=Output(UInt(10)), y_slice=Output(UInt(4)))
            self.elaborate()

        def elaborate(self):
            io = self.io
            io.y_and <<= io.a & io.b;  io.y_or <<= io.u | io.v;  io.y_xor <<= io.a ^ io.u;  io.y_not <<= ~io.a
            io.y_sub <<= io.a - io.b;  io.y_usub <<= io.u - io.v;  io.y_uadd <<= io.u + io.v;  io.y_umul <<= io.u * io.v
            io.y_mixadd <<= io.a + io.u;  io.y_mixmul <<= io.a * io.u
            io.y_shl <<= io.a << 2;  io.y_shr <<= io.u >> io.v[0:3];  io.y_shlv <<= io.u << io.v[0:3]
            io.y_lt <<= io.a < io.b;  io.y_eq <<= io.u == io.v;  io.y_mixlt <<= io.a < io.u;  io.y_ge <<= io.v >= io.u
            io.y_mux <<= mux(io.s, io.a, io.u);  io.y_mux2 <<= mux(io.a < io.b, io.a - io.b, io.b - io.a)
            io.y_cat <<= cat(io.v, io.s, io.u[0:3]);  io.y_slice <<= io.a[2:6]
    return OpZoo()


def bits_components():
    """(golden, equivalent rewrite) over 8-bit unsigned bitwise logic; different netlists, same function."""
    from spire import Component, IORecord, Input, Output, UInt

    class _Bits(Component):
        def __init__(self):
            self.io = IORecord(a=Input(UInt(8)), b=Input(UInt(8)), c=Input(UInt(8)), y=Output(UInt(8)), z=Output(UInt(8)))
            self.elaborate()

    class BitsGolden(_Bits):
        def elaborate(self):
            self.io.y <<= (self.io.a & self.io.b) | (~self.io.a & self.io.c)     # bitwise mux
            self.io.z <<= self.io.a ^ self.io.b

    class BitsAlt(_Bits):
        def elaborate(self):
            self.io.y <<= (self.io.a & (self.io.b ^ self.io.c)) ^ self.io.c       # same mux, xor form
            self.io.z <<= (self.io.a | self.io.b) & ~(self.io.a & self.io.b)      # same xor, and/or form
    return BitsGolden(), BitsAlt()
