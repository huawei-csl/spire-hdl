"""design_db Lean tier, end to end. Requires `lake` (hard error otherwise). Tests run in file order on
one slot: golden = balanced 2x2x4 MAC. Spec = golden equivalence; layers Mmac then Dot; candidates
left-deep (proof via Mmac) and swapped-operand multipliers (delta proof via the admitted left-deep design).
"""
from __future__ import annotations
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

from spire.design_db import (DesignDB, ProofFailed, ProofRejected, SpecRejected, VerificationFailed, add_spec,
                             check_design, freeze_lean_verification, insert_design, register_slot, seed_original)
from spire.design_db.cli import main as cli
from spire.design_db.store import DesignDBError
from spire.design_db.lean.bridge import collect_signed_arith_shapes, design_modules

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lean_matmul_helper import (candidate_proof_lean, delta_proof_lean, layer_proof_lean,  # noqa: E402
                                second_layer_proof_lean)


def candidates():
    spec = importlib.util.spec_from_file_location('mmac_candidates', HERE / 'mmac_candidates.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def design_file(path: Path, expr: str) -> Path:
    path.write_text(f'import sys\nsys.path.insert(0, {str(HERE)!r})\nfrom mmac_candidates import *\ndef build():\n    return {expr}\n')
    return path


MISWIRED = '''
class Miswired(LeftDeepMatmulAccumulate):
    """Uses B[k, 0] for every output column: right ports, wrong function."""
    def elaborate(self):
        rows = []
        for i in range(self.cfg.dims.dim_m):
            row = []
            for j in range(self.cfg.dims.dim_n):
                ps = [build_multiplier(self.A[i, k], self.B[k, 0], self.cfg.mult_cfg) for k in range(self.cfg.dims.dim_k)]
                dot = ps[0]
                for p in ps[1:]: dot = build_adder(dot, p, self.cfg.add_cfg)
                acc = build_adder(self.C[i, j], dot, self.cfg.add_cfg)
                y = Signal(name=f"y_{i}_{j}", typ=self.io_hdl_type(acc.typ.width), kind="output"); y <<= acc
                row.append(y)
            rows.append(Array(row))
        self.Y = Array(rows)
def build():
    return Miswired(CFG, signed_io_type=True)
'''


@pytest.fixture(scope='module')
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp('db'); c = candidates(); files = tmp_path_factory.mktemp('designs')
    key = register_slot(c.CANDIDATES['BalancedMmac'](), db=root, name='mmac')
    shapes = {n: collect_signed_arith_shapes(c.CANDIDATES[n]().to_netlist(n)) for n in c.CANDIDATES}
    bad = files / 'miswired.py'
    bad.write_text(f'import sys\nsys.path.insert(0, {str(HERE)!r})\nfrom mmac_candidates import *\n'
                   'from spire.arithmetic.int_arithmetic_config import build_adder, build_multiplier\n'
                   'from spire.composite.array import Array\nfrom spire.expr import Signal\n' + MISWIRED)
    return dict(db=root, key=key, cand=c, shapes=shapes, files=files, ids={},
                golden_py=design_file(files / 'golden.py', 'CANDIDATES["BalancedMmac"]()'),
                left_py=design_file(files / 'left_deep.py', 'CANDIDATES["LeftDeepMmac"]()'),
                new_py=design_file(files / 'commuted_mul.py', 'CANDIDATES["CommutedMulMmac"]()'), bad_py=bad)


def slot_dir(env):
    return DesignDB.open(env['db']).slot_dir(env['key'])


def workspace(env, design, name):
    """`verify --workspace` without a proof: writes the workspace, tells the agent which file to write."""
    ws = env['files'] / name
    with pytest.raises(VerificationFailed, match=r'no proof given; workspace written') as e:
        check_design(env['key'], design, workdir=ws, db=env['db'])
    hash10 = re.search(r'D([0-9a-f]{10})_Proof\.lean', str(e.value)).group(1)
    return ws, hash10


# --- registration + freeze ----------------------------------------------------------------------

def test_registration_captures_the_golden_lean_model(env):
    cap = json.loads((slot_dir(env) / 'golden_lean.json').read_text())
    assert 'namespace GoldenCircuit' in cap['circuit'] and len(cap['shapes']) == 4


def test_untranslatable_golden_cannot_be_lean_gated(tmp_path):
    from spire import Component, IORecord, Input, Output, UInt
    class Unsigned(Component):
        def __init__(self):
            self.io = IORecord(a=Input(UInt(8)), b=Input(UInt(8)), s=Output(UInt(9))); self.elaborate()
        def elaborate(self): self.io.s <<= self.io.a + self.io.b
    key = register_slot(Unsigned(), db=tmp_path, name='uadd')
    with pytest.raises(DesignDBError, match='not translatable to Lean at registration'):
        freeze_lean_verification(key, db=tmp_path)


def test_freeze_dry_run_writes_nothing(env):
    v = freeze_lean_verification(env['key'], dry_run=True, db=env['db'])
    assert v['dry_run'] and v['method'] == 'lean'
    assert not (slot_dir(env) / 'lean').exists()


def test_freeze_lean_oracle_needs_no_author_and_no_proof(env):
    v = freeze_lean_verification(env['key'], author='agent:x', db=env['db'])
    lean = slot_dir(env) / 'lean'
    assert v['method'] == 'lean' and v['tier'] == 3
    assert {p.name for p in lean.iterdir()} == {'SpireSemantics.lean', 'Interface.lean', 'GoldenCircuit.lean', 'Spec.lean', 'specs'}
    assert 'eval i = GoldenCircuit.eval i' in (lean / 'Spec.lean').read_text()
    assert not os.access(lean / 'Spec.lean', os.W_OK) and not any((lean / 'specs').iterdir())


def test_seed_golden_needs_no_proof(env):
    r = seed_original(env['key'], db=env['db'])
    prov = json.loads((slot_dir(env) / 'designs' / r.design_id / 'provenance.json').read_text())
    assert prov['verification']['proof'].startswith('golden') and prov['verification']['depends_on'] == []


# --- spec layers ---------------------------------------------------------------------------------

def test_add_spec_rejects_bad_names_and_unproven_layers(env):
    c = env['cand']; spec = c.CONTRACT.spec_lean('Mmac')
    with pytest.raises(DesignDBError, match='invalid spec layer name'):
        add_spec(env['key'], 'Spec', spec, 'x', db=env['db'])
    sorry = 'import Mmac\nimport Spec\nopen Interface\nnamespace Mmac\n' \
            'theorem equiv (e : Inputs → Outputs) : Mmac.Correct e ↔ Spec.Correct e := by sorry\nend Mmac\n'
    with pytest.raises(SpecRejected, match='disallowed axioms'):
        add_spec(env['key'], 'Mmac', spec, sorry, db=env['db'])
    assert not (slot_dir(env) / 'lean' / 'specs' / 'Mmac.lean').exists()


def test_add_spec_mmac(env):
    c = env['cand']
    rec = add_spec(env['key'], 'Mmac', c.CONTRACT.spec_lean('Mmac'), layer_proof_lean('Mmac', env['shapes']['BalancedMmac']),
                   author='agent:lean-spec-author', db=env['db'])
    specs = slot_dir(env) / 'lean' / 'specs'
    assert {p.name for p in specs.iterdir()} == {'Mmac.lean', 'MmacProof.lean', 'Mmac.json'}
    assert rec['statement'] == '∀ e, Mmac.Correct e ↔ Spec.Correct e' and 'Mmac.equiv' in rec['axioms']
    with pytest.raises(DesignDBError, match='already exists'):
        add_spec(env['key'], 'Mmac', c.CONTRACT.spec_lean('Mmac'), 'x', db=env['db'])


# --- candidates ----------------------------------------------------------------------------------

def test_workspace_without_proof_names_the_file_to_write(env):
    ws, h = workspace(env, env['left_py'], 'ws_left')
    env['ids']['left'] = h
    circuit, proof = design_modules(h)
    assert {p.name for p in ws.iterdir()} >= {'Spec.lean', 'GoldenCircuit.lean', 'Mmac.lean', 'MmacProof.lean', f'{circuit}.lean'}
    assert not (ws / f'{proof}.lean').exists()                   # the package writes no proof


def test_verilog_text_is_rejected_on_lean_slots(env):
    v = env['cand'].CANDIDATES['CommutedMulMmac']().to_netlist('v').to_verilog()
    with pytest.raises(VerificationFailed, match='spire-native'):
        insert_design(env['key'], v, source='x', db=env['db'])


def test_insert_left_deep_with_proof_through_the_layer(env):
    circuit, proof = design_modules(env['ids']['left'])
    txt = candidate_proof_lean(env['shapes']['LeftDeepMmac'], circuit_module=circuit, proof_module=proof, layer='Mmac')
    assert check_design(env['key'], env['left_py'], proof=txt, db=env['db'])['verdict'] == 'PASS'
    r = insert_design(env['key'], env['left_py'], source='agent:rtl-lean-proof', proof=txt, db=env['db'])
    d = slot_dir(env) / 'designs' / r.design_id
    assert r.design_id.endswith(env['ids']['left'])
    assert {p.name for p in (d / 'lean').iterdir()} == {f'{circuit}.lean', f'{proof}.lean', 'axioms.json', 'build.log'}
    prov = json.loads((d / 'provenance.json').read_text())['verification']
    assert prov['proof'] == f'lean/{proof}.lean' and 'MmacProof' in prov['depends_on']


def test_insert_commuted_mul_with_delta_proof_against_the_admitted_design(env):
    ws, h = workspace(env, env['new_py'], 'ws_new')
    env['ids']['new'] = h
    lc, lp = design_modules(env['ids']['left']); nc, np_ = design_modules(h)
    assert (ws / f'{lc}.lean').exists() and (ws / f'{lp}.lean').exists()      # admitted designs are importable
    txt = delta_proof_lean(circuit_module=nc, proof_module=np_, base_circuit=lc, base_proof=lp)
    r = insert_design(env['key'], env['new_py'], source='agent:rtl-lean-proof', proof=txt, db=env['db'])
    prov = json.loads((slot_dir(env) / 'designs' / r.design_id / 'provenance.json').read_text())['verification']
    assert lp in prov['depends_on'] and 'MmacProof' not in prov['depends_on']
    ax = json.loads((slot_dir(env) / 'designs' / r.design_id / 'lean' / 'axioms.json').read_text())[f'{np_}.implements']
    assert any('bv_decide' in a for a in ax)                                  # inherited transitively via Dleft_Proof


def test_second_layer_via_first_leaves_earlier_proofs_untouched(env):
    c = env['cand']
    before = {p: p.read_text() for p in (slot_dir(env) / 'designs').glob('*/lean/*.lean')}
    add_spec(env['key'], 'Dot', c.CONTRACT.spec_lean('Dot', reverse_terms=True), second_layer_proof_lean('Dot', 'Mmac'), db=env['db'])
    assert {p.name for p in (slot_dir(env) / 'lean' / 'specs').iterdir()} >= {'Dot.lean', 'DotProof.lean', 'Dot.json'}
    assert before == {p: p.read_text() for p in (slot_dir(env) / 'designs').glob('*/lean/*.lean')}
    circuit, proof = design_modules(env['ids']['left'])                       # the same design can now be proven via Dot
    via_dot = candidate_proof_lean(env['shapes']['LeftDeepMmac'], circuit_module=circuit, proof_module=proof, layer='Dot')
    assert check_design(env['key'], env['left_py'], proof=via_dot, db=env['db'])['verdict'] == 'PASS'


def test_dedup_skips_the_proof(env):
    assert insert_design(env['key'], env['left_py'], source='again', db=env['db']).deduped


def test_missing_proof_is_rejected(env):
    with pytest.raises(VerificationFailed, match='proof is required'):
        insert_design(env['key'], env['bad_py'], source='x', db=env['db'])


def test_wrong_design_fails_its_proof(env):
    ws, h = workspace(env, env['bad_py'], 'ws_bad')
    circuit, proof = design_modules(h)
    bad_shapes = collect_signed_arith_shapes(env['cand'].CANDIDATES['LeftDeepMmac']().to_netlist('x'))
    txt = candidate_proof_lean(bad_shapes, circuit_module=circuit, proof_module=proof)
    with pytest.raises(ProofFailed, match='proof does not check'):
        insert_design(env['key'], env['bad_py'], source='x', proof=txt, db=env['db'])
    assert not any(p.name.startswith('x:') for p in (slot_dir(env) / 'designs').iterdir())


def test_weakened_and_forged_proofs_are_rejected(env):
    circuit, proof = design_modules(env['ids']['left'])
    head = f'import {circuit}\nimport Spec\nnamespace {proof}\n'
    with pytest.raises(ProofRejected, match='does not establish'):
        check_design(env['key'], env['left_py'], proof=head + f'theorem implements : True := trivial\nend {proof}\n', db=env['db'])
    forged = head + 'axiom fake._native.bv_decide.ax_1_1 : False\n' \
             f'theorem implements : Spec.Correct {circuit}.eval := False.elim fake._native.bv_decide.ax_1_1\nend {proof}\n'
    with pytest.raises(ProofRejected, match='disallowed axioms'):
        check_design(env['key'], env['left_py'], proof=forged, db=env['db'])


def test_cli_round_trip(tmp_path):
    c = candidates(); db = str(tmp_path / 'db')
    register_slot(c.CANDIDATES['BalancedMmac'](), db=db, name='mmac')
    left = design_file(tmp_path / 'left.py', 'CANDIDATES["LeftDeepMmac"]()')
    shapes = collect_signed_arith_shapes(c.CANDIDATES['BalancedMmac']().to_netlist('g'))
    (tmp_path / 'Mmac.lean').write_text(c.CONTRACT.spec_lean('Mmac'))
    (tmp_path / 'MmacProof.lean').write_text(layer_proof_lean('Mmac', shapes))
    assert cli(['db', 'set-verification', '--db', db, '--slot', 'mmac', '--lean', '--check', '--workspace', str(tmp_path / 'frz')]) == 0
    assert (tmp_path / 'frz' / 'Spec.lean').exists()
    assert cli(['db', 'set-verification', '--db', db, '--slot', 'mmac', '--lean']) == 0
    assert cli(['db', 'add-spec', '--db', db, '--slot', 'mmac', 'Mmac', str(tmp_path / 'Mmac.lean'), str(tmp_path / 'MmacProof.lean'),
                '--author', 'human']) == 0
    ws = tmp_path / 'ws'
    assert cli(['db', 'verify', '--db', db, '--slot', 'mmac', str(left), '--workspace', str(ws)]) == 2   # no proof yet
    circuit = next(ws.glob('D*_Circuit.lean')).stem; h = circuit[1:-8]; proof = design_modules(h)[1]
    (ws / f'{proof}.lean').write_text(candidate_proof_lean(collect_signed_arith_shapes(c.CANDIDATES['LeftDeepMmac']().to_netlist('l')),
                                                           circuit_module=circuit, proof_module=proof))
    assert cli(['db', 'verify', '--db', db, '--slot', 'mmac', str(left), '--proof', str(ws / f'{proof}.lean')]) == 0
    assert cli(['db', 'insert', '--db', db, '--slot', 'mmac', str(left), '--proof', str(ws / f'{proof}.lean'), '--source', 'cli']) == 0
    assert cli(['db', 'insert', '--db', db, '--slot', 'mmac', str(design_file(tmp_path / 'g.py', 'CANDIDATES["CommutedMulMmac"]()'))]) == 2
    assert cli(['db', 'set-verification', '--db', db, '--slot', 'mmac', '--cec']) == 1
    assert cli(['db', 'show', '--db', db, 'mmac']) == 0
