"""spire.design_db.lean.bridge: translation, project assembly, `lake build`, and the proof checker,
on a full layered project (default spec, Mmac layer, layered proof, delta proof, second layer).
`lake` must be on PATH; a missing lake is a hard error (never a skip).
"""
from __future__ import annotations
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

from spire.design_db.lean.bridge import (SEMANTICS_FILE, InterfaceMismatch, LeanExportOptions, ProofCheckError,
                                         TheoremCheck, check_theorems, collect_signed_arith_shapes, default_spec_lean,
                                         design_modules, interface_to_lean, lake_build, lean_imports, netlist_ports,
                                         netlist_to_lean, write_lean_project)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lean_matmul_helper import (MatmulContract, candidate_proof_lean, delta_proof_lean, layer_proof_lean,  # noqa: E402
                                second_layer_proof_lean)


def candidates():
    spec = importlib.util.spec_from_file_location('mmac_candidates', HERE / 'mmac_candidates.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def proj(tmp_path_factory):
    """Golden = balanced; Dleft_ proven via the Mmac layer; Dnew_ (swapped multiplier operands) by delta to Dleft_; Dot via Mmac."""
    c = candidates(); d = tmp_path_factory.mktemp('lean')
    nets = {n: c.CANDIDATES[n]().to_netlist(n) for n in ('BalancedMmac', 'LeftDeepMmac', 'CommutedMulMmac')}
    ports = netlist_ports(nets['BalancedMmac'])
    circ = lambda net, ns: netlist_to_lean(net, options=LeanExportOptions(namespace=ns, ports=ports))
    shapes = {n: collect_signed_arith_shapes(net) for n, net in nets.items()}
    lc, lp = design_modules('left'); nc, np_ = design_modules('new')
    files = {
        'Interface.lean': interface_to_lean(ports), 'GoldenCircuit.lean': circ(nets['BalancedMmac'], 'GoldenCircuit'),
        'Spec.lean': default_spec_lean(),
        'Mmac.lean': c.CONTRACT.spec_lean('Mmac'), 'MmacProof.lean': layer_proof_lean('Mmac', shapes['BalancedMmac']),
        f'{lc}.lean': circ(nets['LeftDeepMmac'], lc),
        f'{lp}.lean': candidate_proof_lean(shapes['LeftDeepMmac'], circuit_module=lc, proof_module=lp),
        f'{nc}.lean': circ(nets['CommutedMulMmac'], nc),
        f'{np_}.lean': delta_proof_lean(circuit_module=nc, proof_module=np_, base_circuit=lc, base_proof=lp),
        'Dot.lean': c.CONTRACT.spec_lean('Dot', reverse_terms=True), 'DotProof.lean': second_layer_proof_lean('Dot', 'Mmac'),
    }
    for n, t in files.items(): (d / n).write_text(t)
    write_lean_project(d, [n[:-5] for n in files])
    checks = [TheoremCheck('MmacProof', 'Mmac.equiv', '∀ e, Mmac.Correct e ↔ Spec.Correct e'),
              TheoremCheck('DotProof', 'Dot.equiv', '∀ e, Dot.Correct e ↔ Spec.Correct e'),
              TheoremCheck(lp, f'{lp}.implements', f'Spec.Correct {lc}.eval'),
              TheoremCheck(np_, f'{np_}.implements', f'Spec.Correct {nc}.eval')]
    return dict(cand=c, nets=nets, ports=ports, dir=d, files=files, checks=checks, lp=lp, lc=lc, nc=nc, np=np_)


def copy_project(src, dst):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('.lake', 'lake-manifest.json')); return dst


def test_interface_and_circuit_roles(proj):
    itf = proj['files']['Interface.lean']
    assert 'structure Inputs' in itf and '@[ext] structure Outputs' in itf and 'a_0_0 : BitVec 8' in itf
    for n in ('GoldenCircuit.lean', f"{proj['lc']}.lean"):
        assert 'structure' not in proj['files'][n] and 'def eval (i : Inputs) : Outputs' in proj['files'][n]
    assert 'def Correct (eval : Inputs → Outputs) : Prop := ∀ i, eval i = GoldenCircuit.eval i' in default_spec_lean()
    assert (proj['dir'] / 'SpireSemantics.lean').read_text() == SEMANTICS_FILE.read_text()


def test_shapes_follow_each_netlist(proj):
    s = {n: sorted(v) for n, v in {k: collect_signed_arith_shapes(v) for k, v in proj['nets'].items()}.items()}
    assert s['BalancedMmac'] == [('*', 8, 8, 16), ('+', 16, 16, 17), ('+', 17, 17, 18), ('+', 20, 18, 21)]
    assert s['LeftDeepMmac'] == [('*', 8, 8, 16), ('+', 16, 16, 17), ('+', 17, 16, 18), ('+', 18, 16, 19), ('+', 20, 19, 21)]
    assert s['CommutedMulMmac'] == s['LeftDeepMmac']         # same shapes, different netlist (operands swapped)


def test_interface_mismatch_reported_in_python(proj):
    bad = MatmulContract(**{**proj['cand'].CONTRACT.__dict__, 'c_width': 16})
    with pytest.raises(InterfaceMismatch, match=r'input c_0_0: netlist=\(20, True\) interface=\(16, True\)'):
        netlist_to_lean(proj['nets']['BalancedMmac'], options=LeanExportOptions(namespace='X', ports=bad.ports()))


def test_custom_verilog_component_is_refused():
    from spire import Component, IORecord, Input, Output, UInt
    from spire.component import CustomVerilogComponent
    class Odd(CustomVerilogComponent):
        def __init__(self):
            self.io = IORecord(a=Input(UInt(8)), y=Output(UInt(8))); self.elaborate()
        def elaborate(self): self.io.y <<= self.io.a
        def custom_verilog(self): return 'assign y = 8\'d0;'
    with pytest.raises(NotImplementedError, match='custom_verilog'):
        netlist_to_lean(Odd().to_netlist('odd'), options=LeanExportOptions(namespace='X'))


def test_candidates_match_math_in_simulation(proj):
    for n in ('BalancedMmac', 'LeftDeepMmac', 'CommutedMulMmac'):
        assert proj['cand'].run_sim(proj['cand'].CANDIDATES[n](), vectors=200, seed=1) == 200, n


def test_layered_and_delta_proofs_check(proj):
    """Both layers prove equivalence to Spec; both designs prove Spec.Correct — one via Mmac, one by delta."""
    assert 'Build completed successfully' in lake_build(proj['dir'])
    ax = check_theorems(proj['dir'], proj['checks'])
    assert set(ax) == {c.theorem for c in proj['checks']}
    assert all('sorryAx' not in a for a in ax.values())
    # the delta proof relies on Dleft_Proof (which relies on Mmac's lemmas): bv_decide certs come from there
    assert any('bv_decide' in a for a in ax[f"{proj['np']}.implements"])
    assert lean_imports(proj['files'][f"{proj['np']}.lean"]) == [proj['nc'], proj['lc'], proj['lp']]


def test_weakened_statement_is_rejected(proj, tmp_path):
    d = copy_project(proj['dir'], tmp_path / 'lean'); lp, lc = proj['lp'], proj['lc']
    t = (d / f'{lp}.lean').read_text(); i = t.index('\ntheorem implements') + 1
    (d / f'{lp}.lean').write_text(t[:i] + f'theorem implements : True := trivial\n\nend {lp}\n')
    lake_build(d, lp)
    with pytest.raises(ProofCheckError, match='does not establish'):
        check_theorems(d, [TheoremCheck(lp, f'{lp}.implements', f'Spec.Correct {lc}.eval')])


def test_forged_bv_decide_named_axiom_is_rejected(proj, tmp_path):
    d = copy_project(proj['dir'], tmp_path / 'lean'); lp, lc = proj['lp'], proj['lc']
    t = (d / f'{lp}.lean').read_text(); i = t.index('\ntheorem implements') + 1
    (d / f'{lp}.lean').write_text(t[:i] + 'axiom fake._native.bv_decide.ax_1_1 : False\n'
                                  f'theorem implements : Spec.Correct {lc}.eval := False.elim fake._native.bv_decide.ax_1_1\n\nend {lp}\n')
    lake_build(d, lp)
    with pytest.raises(ProofCheckError, match='disallowed axioms'):
        check_theorems(d, [TheoremCheck(lp, f'{lp}.implements', f'Spec.Correct {lc}.eval')])
