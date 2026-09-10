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


# --- operator coverage ---------------------------------------------------------------------------

def _lean_eval(project: Path, module: str, inputs: dict, outputs: list, vectors: list) -> list:
    """Evaluate `module.eval` on each input vector via `#eval`; returns [[toNat per output]] in order."""
    import re, subprocess
    from spire.design_db.lean.bridge import find_lake
    lines = [f'import {module}', 'open Interface']
    for vec in vectors:
        lit = ', '.join(f'{n} := BitVec.ofNat {w} {vec[n]}' for n, w in inputs.items())
        lines.append(f'#eval [{", ".join(f"({module}.eval {{ {lit} }}).{o}.toNat" for o in outputs)}]')
    (project / '_Eval.lean').write_text('\n'.join(lines) + '\n')
    out = subprocess.run([find_lake(), 'env', 'lean', '_Eval.lean'], cwd=project, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return [[int(x) for x in re.findall(r'\d+', l)] for l in out.stdout.splitlines() if l.startswith('[')]


def test_operator_semantics_match_simulator(tmp_path):
    """Differential test of SpireSemantics + translator against spire.simulator on every supported operator."""
    import random
    from spire.simulator import Simulator
    from lean_matmul_helper import opzoo_component
    core = opzoo_component(); net = core.to_netlist('zoo')
    ports = netlist_ports(net)
    ins = {p['name']: p['width'] for p in ports if p['dir'] == 'input'}
    outs = [p['name'] for p in ports if p['dir'] == 'output']; outw = {p['name']: p['width'] for p in ports}
    files = {'Interface.lean': interface_to_lean(ports),
             'OpZooCircuit.lean': netlist_to_lean(net, options=LeanExportOptions(namespace='OpZooCircuit', ports=ports))}
    for n, t in files.items(): (tmp_path / n).write_text(t)
    write_lean_project(tmp_path, ['Interface', 'OpZooCircuit'])
    lake_build(tmp_path, 'OpZooCircuit')
    rng = random.Random(7); corners = [0, 1, 0x7F, 0x80, 0xFF]
    vectors = [{n: rng.choice(corners + [rng.randrange(1 << w)]) & ((1 << w) - 1) for n, w in ins.items()} for _ in range(40)]
    sim = Simulator(net); expected = []
    for vec in vectors:
        for n, v in vec.items(): sim.set(getattr(core.io, n), v)
        sim.eval()
        expected.append([sim.get(getattr(core.io, o)) & ((1 << outw[o]) - 1) for o in outs])
    got = _lean_eval(tmp_path, 'OpZooCircuit', ins, outs, vectors)
    assert len(got) == len(vectors)
    for vec, e, g in zip(vectors, expected, got):
        assert e == g, {o: (a, b) for o, a, b in zip(outs, e, g) if a != b} | {'inputs': vec}


def test_whole_design_bv_decide_proof_is_accepted(tmp_path):
    """A bitwise rewrite proven by SAT on the whole (small) design — allowed; the certificate hangs off `implements`."""
    from lean_matmul_helper import bits_components
    golden, alt = bits_components()
    gnet, anet = golden.to_netlist('g'), alt.to_netlist('a'); ports = netlist_ports(gnet)
    circ, proof = design_modules('alt')
    files = {'Interface.lean': interface_to_lean(ports), 'Spec.lean': default_spec_lean(),
             'GoldenCircuit.lean': netlist_to_lean(gnet, options=LeanExportOptions(namespace='GoldenCircuit', ports=ports)),
             f'{circ}.lean': netlist_to_lean(anet, options=LeanExportOptions(namespace=circ, ports=ports)),
             f'{proof}.lean': f'''import {circ}
import Spec
import Std.Tactic
open SpireSemantics Interface
namespace {proof}
theorem implements : Spec.Correct {circ}.eval := by
  intro i
  apply Outputs.ext <;> (simp only [{circ}.eval, GoldenCircuit.eval, bitAnd, bitOr, bitXor, bitNot]; bv_decide)
end {proof}
'''}
    for n, t in files.items(): (tmp_path / n).write_text(t)
    write_lean_project(tmp_path, [n[:-5] for n in files])
    lake_build(tmp_path, proof)
    ax = check_theorems(tmp_path, [TheoremCheck(proof, f'{proof}.implements', f'Spec.Correct {circ}.eval')])
    assert any(a.startswith(f'{proof}.implements.') and 'bv_decide' in a for a in ax[f'{proof}.implements'])


# --- sequential ------------------------------------------------------------------------------------

def _sim_trace(net, cycles, out_name, mask):
    from spire.simulator import Simulator
    io = net.component.io
    sim = Simulator(net); sim.reset(True); sim.deassert_reset(); trace = []
    for c in cycles:
        for n, v in c.items(): sim.set(getattr(io, n), v)
        sim.eval(); trace.append(sim.get(getattr(io, out_name)) & mask); sim.step()
    return trace


@pytest.fixture(scope='module')
def seq(tmp_path_factory):
    """MAC accumulator: golden, swap (same state), split (hi/lo halves), retimed (extra latency)."""
    from lean_matmul_helper import mac_components, machine_proof_lean
    from spire.design_db.lean.bridge import default_spec_lean, is_sequential
    d = tmp_path_factory.mktemp('seq'); nets = {n: c() for n, c in mac_components().items()}
    ports = netlist_ports(nets['golden'])
    assert all(is_sequential(n) for n in nets.values()) and {p['name'] for p in ports} == {'a', 'b', 'en', 'clear', 'y'}
    circ = lambda net, ns: netlist_to_lean(net, options=LeanExportOptions(namespace=ns, ports=ports))
    files = {'Interface.lean': interface_to_lean(ports), 'GoldenCircuit.lean': circ(nets['golden'], 'GoldenCircuit'),
             'Spec.lean': default_spec_lean(sequential=True)}
    mods = {n: design_modules(n) for n in ('swap', 'split', 'retimed')}
    for n, (c, _p) in mods.items(): files[f'{c}.lean'] = circ(nets[n], c)
    files[f"{mods['swap'][1]}.lean"] = machine_proof_lean(circuit_module=mods['swap'][0], proof_module=mods['swap'][1], relation='t.acc = s.acc')
    files[f"{mods['split'][1]}.lean"] = machine_proof_lean(circuit_module=mods['split'][0], proof_module=mods['split'][1], relation='t.hi ++ t.lo = s.acc')
    for n, t in files.items(): (d / n).write_text(t)
    write_lean_project(d, [n[:-5] for n in files])
    return dict(dir=d, nets=nets, ports=ports, mods=mods, files=files)


def test_sequential_translation_shape(seq):
    g = seq['files']['GoldenCircuit.lean']
    for piece in ('@[ext] structure State where', 'acc : BitVec 24', 'def init : State', 'def next (s : State) (i : Inputs) : State',
                  'def out (s : State) (i : Inputs) : Outputs', 'def machine : Machine Inputs Outputs State'):
        assert piece in g
    assert 'clk' not in seq['files']['Interface.lean'] and 'rst' not in seq['files']['Interface.lean']
    assert 'm.trace tr = GoldenCircuit.machine.trace tr' in seq['files']['Spec.lean']
    assert 'hi : BitVec 12' in seq['files'][f"{seq['mods']['split'][0]}.lean"]


def test_sequential_trace_matches_simulator(seq):
    """Lean `machine.trace` equals the simulator (reset, then per cycle: inputs → outputs → edge) on random cycles."""
    import random, re, subprocess
    from spire.design_db.lean.bridge import find_lake
    ins = [(p['name'], p['width']) for p in seq['ports'] if p['dir'] == 'input']
    rng = random.Random(3); cycles = [{n: rng.randrange(1 << w) for n, w in ins} for _ in range(40)]
    for c in cycles[:3]: c['clear'] = 0
    lake_build(seq['dir'], 'GoldenCircuit')
    lit = ', '.join('{ ' + ', '.join(f'{n} := BitVec.ofNat {w} {c[n]}' for n, w in ins) + ' }' for c in cycles)
    (seq['dir'] / '_Trace.lean').write_text(f'import GoldenCircuit\nopen Interface\n#eval (GoldenCircuit.machine.trace [{lit}]).map (fun o => o.y.toNat)\n')
    out = subprocess.run([find_lake(), 'env', 'lean', '_Trace.lean'], cwd=seq['dir'], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert [int(x) for x in re.findall(r'\d+', out.stdout)] == _sim_trace(seq['nets']['golden'], cycles, 'y', 0xFFFFFF)


def test_sequential_relation_proofs_check(seq):
    """swap: state equality; split: `hi ++ lo = acc`. Three SAT-closed local lemmas + Machine.trace_eq_of_rel each."""
    assert 'Build completed successfully' in lake_build(seq['dir'])
    checks = [TheoremCheck(p, f'{p}.implements', f'Spec.Correct {c}.machine') for n, (c, p) in seq['mods'].items() if n != 'retimed']
    ax = check_theorems(seq['dir'], checks)
    assert all('sorryAx' not in a and any('bv_decide' in x for x in a) for a in ax.values())


def test_retimed_design_is_not_provable_and_differs_in_simulation(seq):
    """Extra latency: the trace differs, and the same relation recipe yields a counterexample (not a proof)."""
    import random, subprocess
    from lean_matmul_helper import machine_proof_lean
    from spire.design_db.lean.bridge import LeanBuildError, find_lake
    ins = [(p['name'], p['width']) for p in seq['ports'] if p['dir'] == 'input']
    rng = random.Random(5); cycles = [{n: rng.randrange(1 << w) for n, w in ins} for _ in range(20)]
    for c in cycles: c['clear'] = 0; c['en'] = 1
    assert _sim_trace(seq['nets']['golden'], cycles, 'y', 0xFFFFFF) != _sim_trace(seq['nets']['retimed'], cycles, 'y', 0xFFFFFF)
    c, p = seq['mods']['retimed']
    (seq['dir'] / f'{p}.lean').write_text(machine_proof_lean(circuit_module=c, proof_module=p, relation='t.acc = s.acc'))
    out = subprocess.run([find_lake(), 'env', 'lean', f'{p}.lean'], cwd=seq['dir'], capture_output=True, text=True)
    assert out.returncode != 0 and 'counterexample' in out.stdout + out.stderr
