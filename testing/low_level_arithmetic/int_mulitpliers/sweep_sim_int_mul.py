import random
import time
from typing import Callable, Dict, List, Optional, Tuple

from aigverse import aig_cut_rewriting, aig_resubstitution, sop_refactoring
import numpy as np
from testing.low_level_arithmetic.compressor_tree.compressor_tree_spire_hdl import gen_compressor_tree_graph_and_spirehdl_module
from spirehdl.aig.aig_aigerverse import _get_aag_sym, conv_aag_into_aig, conv_aig_into_aag
from spirehdl.helpers import optimize_aag, run_vectors
from spirehdl.spirehdl import Bool, Concat, Const, Expr, Op2, SInt, UInt
from spirehdl.spirehdl_aiger import AigerExporter, AigerImporter
from spirehdl.spirehdl_module import IOCollector
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_simulator import Simulator


import matplotlib.pyplot as plt


def run_vectors_io_log(
    m: Module, vectors: List[Tuple[str, Dict[str, int], Dict[str, int]]], *, 
    decoder: Callable[[int], float] | None = None, exprs: List[Expr] = [],
    use_signed: bool = False,
) -> None:
    """
    Generic runner:
      vectors: list of (label, inputs{name->int}, expected{name->int})
    Prints mismatches; raises AssertionError at the end if any failed.
    """

    states_list = []

    sim = Simulator(m)
    fails = 0
    for name, ins, outs in vectors:
        for k, v in ins.items():
            sim.set(k, v)
        sim.eval()
        bad = []
        for oname, exp in outs.items():
            got_raw = sim.peek(oname) #sim.get(oname)
            got_signed = sim.get(oname)
            got = got_signed if use_signed else got_raw
            if got != exp:
                if decoder and oname == "y":
                    bad.append(f"{oname}: got=0x{got:0X} ({decoder(got):.8g})  exp=0x{exp:0X} ({decoder(exp):.8g})")
                else:
                    bad.append(f"{oname}: got=0x{got:0X}  exp=0x{exp:0X}")
            else:
                if decoder and oname == "y":
                    # print(f"PASS {name}: {oname}=0x{got:0X} ({decoder(got):.8g})")
                    pass
                else:
                    # print(f"PASS {name}: {oname}=0x{got:0X} ({got})")
                    pass
        if bad:
            fails += 1
            print(f"FAIL  {name}:  " + " | ".join(bad))
        state = sim._get_expr_snapshot(exprs)
        # convert expr to id
        state = [(id(e), v) for e, v in state]
        # and convert to dict for easy comparison
        state = dict(state)
        states_list.append(state)
    if fails:
        raise AssertionError(f"{fails}/{len(vectors)} vectors failed")

    print(f"Number of vectors: {len(vectors)}, {fails} failures")

    return states_list


def build_multiplier(W: int = 8, tb_sigma: Optional[float] = None, n_vecs: int = 64) -> Tuple[Module, Dict[str, UInt], List]:
    m = Module(f"Mul{W}", with_clock=False, with_reset=False)
    a = m.input(UInt(W), "a")
    b = m.input(UInt(W), "b")
    y = m.output(UInt(2*W), "y")
    y <<= a * b
    vecs = []
    for _ in range(n_vecs):
        if tb_sigma is not None:
            va = int(np.round((np.random.normal(1 << (W-1), tb_sigma))))
            vb = int(np.round((np.random.normal(1 << (W-1), tb_sigma))))
            # clamp to range
            va = max(min(va, (1 << W) - 1), 0)
            vb = max(min(vb, (1 << W) - 1), 0)
        else:
            va = random.getrandbits(W)
            vb = random.getrandbits(W)
        vecs.append((f"{va}*{vb}", {"a": va, "b": vb}, {"y": (va * vb) & ((1 << (2*W)) - 1)}))
    spec = {"a": UInt(W), "b": UInt(W), "y": UInt(2*W)}
    return m, spec, vecs, None

def build_multiplier_from_compressor_tree(W: int = 8, tb_sigma: Optional[float] = None, n_vecs: int = 64) -> Tuple[Module, Dict[str, UInt], List]:
    #m = Module(f"Mul{W}", with_clock=False, with_reset=False)
    #a = m.input(UInt(W), "a")
    #b = m.input(UInt(W), "b")
    #y = m.output(UInt(2*W), "y")
    #y <<= a * b
    #
    g, m = gen_compressor_tree_graph_and_spirehdl_module(W, policy="wallace", name=f"Mul{W}_ct")
    vecs = []
    for _ in range(n_vecs):
        if tb_sigma is not None:
            va = int(np.round((np.random.normal(1 << (W-1), tb_sigma))))
            vb = int(np.round((np.random.normal(1 << (W-1), tb_sigma))))
            # clamp to range
            va = max(min(va, (1 << W) - 1), 0)
            vb = max(min(vb, (1 << W) - 1), 0)
        else:
            va = random.getrandbits(W)
            vb = random.getrandbits(W)
        vecs.append((f"{va}*{vb}", {"a": va, "b": vb}, {"y": (va * vb) & ((1 << (2*W)) - 1)}))
    spec = {"a": UInt(W), "b": UInt(W), "y": UInt(2*W)}
    return m, spec, vecs, None


def build_signed_multiplier(W: int = 8, tb_sigma: Optional[float]= None, n_vecs: int = 64) -> Tuple[Module, Dict[str, UInt], List]:
    m = Module(f"SMul{W}", with_clock=False, with_reset=False)
    a = m.input(SInt(W), "a")
    b = m.input(SInt(W), "b")
    y = m.output(SInt(2*W), "y")
    y <<= a * b
    vecs = []
    for _ in range(n_vecs):
        if tb_sigma is not None:
            va = int(np.round((np.random.normal(0, tb_sigma))))
            vb = int(np.round((np.random.normal(0, tb_sigma))))
            # clamp to range
            va = max(min(va, (1 << (W-1)) - 1), -(1 << (W-1)))
            vb = max(min(vb, (1 << (W-1)) - 1), -(1 << (W-1)))
            #va = -1 if random.random() < 0.5 else 0
            #vb = -1 if random.random() < 0.5 else 0
        else:
            va = random.getrandbits(W) - (1 << (W-1))
            vb = random.getrandbits(W) - (1 << (W-1))

        vecs.append((f"{va}*{vb}", {"a": va, "b": vb}, {"y": va * vb}))
    spec = {"a": SInt(W), "b": SInt(W), "y": SInt(2*W)}
    decoder = None
    return m, spec, vecs, decoder


# input in sign magnitude, output in two's complement
def build_signed_multiplier_sign_magnitude(W: int = 8, tb_sigma: Optional[float] = None, n_vecs: int = 64) -> Tuple[Module, Dict[str, UInt], List]:
    m = Module(f"SMulSM{W}", with_clock=False, with_reset=False)
    a = m.input(UInt(W), "a")  # sign-magnitude
    b = m.input(UInt(W), "b")  # sign-magnitude
    y = m.output(UInt(2*W), "y") # two's complement
    sa = a[W-1]
    sb = b[W-1]
    mag_a = a[0:W-1]  # make magnitude unsigned
    mag_b = b[0:W-1]  # make magnitude unsigned
    mag_y = mag_a * mag_b
    sy = sa ^ sb
    y <<= Concat([mag_y, Const(False, Bool()), sy])  # sign + magnitude (drop overflow bit)
    vecs = []
    for _ in range(n_vecs):
        abs_max = (1 << (W - 1)) - 1
        if tb_sigma is not None:
            va = int(np.round((np.random.normal(0, tb_sigma))))
            vb = int(np.round((np.random.normal(0, tb_sigma))))
            # clamp to range
            va = max(min(va, abs_max), -abs_max)
            vb = max(min(vb, abs_max), -abs_max)
            #  randomly choose between 0 and -1
            # va = -1 if random.random() < 0.5 else 0
            # vb = -1 if random.random() < 0.5 else 0
        else:
            va = random.getrandbits(W) - (1 << (W - 1))
            vb = random.getrandbits(W) - (1 << (W - 1))
            va = max(min(va, abs_max), -abs_max)
            vb = max(min(vb, abs_max), -abs_max)

        sa = 1 if va >= 0 else 0 # (va >> (W-1)) & 1
        sb = 1 if vb >= 0 else 0 # (vb >> (W-1)) & 1
        mag_a = int(np.abs(va)) # va & ((1 << (W-1)) - 1)
        mag_b = int(np.abs(vb)) # vb & ((1 << (W-1)) - 1)
        va_sm = (sa << (W-1)) | (mag_a & ((1 << (W-1)) - 1))
        vb_sm = (sb << (W-1)) | (mag_b & ((1 << (W-1)) - 1))
        sy = sa ^ sb
        mag_y = mag_a * mag_b
        vy_sm = (sy << (2*W-1)) | (mag_y & ((1 << (2*W-1)) - 1))

        vecs.append((f"{va}*{vb}", {"a": va_sm, "b": vb_sm}, {"y": vy_sm}))
    spec = {"a": UInt(W), "b": UInt(W), "y": UInt(2*W)}
    return m, spec, vecs, None


def main():

    optim=True
    n_vecs = 5000//100

    sigma_factor = 0.5
    # sigmas = [1, 2]
    n_bits_vec = [4, 8, 12, 16]

    for n_bits in n_bits_vec:

        #sigmas = list(range(1, 9))
        n_steps = 8
        sigma_max = 2**n_bits * sigma_factor
        sigma_start = sigma_max / n_steps
        sigmas = np.linspace(sigma_start, sigma_max, n_steps)
        results = {}

        for builder_f in [build_multiplier, build_multiplier_from_compressor_tree, build_signed_multiplier, build_signed_multiplier_sign_magnitude]:

            t0 = time.time()

            # m, spec, vecs, dec = gen_m_case(3)

            # builder_f = build_multiplier
            # builder_f = build_signed_multiplier
            # builder_f = build_signed_multiplier_sign_magnitude

            # m, spec, vecs, dec = build_multiplier(4)
            m, spec, vecs, dec = builder_f(n_bits)

            print(f"\n=== {m.name} ===")

            # 1) Original sim
            print("Sim (original) …")
            run_vectors(m, vecs, decoder=dec, use_signed=True)

            # get AIG
            aag = AigerExporter(m).get_aag()
            if optim:
                aag = optimize_aag(aag, n_iter_optimizations=10)
            m_aig = AigerImporter(aag).get_spirehdl_module()
            IOCollector().group(m_aig, spec) # regroup I/Os to match original port widths

            # AIG network test sim
            print("Sim (AIG) …")
            run_vectors(m_aig, vecs, decoder=dec, use_signed=True)

            exprs = m_aig.all_exprs()

            all_ands = [e for e in exprs if isinstance(e, Op2) and e.op == "&"]

            def run_and_count(vecs_run) -> int:

                # if vecs_diff is not None:
                #    vecs = vecs_diff

                states_list = run_vectors_io_log(m_aig, vecs_run, decoder=dec, exprs=all_ands, use_signed=True)

                # get all ids from step 0
                ids = set(states_list[0].keys())
                # count number of switches per id
                switches = {i: 0 for i in ids}
                for i in ids:
                    last = states_list[0][i]
                    for s in states_list[1:]:
                        if s[i] != last:
                            switches[i] += 1
                            last = s[i]

                # sum up all switches
                total_switches = sum(switches.values())
                return total_switches / len(states_list)  # average per vector

            # switches = run_and_count(vecs)
            # print(f"Average AND switches: {switches}")

            switches = []
            for sigma in sigmas:
                _, _, vecs, _ = builder_f(n_bits, tb_sigma=sigma, n_vecs=n_vecs)
                switches.append(run_and_count(vecs))
                print(f"Average AND switches (sigma={sigma}): {switches[-1]}")

            print("Sigma vs AND switches:")
            for sigma, change in zip(sigmas, switches):
                print(f"{sigma:.2f} {change}")

            t_end = time.time()
            print(f"Time: {t_end - t0:.2f} seconds")

            results[m.name] = (sigmas, switches)

        # plot
        plt.figure(figsize=(8, 6))
        for m_name, (sigmas, switches) in results.items():       
            plt.plot(sigmas, switches, marker="o", label=m_name)
        plt.legend()
        plt.xlabel("Input sigma")
        plt.ylabel("Average AND switches")
        plt.title(f"AND switches in {m.name} vs input sigma")
        plt.grid()
        plt.savefig(f"switches_all_{n_bits}.png")


if __name__ == "__main__":
    main()