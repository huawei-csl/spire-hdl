# pick_diag.py
import random
from math import ceil, log2

from spire.aig.aig_aigerverse import _get_aag_sym, conv_aag_into_aig
from spire.expr import UInt, mux, fit_width
from spire.aiger import AigerExporter, AigerImporter
from spire.component import Netlist
from spire.simulator import Simulator
from spire.component import IOCollector

def _eq_const_bits(x, k: int, w: int):
    xw = fit_width(x, UInt(w))
    acc = 1  # Bool(1)
    for b in range(w):
        kb = 1 if ((k >> b) & 1) else 0
        acc = acc & ~(xw[b] ^ kb)  # XNOR
    return acc

def build_pick_diag(W: int):
    """
    Inputs:
      vec : UInt(W)   (LSB we *expect* at bit 0)
      idx : UInt(IW)

    Outputs:
      tap[k] : each vec[k] exposed as a separate 1-bit output
      dec_idx[k], dec_i1[k], dec_i2[k] : one-hot decodes of idx, idx-1, idx-2 (robust)
      pick_naive_{idx,i1,i2} : naive pick_from_vec
      pick_robust_{idx,i1,i2}: robust pick_from_vec
    """
    IW = max(1, ceil(log2(W + 2)))
    m = Netlist(f"PickDiag_W{W}", with_clock=False, with_reset=False)

    vec = m.input(UInt(W), "vec")
    idx = m.input(UInt(IW), "idx")

    # Expose every vec[k] as its own 1-bit output (to check indexing!)
    taps = []
    for k in range(W):
        t = m.output(UInt(1), f"tap_{k}")
        t <<= vec[k]
        taps.append(t)

    # Robust one-hot decoders for idx, idx-1, idx-2
    idx_w   = fit_width(idx, UInt(IW))
    idxm1_w = fit_width(idx - 1, UInt(IW))
    idxm2_w = fit_width(idx - 2, UInt(IW))

    dec_idx = []
    dec_i1  = []
    dec_i2  = []
    for k in range(W):
        d0 = m.output(UInt(1), f"dec_idx_{k}"); d0 <<= _eq_const_bits(idx_w,   k, IW); dec_idx.append(d0)
        d1 = m.output(UInt(1), f"dec_i1_{k}");  d1 <<= _eq_const_bits(idxm1_w, k, IW); dec_i1.append(d1)
        d2 = m.output(UInt(1), f"dec_i2_{k}");  d2 <<= _eq_const_bits(idxm2_w, k, IW); dec_i2.append(d2)

    # Naive picks
    def pick_naive(bits, sel):
        acc = 0
        for k, x in enumerate(bits):
            acc = mux(sel == k, x, acc)
        return acc

    # Robust picks (AND( onehot , bit ) → OR)
    def pick_robust(bits, decs):
        acc = 0
        for k, x in enumerate(bits):
            acc = acc | (decs[k] & x)
        return acc

    n_idx  = m.output(UInt(1), "pick_naive_idx");  n_idx  <<= pick_naive([vec[k] for k in range(W)], idx_w)
    n_i1   = m.output(UInt(1), "pick_naive_i1");   n_i1   <<= pick_naive([vec[k] for k in range(W)], idxm1_w)
    n_i2   = m.output(UInt(1), "pick_naive_i2");   n_i2   <<= pick_naive([vec[k] for k in range(W)], idxm2_w)

    r_idx  = m.output(UInt(1), "pick_robust_idx"); r_idx  <<= pick_robust([vec[k] for k in range(W)], dec_idx)
    r_i1   = m.output(UInt(1), "pick_robust_i1");  r_i1   <<= pick_robust([vec[k] for k in range(W)], dec_i1)
    r_i2   = m.output(UInt(1), "pick_robust_i2");  r_i2   <<= pick_robust([vec[k] for k in range(W)], dec_i2)

    return m

# -------- Python reference helpers --------
def bit(vec_int: int, k: int) -> int:
    return (vec_int >> k) & 1 if k >= 0 else 0

def onehot_ref(idx_val: int, W: int):
    # 1 at idx if 0<=idx<W else all-zeros
    if 0 <= idx_val < W:
        return 1 << idx_val
    return 0

def run_diag(W=11, trials=200, seed=42):
    m = build_pick_diag(W)
    sim = Simulator(m)

    import random
    random.seed(seed)
    IW = max(1, ceil(log2(W + 2)))

    for t in range(trials):
        vec_int = random.getrandbits(W)
        idx_val = random.randint(0, W + 1)  # include out-of-range

        sim.set("vec", vec_int).set("idx", idx_val).eval()

        # 1) Check taps: compare every tap_k against Python (LSB=k)
        bad_taps = []
        taps = []
        for k in range(W):
            tk = sim.get(f"tap_{k}")
            taps.append(tk)
            if tk != bit(vec_int, k):
                bad_taps.append((k, tk, bit(vec_int,k)))

        # 2) Check one-hots: (idx, idx-1, idx-2)
        def read_onehot(prefix):
            val = 0
            for k in range(W):
                if sim.get(f"{prefix}_{k}"):
                    val |= (1 << k)
            return val

        oh_idx = read_onehot("dec_idx")
        oh_i1  = read_onehot("dec_i1")
        oh_i2  = read_onehot("dec_i2")

        ref_oh_idx = onehot_ref(idx_val,     W)
        ref_oh_i1  = onehot_ref(idx_val - 1, W)
        ref_oh_i2  = onehot_ref(idx_val - 2, W)

        # 3) Picks (naive/robust) vs Python
        n_idx = sim.get("pick_naive_idx");  r_idx = sim.get("pick_robust_idx")
        n_i1  = sim.get("pick_naive_i1");   r_i1  = sim.get("pick_robust_i1")
        n_i2  = sim.get("pick_naive_i2");   r_i2  = sim.get("pick_robust_i2")

        ref_idx = bit(vec_int, idx_val)
        ref_i1  = bit(vec_int, idx_val - 1)
        ref_i2  = bit(vec_int, idx_val - 2)

        ok = (oh_idx == ref_oh_idx and oh_i1 == ref_oh_i1 and oh_i2 == ref_oh_i2 and
              n_idx == ref_idx and n_i1 == ref_i1 and n_i2 == ref_i2 and
              r_idx == ref_idx and r_i1 == ref_i1 and r_i2 == ref_i2 and
              not bad_taps)

        if not ok:
            print(f"\nFAIL t={t} W={W} vec=0b{vec_int:0{W}b} idx={idx_val}")
            if bad_taps:
                print("  Tap mismatches (k, got, exp):", bad_taps)
                # If taps are wrong, everything downstream will be wrong too.
            if oh_idx != ref_oh_idx or oh_i1 != ref_oh_i1 or oh_i2 != ref_oh_i2:
                print(f"  One-hot dec idx : got 0b{oh_idx:0{W}b} exp 0b{ref_oh_idx:0{W}b}")
                print(f"  One-hot dec i-1: got 0b{oh_i1:0{W}b} exp 0b{ref_oh_i1:0{W}b}")
                print(f"  One-hot dec i-2: got 0b{oh_i2:0{W}b} exp 0b{ref_oh_i2:0{W}b}")
            print(f"  Picks naive: idx={n_idx} i-1={n_i1} i-2={n_i2} | robust: idx={r_idx} i-1={r_i1} i-2={r_i2}")
            print(f"  Picks  ref : idx={ref_idx} i-1={ref_i1} i-2={ref_i2}")
            # stop at first fail
            break
    else:
        print(f"All {trials} trials passed.")

if __name__ == "__main__":
    run_diag(W=11, trials=5000, seed=123)