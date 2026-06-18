# pick_probe.py
import random
from math import ceil, log2

from spire.aig.aig_aigerverse import _get_aag_sym, conv_aag_into_aig
from spire.expr import UInt, mux, fit_width
from spire.aiger import AigerExporter, AigerImporter
from spire.component import Module
from spire.simulator import Simulator
from spire.component import IOCollector




# ---------------- naive + robust pick ----------------
def pick_from_vec_naive(vec_bits, idx_expr):
    acc = 0
    for k, x in enumerate(vec_bits):
        acc = mux(idx_expr == k, x, acc)
    return acc

def _eq_const_bits(x, k: int, w: int):
    """Unsigned equality x == k at fixed width w (XNOR+AND only)."""
    xw = fit_width(x, UInt(w))
    acc = 1
    for b in range(w):
        kb = 1 if ((k >> b) & 1) else 0
        acc = acc & ~(xw[b] ^ kb)  # XNOR
    return acc

def pick_from_vec_robust(vec_bits, idx_expr, w_idx: int):
    """One-hot decode on idx (0..len(vec_bits)-1). Out-of-range -> 0."""
    decs = [_eq_const_bits(idx_expr, k, w_idx) for k in range(len(vec_bits))]
    acc = 0
    for k, x in enumerate(vec_bits):
        acc = acc | (decs[k] & x)
    return acc


# ---------------- DUT: exposes both versions ----------------
def build_pick_probe(W: int):
    """
    Ports:
      vec[W-1:0], idx[IW-1:0]
      naive_idx, naive_idxm1, naive_idxm2
      robust_idx, robust_idxm1, robust_idxm2
    """
    IW = max(1, ceil(log2(W + 2)))  # enough to represent 0..W+1 safely
    m = Module(f"PickProbe_W{W}", with_clock=False, with_reset=False)

    vec = m.input(UInt(W), "vec")
    idx = m.input(UInt(IW), "idx")

    bits = [vec[i] for i in range(W)]
    idx_w   = fit_width(idx, UInt(IW))
    idxm1_w = fit_width(idx - 1, UInt(IW))   # wrap happens; robust will handle
    idxm2_w = fit_width(idx - 2, UInt(IW))

    # naive
    n_idx  = m.output(UInt(1), "naive_idx");   n_idx  <<= pick_from_vec_naive(bits, idx_w)
    n_i1   = m.output(UInt(1), "naive_idxm1"); n_i1   <<= pick_from_vec_naive(bits, idxm1_w)
    n_i2   = m.output(UInt(1), "naive_idxm2"); n_i2   <<= pick_from_vec_naive(bits, idxm2_w)

    # robust
    r_idx  = m.output(UInt(1), "robust_idx");  r_idx  <<= pick_from_vec_robust(bits, idx_w, IW)
    r_i1   = m.output(UInt(1), "robust_idxm1");r_i1   <<= pick_from_vec_robust(bits, idxm1_w, IW)
    r_i2   = m.output(UInt(1), "robust_idxm2");r_i2   <<= pick_from_vec_robust(bits, idxm2_w, IW)

    return m


# ---------------- Python reference ----------------
def ref_pick(vec_int: int, idx_val: int, W: int) -> int:
    """Return vec[idx] if 0<=idx<W else 0. vec is LSB-first bits in vec_int."""
    if 0 <= idx_val < W:
        return (vec_int >> idx_val) & 1
    return 0


# ---------------- randomized test ----------------
def run_random(W=11, trials=5000, seed=123):
    m = build_pick_probe(W)
    
    
    # build m2
    aag = AigerExporter(m).get_aag()
    aig = conv_aag_into_aig(aag)

    aag_sym = _get_aag_sym(aag)
    m2 = AigerImporter(aag[:-2]+aag_sym).get_spire_module()

    collector = IOCollector()
    collector.group(m2, {
        "vec": UInt(W),
        "idx": UInt(max(1, ceil(log2(W + 2)))),
        "naive_idx": UInt(1),
        "naive_idxm1": UInt(1),
        "naive_idxm2": UInt(1),
        "robust_idx": UInt(1),
        "robust_idxm1": UInt(1),
        "robust_idxm2": UInt(1),
    })
        
    
    sim = Simulator(m2)

    random.seed(seed)
    IW = max(1, ceil(log2(W + 2)))
    failures = 0
    naive_bad = 0
    robust_bad = 0

    for t in range(trials):
        vec_int = random.getrandbits(W)
        idx_val = random.randint(0, W + 1)  # include out-of-range cases

        sim.set("vec", vec_int).set("idx", idx_val).eval()

        # grab DUT
        got = {
            "n_idx":  sim.get("naive_idx"),
            "n_i1":   sim.get("naive_idxm1"),
            "n_i2":   sim.get("naive_idxm2"),
            "r_idx":  sim.get("robust_idx"),
            "r_i1":   sim.get("robust_idxm1"),
            "r_i2":   sim.get("robust_idxm2"),
        }

        # refs (careful with idx-1/idx-2 underflow → out-of-range → 0)
        ref = {
            "r_idx": ref_pick(vec_int, idx_val,     W),
            "r_i1":  ref_pick(vec_int, idx_val - 1, W),
            "r_i2":  ref_pick(vec_int, idx_val - 2, W),
        }

        # robust should match ref exactly
        ok_robust = (got["r_idx"] == ref["r_idx"] and
                     got["r_i1"]  == ref["r_i1"]  and
                     got["r_i2"]  == ref["r_i2"])

        # naive should also match ref; if not, we flag it
        ok_naive = (got["n_idx"] == ref["r_idx"] and
                    got["n_i1"]  == ref["r_i1"]  and
                    got["n_i2"]  == ref["r_i2"])

        if not ok_robust or not ok_naive:
            failures += 1
            if not ok_naive:
                naive_bad += 1
            if not ok_robust:
                robust_bad += 1
            print(f"FAIL t={t}  W={W}  vec=0b{vec_int:0{W}b}  idx={idx_val}")
            print(f"   ref: idx={ref['r_idx']}  i-1={ref['r_i1']}  i-2={ref['r_i2']}")
            print(f"   got: n  ={got['n_idx']},{got['n_i1']},{got['n_i2']}   r={got['r_idx']},{got['r_i1']},{got['r_i2']}")
            # uncomment to stop at first fail
            # break

    print(f"Summary: {trials-failures}/{trials} trials had no mismatch; "
          f"naive mismatches: {naive_bad}/{trials}  "
          f"robust mismatches: {robust_bad}/{trials}")

if __name__ == "__main__":
    # W = FW+1 for your rounding cone; e.g. FW=10 -> W=11
    run_random(W=11, trials=5000, seed=42)