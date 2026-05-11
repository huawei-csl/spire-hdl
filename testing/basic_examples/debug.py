# round_probe_test.py
import random
from math import ceil, log2

from spirehdl.aig.aig_aigerverse import _get_aag_sym, conv_aag_into_aig
from spirehdl.spirehdl import Const, UInt, mux


from spirehdl.spirehdl_aiger import AigerExporter, AigerImporter
from spirehdl.spirehdl_simulator import Simulator
from spirehdl.spirehdl_module import Module
from spirehdl.spirehdl_module import IOCollector


def build_round_probe(FW: int) -> Module:
    """
    Build a tiny module that computes:
      sig_shiftN = sig_pre >> shift_amt
      frac_trunc = sig_shiftN[FW-1:0]
      guard_s = bit_at(sig_pre, shift_amt-1)
      sticky_s = OR(sig_pre[0 .. shift_amt-2])
      lsb_s = frac_trunc[0]
      round_up_s = guard_s & (sticky_s | lsb_s)
    Ports:
      - sig_pre : UInt(FW+1)
      - sh      : UInt(ceil_log2(FW+2))
      - lsb_s, guard_s, sticky_s, round_up_s : 1-bit outputs
      - (debug) frac_trunc : UInt(FW), sig_shiftN : UInt(FW+1)
    """
    W = FW + 1
    SAW = max(1, ceil(log2(FW + 2)))  # need to represent 0..FW+1

    m = Module(f"RoundProbe_F{FW}", with_clock=False, with_reset=False)
    sig_pre = m.input(UInt(W), "sig_pre")
    sh = m.input(UInt(SAW), "sh")

    # variable right shift, then slice FW bits (LSBs)
    sig_shiftN = m.output(UInt(W), "sig_shiftN")
    sig_shiftN <<= sig_pre >> sh

    frac_trunc = m.output(UInt(FW), "frac_trunc")
    frac_trunc <<= sig_shiftN[FW - 1 : 0]

    # prefix ORs of sig_pre (LSB upward)
    pref = []
    acc = sig_pre[0]
    pref.append(acc)
    for i in range(1, W):
        acc = acc | sig_pre[i]
        pref.append(acc)

    # helpers: pick element by dynamic index (using equality + mux chain)
    def pick_from_vec(vec, idx_expr):
        acc = Const(0, UInt(1))
        for k, x in enumerate(vec):
            #acc = mux(idx_expr == k, x, acc)
            acc = mux(idx_expr == Const(k, UInt(5)), x, acc)
        return acc

    # guard = bit (sh - 1) of sig_pre  (0 if sh-1 outside [0..W-1])
    guard_sel = sh - 1
    guard_s = m.output(UInt(1), "guard_s")
    guard_s <<= pick_from_vec([sig_pre[k] for k in range(W)], guard_sel)

    # sticky = OR(sig_pre[0..(sh-2)])  (0 if sh-2 < 0); use prefix OR table
    sticky_sel = sh - 2
    sticky_s = m.output(UInt(1), "sticky_s")
    sticky_s <<= pick_from_vec(pref, sticky_sel)

    # lsb_s = frac_trunc[0]
    lsb_s = m.output(UInt(1), "lsb_s")
    lsb_s <<= frac_trunc[0]

    # round_up_s = guard_s & (sticky_s | lsb_s)
    round_up_s = m.output(UInt(1), "round_up_s")
    round_up_s <<= guard_s & (sticky_s | lsb_s)

    return m


# ---------------- reference model (Python ints) ----------------
def ref_round_bits(sig_pre: int, sh: int, FW: int):
    W = FW + 1
    maskW = (1 << W) - 1
    sig_pre &= maskW

    # Python shift (logical right)
    sig_shiftN = (sig_pre >> sh) & maskW

    # frac_trunc LSB
    lsb = (sig_shiftN >> 0) & 1 if FW > 0 else 0

    # guard = bit at (sh-1) if in range
    if 0 <= sh - 1 <= W - 1:
        guard = (sig_pre >> (sh - 1)) & 1
    else:
        guard = 0

    # sticky = OR(sig_pre[0..(sh-2)]) if index in range
    if sh - 2 >= 0:
        up_to = min(sh - 2, W - 1)
        sticky = 1 if (sig_pre & ((1 << (up_to + 1)) - 1)) != 0 else 0
    else:
        sticky = 0

    round_up = guard & (sticky | lsb)
    return {
        "sig_shiftN": sig_shiftN,
        "frac_trunc": sig_shiftN & ((1 << FW) - 1),
        "guard_s": guard,
        "sticky_s": sticky,
        "lsb_s": lsb,
        "round_up_s": round_up,
    }


# ---------------- random test ----------------
def run_random(FW=10, trials=2000, seed=1):
    random.seed(seed)
    m = build_round_probe(FW)

    # build m2
    aag = AigerExporter(m).get_aag()
    aig = conv_aag_into_aig(aag)

    aag_sym = _get_aag_sym(aag)
    m2 = AigerImporter(aag[:-2]+aag_sym).get_spirehdl_module()

    collector = IOCollector()
    collector.group(m2, {
        "sig_pre": UInt(11),
        "sh": UInt(4),
        "sig_shiftN": UInt(11),
        "frac_trunc": UInt(10),
        "guard_s": UInt(1),
        "sticky_s": UInt(1),
        "lsb_s": UInt(1),
        "round_up_s": UInt(1),
    })

    sim = Simulator(m2)

    W = FW + 1
    SAW = max(1, ceil(log2(FW + 2)))
    max_sh = (1 << SAW) - 1  # we’ll clamp draws to FW+1

    fails = 0
    for t in range(trials):
        sig_pre = random.getrandbits(W)
        sh = random.randint(0, FW + 1)  # only values that matter

        # drive DUT
        sim.set("sig_pre", sig_pre).set("sh", sh).eval()

        got = {
            "sig_shiftN": sim.get("sig_shiftN"),
            "frac_trunc": sim.get("frac_trunc"),
            "guard_s": sim.get("guard_s"),
            "sticky_s": sim.get("sticky_s"),
            "lsb_s": sim.get("lsb_s"),
            "round_up_s": sim.get("round_up_s"),
        }
        exp = ref_round_bits(sig_pre, sh, FW)

        if got != exp:
            fails += 1
            #print(f"FAIL t={t}  FW={FW}  sig_pre=0x{sig_pre:0{(W+3)//4}X}  sh={sh}")
            print(f"FAIL t={t}  FW={FW}  sig_pre=0b{sig_pre:0{W}b}  sh={sh}")
            for k in ["sig_shiftN", "frac_trunc", "guard_s", "sticky_s", "lsb_s", "round_up_s"]:
                gv = got[k]
                ev = exp[k]
                if gv == ev:
                    continue
                if k in ("sig_shiftN", "frac_trunc"):
                    print(f"  {k:>10}: got=0x{gv:X}  exp=0x{ev:X}")
                else:
                    print(f"  {k:>10}: got={gv}  exp={ev}")
            # Uncomment to stop at first fail:
            # break

    ok = trials - fails
    print(f"Summary: {ok}/{trials} passed (FW={FW}).")


if __name__ == "__main__":
    # Try both float16/ bfloat16 mantissa widths
    run_random(FW=10, trials=5000, seed=42)  # float16: FW=10
    #run_random(FW=7, trials=5000, seed=123)  # bfloat16: FW=7