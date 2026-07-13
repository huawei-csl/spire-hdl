# Plots for 8-bit formats: E4M3 (EW=4, FW=3) and E5M2 (EW=5, FW=2)
# Uses the same fp_decode interface the user provided.

from math import copysign, isfinite, isnan
import math
import os
import numpy as np
import matplotlib.pyplot as plt

out_folder = 'fp_value_distribution_outputs'
os.makedirs(out_folder, exist_ok=True)


def fp_bias(EW: int) -> int:
    return (1 << (EW - 1)) - 1


def fp_unpack(bits: int, EW: int, FW: int):
    s = (bits >> (EW + FW)) & 0x1
    e = (bits >> FW) & ((1 << EW) - 1)
    f = bits & ((1 << FW) - 1)
    return s, e, f


def fp_decode_ieee_like(bits: int, EW: int, FW: int, subnormals: bool=True) -> float:
    """IEEE-like decode (bias=2^(EW-1)-1), with subnormals and Inf/NaN."""
    s, e, f = fp_unpack(bits, EW, FW)
    bias = fp_bias(EW)
    if e == 0:
        if f == 0:
            return -0.0 if s else 0.0
        # subnormal
        if subnormals:
            return copysign((f / (1 << FW)) * 2.0 ** (1 - bias), -1.0 if s else 1.0)
        else:
            return float("nan")
    if e == (1 << EW) - 1:
        if f == 0:
            return float("-inf") if s else float("inf")
        return float("nan")
    # normal
    return copysign((1.0 + f / (1 << FW)) * 2.0 ** (e - bias), -1.0 if s else 1.0)

# high-float 8:
def _is_nan(enc: int) -> bool:
    return enc == 0x80

def _is_inf(enc: int) -> bool:
    return enc in (0x6F, 0xEF)

def _is_zero(enc: int) -> bool:
    return enc == 0x00

def hif8_to_float(enc: int, EW: int, FW: int, subnormals: bool) -> float:
    """Decode an 8-bit HiFloat8 payload into a Python float."""
    if _is_nan(enc):
        return math.nan
    if _is_zero(enc):
        return 0.0
    if _is_inf(enc):
        return -math.inf if enc & 0x80 else math.inf

    sign = -1.0 if (enc & 0x80) else 1.0
    payload = enc & 0x7F

    # Dot-field prefix decoding (variable length)
    bits = payload
    if bits >> 5 == 0b11:  # D = 4  (2-bit dot)
        exp_field = (bits >> 1) & 0xF
        mant_field = bits & 0x1
        exp_sign = (exp_field >> 3) & 0x1
        mag = 8 + (exp_field & 0x7)
        exponent = -mag if exp_sign else mag
        significand = 1.0 + mant_field / 2.0
    elif bits >> 5 == 0b10:  # D = 3
        exp_field = (bits >> 2) & 0x7
        mant_field = bits & 0x3
        exp_sign = (exp_field >> 2) & 0x1
        mag = 4 + (exp_field & 0x3)
        exponent = -mag if exp_sign else mag
        significand = 1.0 + mant_field / 4.0
    elif bits >> 5 == 0b01:  # D = 2
        exp_field = (bits >> 3) & 0x3
        mant_field = bits & 0x7
        exp_sign = (exp_field >> 1) & 0x1
        mag = 2 + (exp_field & 0x1)
        exponent = -mag if exp_sign else mag
        significand = 1.0 + mant_field / 8.0
    elif bits >> 4 == 0b001:  # D = 1
        exp_sign = (bits >> 3) & 0x1
        exponent = -1 if exp_sign else 1
        mant_field = bits & 0x7
        significand = 1.0 + mant_field / 8.0
    elif bits >> 3 == 0b0001:  # D = 0
        exponent = 0
        mant_field = bits & 0x7
        significand = 1.0 + mant_field / 8.0
    else:  # DML denormals
        mant_field = bits & 0x7
        if mant_field == 0:
            return 0.0
        exponent = mant_field - 23  # M encodes exponent offset
        significand = 1.0

    return sign * math.ldexp(significand, exponent)

def hif8_get_D(enc: int) -> int:
    """Get the dot-field prefix D value from a HiFloat8 encoding."""
    if _is_nan(enc) or _is_inf(enc) or _is_zero(enc):
        return -1  # special
    bits = enc & 0x7F
    if bits >> 5 == 0b11:
        return 4
    elif bits >> 5 == 0b10:
        return 3
    elif bits >> 5 == 0b01:
        return 2
    elif bits >> 4 == 0b001:
        return 1
    elif bits >> 3 == 0b0001:
        return 0
    else:
        return -2  # denormal
    
def hif8_get_mantissa(enc: int) -> int:
    """Get the mantissa field from a HiFloat8 encoding."""
    if _is_nan(enc) or _is_inf(enc) or _is_zero(enc):
        return 0
    bits = enc & 0x7F
    if bits >> 5 == 0b11:  # D = 4
        return bits & 0x1
    elif bits >> 5 == 0b10:  # D = 3
        return bits & 0x3
    elif bits >> 5 == 0b01:  # D = 2
        return bits & 0x7
    elif bits >> 4 == 0b001:  # D = 1
        return bits & 0x7
    elif bits >> 3 == 0b0001:  # D = 0
        return bits & 0x7
    else:
        return bits & 0x7  # DML denormals

# end high-float 8




def make_plots(EW: int, FW: int, fp_decode: callable, hifloat: bool = False, subnormals: bool = True):

    if subnormals and not hifloat:
        sn_str = "sn_"
    elif not subnormals and not hifloat:
        sn_str = "no_sn_"
    else:
        sn_str = "hf8_"

    TOTAL = 1 << (1 + EW + FW)
    vals = []
    es = []
    Ds = []  # HiFloat8 dot-field class per encoding (used when hifloat)
    Ms = []  # HiFloat8 mantissa field per encoding (used when hifloat)
    ninf = pinf = nans = 0
    for b in range(TOTAL):
        v = fp_decode(b, EW, FW, subnormals=subnormals)
        s, e, f = fp_unpack(b, EW, FW)
        if isnan(v):
            nans += 1
            continue
        if v == float("inf"):
            pinf += 1
            continue
        if v == float("-inf"):
            ninf += 1
            continue
        if isfinite(v):
            vals.append(v)
            es.append(e)
            if hifloat:
                Ds.append(hif8_get_D(b))
                Ms.append(hif8_get_mantissa(b))

    vals = np.array(vals, dtype=float)
    es = np.array(es, dtype=int)

    # 1) Histogram of log2(|value|) for nonzero finite values
    nz = vals != 0.0
    absvals = np.abs(vals[nz])
    log2abs = np.log2(absvals)
    
    subnormal_str = ", subnormals on" if subnormals else ", subnormals off"

    plt.figure(figsize=(9, 4.5))
    bins = 200  # fixed; 8-bit is tiny
    plt.hist(log2abs, bins=bins)
    plt.xlabel("log2(|value|)")
    plt.ylabel("Count")
    if not hifloat:
        plt.title(f"8-bit FP log2 magnitude distribution (EW={EW}, FW={FW}{subnormal_str})")
    else:
        plt.title(f"HiFloat8 log2 magnitude distribution")
    plt.tight_layout()
    out1 = f"{out_folder}/fp8_EW{EW}_FW{FW}_{sn_str}log2abs.png"
    plt.savefig(out1, dpi=150)
    plt.close()

    # 2) Counts per exponent field e (finite only)
    maxe = (1 << EW) - 1
    bins_e = np.arange(0, maxe + 1)
    counts_e = [(es == ee).sum() for ee in bins_e]

    plt.figure(figsize=(9, 4))
    plt.bar(bins_e, counts_e, width=0.8)
    plt.xlabel("Exponent field e")
    plt.ylabel("Finite count with this e")
    if not hifloat:
        plt.title(f"8-bit FP counts per exponent field (EW={EW}, FW={FW}{subnormal_str})")
    else:
        plt.title(f"HiFloat8 counts per exponent field")
    plt.tight_layout()
    out2 = f"{out_folder}/fp8_EW{EW}_FW{FW}_{sn_str}counts_per_e.png"
    plt.savefig(out2, dpi=150)
    plt.close()

    # 3) |value| vs ULP spacing (sorted uniques, forward diff), log–log
    uniq = np.unique(vals)  # includes zeros
    d = np.diff(uniq)
    x = uniq[:-1]
    mask = np.abs(x) > 0
    x_abs = np.abs(x[mask])
    d_nonzero = d[mask]

    plt.figure(figsize=(9, 5))
    plt.loglog(x_abs, d_nonzero, ".", markersize=3)
    plt.xlabel("|value|")
    plt.ylabel("Next representable spacing (ULP)")
    plt.title(f"8-bit FP spacing vs magnitude (EW={EW}, FW={FW}{subnormal_str})")
    plt.tight_layout()
    out3 = f"{out_folder}/fp8_EW{EW}_FW{FW}_{sn_str}spacing.png"
    plt.savefig(out3, dpi=150)
    plt.close()

    # 4) Empirical CDF of finite values (including zeros and sign)
    x_sorted = np.sort(vals)
    n = x_sorted.size
    if n > 0:
        y = np.arange(1, n + 1) / n
        plt.figure(figsize=(9, 4.5))
        plt.step(x_sorted, y, where="post")
        plt.xlabel("value")
        plt.ylabel("Empirical CDF")
        if not hifloat:
            plt.title(f"8-bit FP empirical CDF (EW={EW}, FW={FW}{subnormal_str})")
        else:
            plt.title("HiFloat8 empirical CDF")
        plt.tight_layout()
        out4 = f"{out_folder}/fp8_EW{EW}_FW{FW}_{sn_str}cdf_values.png"
        plt.savefig(out4, dpi=150)
        plt.close()
    else:
        out4 = None

    # 5) Empirical CDF in log-domain (log2 |value|) for nonzero finite values
    x_sorted_log = np.sort(log2abs)
    nlog = x_sorted_log.size
    if nlog > 0:
        ylog = np.arange(1, nlog + 1) / nlog
        plt.figure(figsize=(9, 4.5))
        plt.step(x_sorted_log, ylog, where="post")
        plt.xlabel("log2(|value|)")
        plt.ylabel("Empirical CDF")
        if not hifloat:
            plt.title(f"8-bit FP empirical CDF of log2(|value|) (EW={EW}, FW={FW}{subnormal_str})")
        else:
            plt.title("HiFloat8 empirical CDF of log2(|value|)")
        plt.tight_layout()
        out5 = f"{out_folder}/fp8_EW{EW}_FW{FW}_{sn_str}cdf_log2abs.png"
        plt.savefig(out5, dpi=150)
        plt.close()
    else:
        out5 = None

    # 6) If HiFloat8: log-domain CDFs split by D (scaled to sum to total CDF)
    out6 = None
    out7 = None
    if hifloat:
        vals_np = np.array(vals, dtype=float)
        Ds_np = np.array(Ds, dtype=int) if len(Ds) else np.array([], dtype=int)
        Ms_np = np.array(Ms, dtype=int) if len(Ms) else np.array([], dtype=int)

        nz_idx = np.where(vals_np != 0.0)[0]
        N = nz_idx.size
        if N > 0:
            from collections import defaultdict
            log_nonzero = np.log2(np.abs(vals_np[nz_idx]))

            # Group by D
            by_d = defaultdict(list)
            for pos, idx in enumerate(nz_idx):
                by_d[Ds_np[idx]].append(log_nonzero[pos])

            plt.figure(figsize=(12, 7))
            # Overall CDF in bold
            plt.step(x_sorted_log, ylog, where="post", color="k", linewidth=2.0, label="All (log2|v|)")

            color_cycle = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
            present_ds = sorted(by_d.keys(), reverse=True)
            order = [d for d in [4, 3, 2, 1, 0, -2, -1] if d in present_ds]
            for i, d in enumerate(order):
                xs = np.sort(np.array(by_d[d], dtype=float))
                nd = xs.size
                if nd == 0:
                    continue
                ys = (np.arange(1, nd + 1) / nd) * (nd / N)
                color = color_cycle[i % len(color_cycle)] if color_cycle else None
                if d >= 0:
                    lab = f"D={d} (w={nd/N:.2f})"
                elif d == -2:
                    lab = f"DML (w={nd/N:.2f})"
                else:
                    lab = f"Zero/Special (w={nd/N:.2f})"
                plt.step(xs, ys, where="post", label=lab, color=color, alpha=0.9)

            plt.xlabel("log2(|value|)")
            plt.ylabel("Empirical CDF (scaled)")
            plt.title("HiFloat8 log-domain CDF split by D (scaled)")
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            out6 = f"{out_folder}/fp8_EW{EW}_FW{FW}_{sn_str}cdf_log2abs_by_D_scaled.png"
            plt.savefig(out6, dpi=150)
            plt.close()

            # 7) Group by mantissa (variable width) and plot scaled CDFs
            by_m = defaultdict(list)
            for pos, idx in enumerate(nz_idx):
                by_m[Ms_np[idx]].append(log_nonzero[pos])

            plt.figure(figsize=(12, 7))
            # Overall CDF in bold
            plt.step(x_sorted_log, ylog, where="post", color="k", linewidth=2.0, label="All (log2|v|)")

            for i, m in enumerate(sorted(by_m.keys())):
                xs = np.sort(np.array(by_m[m], dtype=float))
                nm = xs.size
                if nm == 0:
                    continue
                ys = (np.arange(1, nm + 1) / nm) * (nm / N)
                color = color_cycle[i % len(color_cycle)] if color_cycle else None
                plt.step(xs, ys, where="post", label=f"m={m} (w={nm/N:.2f})", color=color, alpha=0.9)

            plt.xlabel("log2(|value|)")
            plt.ylabel("Empirical CDF (scaled)")
            plt.title("HiFloat8 log-domain CDF split by mantissa (scaled)")
            plt.legend(loc="best", fontsize=8, ncol=2)
            plt.tight_layout()
            out7 = f"{out_folder}/fp8_EW{EW}_FW{FW}_{sn_str}cdf_log2abs_by_m_scaled.png"
            plt.savefig(out7, dpi=150)
            plt.close()

    return out1, out2, out3, out4, out5, out6, out7, pinf, ninf, nans


# Generate for E4M3 and E5M2
def main():
    files_e4m3 = make_plots(4, 3, fp_decode_ieee_like, subnormals=True)
    files_e5m2 = make_plots(5, 2, fp_decode_ieee_like, subnormals=True)
    files_e4m3 = make_plots(4, 3, fp_decode_ieee_like, subnormals=False)
    files_e5m2 = make_plots(5, 2, fp_decode_ieee_like, subnormals=False)
    files_hif8 = make_plots(8-1, 0, hif8_to_float, hifloat=True)
    return files_e4m3, files_e5m2, files_hif8


if __name__ == "__main__":
    main()
