import abc
from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Dict, Literal, Optional, Tuple, List
import numpy as np
from sprouthdl.helpers import get_yosys_transistor_count, run_vectors
from sprouthdl.sprouthdl_module import gen_spec
from sprouthdl.sprouthdl import Bool, Concat, Const, Expr, SInt, Signal, UInt, fit_type
from sprouthdl.sprouthdl_module import Module

# abstract Component Class
class Component(abc.ABC):

    io : dataclass

    @abc.abstractmethod
    def elaborate(self):
        pass

class MultiplierCompressorTree(Component):

    def __init__(self,
                 a_w: int, 
                 b_w: int,
                 signed_a: bool = False,
                 signed_b: bool = False,
                 optim_type: Literal["area", "speed"] = "area"):

        # set attributes
        self.n_bits = a_w
        self.optim_type = optim_type

        # declare the inputs and outputs
        @dataclass
        class IO:
            a: Signal
            b: Signal
            y: Signal

        base_typ_a = SInt if signed_a else UInt
        base_typ_b = SInt if signed_b else UInt
        base_type_y = SInt if (signed_a or signed_b) else UInt

        self.io: IO = IO(
            a=Signal(name="a", typ=base_typ_a(self.n_bits), kind="input"),
            b=Signal(name="b", typ=base_typ_b(self.n_bits), kind="input"),
            y=Signal(name="y", typ=base_type_y(2 * self.n_bits), kind="output"),
        )

        # build the module
        self.elaborate()

    def elaborate(self):

        def generate_partial_products() -> defaultdict:

            use_precompute = False
            use_precompute_v2 = False # taken into account when use_precompute is True
            if not use_precompute:
                use_precompute_v2 = False

            cols: Dict[int, List[Expr]] = defaultdict(list)  # weight -> signal node indices

            a = self.io.a
            b = self.io.b
            n = self.n_bits
            wa = a.typ.width
            wb = b.typ.width

            out_bits = a.typ.width + b.typ.width

            a_signed = a.typ.signed
            b_signed = b.typ.signed

            # --- helper to get multiplier bits with proper out-of-range behavior
            def bbit(k: int) -> Expr:
                if 0 <= k < wb:
                    return b[k]
                if k < 0:
                    return Const(False, Bool())  # below LSB is 0
                # beyond MSB uses sign bit if signed, else 0
                return b[wb - 1] if b_signed else Const(False, Bool())

            # ---------- Precompute selectable versions of a ----------
            # |a| with 1-bit sign extension (for left-shift headroom)
            mag1: List[Expr] = [a[i] for i in range(wa)] + [a[wa - 1] if a_signed else Const(False, Bool())]  # len = wa+1
            # |2a| = mag1 << 1
            mag2: List[Expr] = [Const(False, Bool())] + mag1[:-1]  # len = wa+2

            # Bitwise inverses for negative terms (two's complement done via +1 later)
            mag1_inv: List[Expr] = [~bit for bit in mag1]
            mag2_inv: List[Expr] = [~bit for bit in mag2]

            if use_precompute_v2:
                # a_signedv = Signal('a_signed', SInt(wa+1), kind='wire')
                # a_signedv <<= a
                # mag1_invs = -a_signedv
                mag1_invs = -fit_type(a, SInt(wa+1))
                mag2_invs = mag1_invs << 1

                mag1_invs_v = [mag1_invs[i] for i in range(wa+1)]
                mag2_invs_v = [mag2_invs[i] for i in range(wa+1)]

            # multiplicand extended by one sign bit for ±2a (shift-left) headroom
            a_ext: List[Expr] = [a[i] for i in range(wa)] + [a[wa - 1] if a_signed else Const(False, Bool())]  # len = wa+1
            a2_ext: List[Expr] = [Const(False, Bool())] + a_ext  # == (a_ext << 1), len = wa+2

            # Radix-4 Booth: one term per 2 multiplier bits
            extra_pp_lines = 0 if b_signed else 1
            n_groups = (wb+1 + extra_pp_lines) //2 #np.ceil( (wb + 1) / 2).astype(int)  # ceil(wb/2)
            for i in range(n_groups):
                x = bbit(2 * i - 1)  # low
                y = bbit(2 * i)      # mid
                z = bbit(2 * i + 1)  # high

                # Decode the 3-bit Booth code (z y x):
                # 000/111 -> 0; 001/010 -> +1; 011 -> +2; 100 -> -2; 101/110 -> -1
                nz, ny, nx = ~z, ~y, ~x

                eq000 = nz & ny & nx
                eq001 = nz & ny &  x
                eq010 = nz &  y & nx
                eq011 = nz &  y &  x
                eq100 =  z & ny & nx
                eq101 =  z & ny &  x
                eq110 =  z &  y & nx
                eq111 =  z &  y &  x

                pos1 = eq001 | eq010
                pos2 = eq011
                neg1 = eq101 | eq110
                neg2 = eq100
                neg  = neg1 | neg2
                use1 = pos1 | neg1         # select |a|
                use2 = pos2 | neg2         # select |2a|
                zero = eq000 | eq111        # zero term

                if not use_precompute:
                    use1 = x ^ y # simplified logic
                    neg = z # simplified logic

                if use_precompute:
                    sel0      = eq000 | eq111             # 0
                    sel_pos1  = eq001 | eq010             # +a
                    sel_pos2  = eq011                     # +2a
                    sel_neg1  = eq101 | eq110             # -a
                    sel_neg2  = eq100                     # -2a

                # Emit magnitude bits (a_ext or a2_ext), conditionally inverted if neg
                # Then add +1 correction at the block LSB when neg (two's complement).
                # Place bits starting at column base_w = 2*i (radix-4 shift).
                base_w = 2 * i

                max_len = max(len(a_ext), len(a2_ext)) *2
                extend_bit = None
                for t in range(max_len):

                    if t < wa+1: # actual booth decoded / booth selector bits (without +1 correction)
                        

                        if use_precompute:
                            a_pos1 = mag1[t]
                            a_pos2 = mag2[t]
                            a_neg1 = mag1_inv[t]
                            a_neg2 = mag2_inv[t]

                            if use_precompute_v2:                            
                                a_neg1 = mag1_invs_v[t]
                                a_neg2 = mag2_invs_v[t]

                            emit_bit = (a_pos1 & sel_pos1) | (a_pos2 & sel_pos2) | (a_neg1 & sel_neg1) | (a_neg2 & sel_neg2)
                            # emit_bit = (a_pos1 & (use1 & ~neg)) | (a_pos2 & (use2 & ~neg)) | (a_neg1 & (use1 & neg)) | (a_neg2 & (use2 & neg))

                            # emit_bit = (a_ext[t] & (use1 & (~neg))) | (a2_ext[t] & (use2 & (~neg))) | ((~a_ext[t]) & (use1 & neg)) | ((~a2_ext[t]) & (use2 & neg))

                            # mag = (a_ext[t] & use1) | (a2_ext[t] & use2)
                            # emit_bit = mux_if(neg, ~mag, mag)

                        else: 
                            mag = (a_ext[t] & use1) | (a2_ext[t] & use2)
                            emit_bit = (mag ^ neg)

                        extend_bit = ~emit_bit # save for sign extension # flip from  1 to 0 when MSB is 0

                    elif t == wa + 1:
                        if a_signed:
                            emit_bit = extend_bit  # sign extend
                        else:
                            emit_bit = ~neg # S inverse                        
                    elif t == wa + 1 + 1:                        
                        emit_bit = Const(True, Bool()) # constant 1
                    else:
                        emit_bit = None  # beyond this, all terms are zero

                    w = base_w + t
                    if w < out_bits and emit_bit is not None:      # discard columns beyond output width
                        cols[w].append(emit_bit)  

                # two's-complement +1 correction at the block’s LSB when neg
                if not use_precompute_v2:
                    if base_w < out_bits:
                        cols[base_w].append(neg)

                # Correction
                if i == 0:
                    cols[len(mag1)].append(Const(True, Bool()))
                    
            return cols

        cols = generate_partial_products()
        print(f"Generated {sum(len(v) for v in cols.values())} partial product bits in {len(cols)} columns")

        def half_adder(x: Expr, y: Expr) -> Tuple[Expr, Expr]:
            return x ^ y, x & y  # sum, carry

        def full_adder_fast(x: Expr, y: Expr, z: Expr) -> Tuple[Expr, Expr]:
            s = x ^ y ^ z
            c = (x & y) | (y & z) | (z & x)

            return s, c  # sum, carry

        def full_adder_low_area(x: Expr, y: Expr, z: Expr) -> Tuple[Expr, Expr]:
            s1 = x ^ y
            s = s1 ^ z
            c = (s1 & z) | (x & y)

            return s, c  # sum, carry

        full_adder = full_adder_low_area  if self.optim_type == "area" else full_adder_fast

        def compress_column(bits: List[Expr]) -> Tuple[List[Expr], List[Expr]]:
            """Compress a column of bits using full and half adders.
            Returns (sum_bits, carry_bits) where carry_bits are to be added to the next column.
            """
            sum_bits = []
            carry_bits = []
            while len(bits) >= 3:
                x, y, z = bits[:3]
                bits = bits[3:]
                s, c = full_adder(x, y, z)
                sum_bits.append(s)
                carry_bits.append(c)
            if len(bits) == 2:
                x, y = bits
                s, c = half_adder(x, y)
                sum_bits.append(s)
                carry_bits.append(c)
                # or
                # sum_bits.extend(bits)  # leave uncompressed
            elif len(bits) == 1:
                sum_bits.append(bits[0])
            return sum_bits, carry_bits

        # Compression stages
        while True:
            new_cols: Dict[int, List[Expr]] = defaultdict(list)
            done = True
            for w in sorted(cols.keys()):
                bits = cols[w]
                if len(bits) > 2:
                    done = False
                    sum_bits, carry_bits = compress_column(bits)
                    new_cols[w].extend(sum_bits)
                    # for c in carry_bits:
                    #    new_cols[w + 1].append(c)
                    new_cols[w + 1].extend(carry_bits)
                else:
                    new_cols[w].extend(bits)
            cols = new_cols
            if done:
                break

        # Final addition
        result_bits = []
        carry: Optional[Expr] = None
        for w in range(2 * self.n_bits):
            bits = cols.get(w, [])
            if carry is not None:
                bits.append(carry)
            if len(bits) == 0:
                s = Const(False, Bool())
                carry = None
            elif len(bits) == 1:
                s = bits[0]
                carry = None
            elif len(bits) == 2:
                s, carry = half_adder(bits[0], bits[1])
            elif len(bits) == 3:
                s, carry = full_adder(bits[0], bits[1], bits[2])
            else:
                raise ValueError("Unexpected number of bits in final addition")
            result_bits.append(s)
        if carry is not None:
            result_bits.append(carry)

        # Assign to output
        self.io.y <<= Concat(result_bits[: 2 * self.n_bits])
        print(f"MultiplierCompressorTree: {self.n_bits}x{self.n_bits} -> {2*self.n_bits} bits")        
        print(f"Final result bits: {len(result_bits)}")


class MultiplierTestVectors:

    def __init__(self, a_w: int, b_w: int, num_vectors: int = 64, tb_sigma: Optional[float] = None,
                 signed_a: bool = False, signed_b: bool = False):
        self.a_w = a_w
        self.b_w = b_w
        self.y_w = a_w + b_w
        self.num_vectors = num_vectors
        self.tb_sigma = tb_sigma
        self.a_signed = signed_a
        self.b_signed = signed_b
        
    def generate(self) -> Tuple:

        vecs = []
        for _ in range(self.num_vectors):
            if self.tb_sigma is not None:
                
                # generate random values from normal distribution
                def rand_gen_unsigned(n):
                    return int(np.round((np.random.normal((1 << (n - 1)), self.tb_sigma))))
                
                def rand_gen_signed(n):
                    return int(np.round((np.random.normal(0, self.tb_sigma))))

                va = rand_gen_signed(self.a_w) if self.a_signed else rand_gen_unsigned(self.a_w)
                vb = rand_gen_signed(self.b_w) if self.b_signed else rand_gen_unsigned(self.b_w)

                # clamp to range
                def clam_unsigned(v, n):
                    return max(min(v, (1 << n) - 1), 0)
                def clam_signed(v, n):
                    return max(min(v, (1 << (n - 1)) - 1), -(1 << (n - 1)))
                va = clam_signed(va, self.a_w) if self.a_signed else clam_unsigned(va, self.a_w)
                vb = clam_signed(vb, self.b_w) if self.b_signed else clam_unsigned(vb, self.b_w)

            else:                
                va = random.getrandbits(self.a_w)
                vb = random.getrandbits(self.b_w)
                if self.a_signed:
                    va = va - (1 << (self.a_w - 1))
                if self.b_signed:
                    vb = vb - (1 << (self.b_w - 1))

            # append test vector
            vecs.append((f"{va}*{vb}", {"a": va, "b": vb}, {"y": va * vb}))

        spec = {"a": UInt(self.a_w), "b": UInt(self.b_w), "y": UInt(self.y_w)}
        return spec, vecs, None


def gen_sprout_module(class_instance: MultiplierCompressorTree) -> Module:
    m = Module(f"Mul{class_instance.n_bits}_ct", with_clock=False, with_reset=False)
    for sig in class_instance.io.__dict__.values():
        if sig.kind == "input":
            m.add_input(sig)
        elif sig.kind == "output":
            m.add_output(sig)
        else:
            raise ValueError(f"Signal {sig.name} has unsupported kind '{sig.kind}'")
    return m


def main():
    n_bits = 16
    signed = True
    mult = MultiplierCompressorTree(a_w=n_bits, b_w=n_bits, signed_a=signed, signed_b=signed)
    m = gen_sprout_module(mult)
    # get size in # t transistors
    #print(m.to_verilog())
    
    tc = get_yosys_transistor_count(m, n_iter_optimizations=10)
    print(f"Yosys-reported transistor count: {tc}")

    specs, vecs, dec = MultiplierTestVectors(a_w=n_bits, b_w=n_bits, num_vectors=16, tb_sigma=None, signed_a=signed, signed_b=signed).generate()
    specs2 = gen_spec(mult)
    run_vectors(m, vecs, decoder=dec, use_signed=True)


if __name__ == "__main__":
    main()
