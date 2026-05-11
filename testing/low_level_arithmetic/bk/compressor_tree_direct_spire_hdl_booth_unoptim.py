import abc
from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Dict, Literal, Optional, Tuple, List
import numpy as np
from spirehdl.helpers import get_yosys_transistor_count, run_vectors
from spirehdl.spirehdl_module import gen_spec
from spirehdl.spirehdl import Bool, Concat, Const, Expr, SInt, Signal, UInt, mux, mux_if
from spirehdl.spirehdl_module import Module

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
            # beyond MSB uses sign bit if signed, else 0
            return b[wb - 1] if b_signed else Const(False, Bool())
        
        # ---------- Precompute selectable versions of a ----------
        # |a| with 1-bit sign extension (for left-shift headroom)
        mag1: List[Expr] = [a[i] for i in range(wa)] + [a[wa - 1] if a_signed else Const(False, Bool())]  # len = wa+1
        # |2a| = mag1 << 1
        mag2: List[Expr] = [Const(False, Bool())] + mag1  # len = wa+2
        
        # Bitwise inverses for negative terms (two's complement done via +1 later)
        mag1_inv: List[Expr] = [~bit for bit in mag1]
        mag2_inv: List[Expr] = [~bit for bit in mag2]
        
        # Utility: sign-extend a vector "v" on reads beyond its length
        def get_se(v: List[Expr], t: int) -> Expr:
            return v[t] if t < len(v) else v[-1]  # extend with last (sign) bit

        
        # multiplicand extended by one sign bit for ±2a (shift-left) headroom
        a_ext: List[Expr] = [a[i] for i in range(wa)] + [a[wa - 1] if a_signed else Const(False, Bool())]  # len = wa+1
        a2_ext: List[Expr] = [Const(False, Bool())] + a_ext  # == (a_ext << 1), len = wa+2
        
        # Radix-4 Booth: one term per 2 multiplier bits
        n_groups = (wb+2) //2 #np.ceil( (wb + 1) / 2).astype(int)  # ceil(wb/2)
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
            
            sel0      = eq000 | eq111             # 0
            sel_pos1  = eq001 | eq010             # +a
            sel_pos2  = eq011                     # +2a
            sel_neg1  = eq101 | eq110             # -a
            sel_neg2  = eq100                     # -2a
        
            # Emit magnitude bits (a_ext or a2_ext), conditionally inverted if neg
            # Then add +1 correction at the block LSB when neg (two's complement).
            # Place bits starting at column base_w = 2*i (radix-4 shift).
            base_w = 2 * i
        
            # Use the longer of the two candidates to cover both |a| and |2a|.
            max_len = max(len(a_ext), len(a2_ext)) *2
            for t in range(max_len):
                # pick bit t of |a| or |2a| guarded by enables; missing bits are 0
                uses_precomute = False
                if not uses_precomute:
                    # do sign extension for out-of-range bits
                    bit_a  = a_ext[t]  if t < len(a_ext)  else a_ext[-1]
                    bit_2a = a2_ext[t] if t < len(a2_ext) else a2_ext[-1]
                    mag    = (bit_a & use1) | (bit_2a & use2)  # selected magnitude bit
            
                    # conditional inversion for negative terms
                    emit_bit = (mag ^ neg)  # zero term forces 0 output
                else:
                    b_pos1 = get_se(mag1, t)
                    b_pos2 = get_se(mag2, t)
                    b_neg1 = get_se(mag1_inv, t)
                    b_neg2 = get_se(mag2_inv, t)
    
                    emit_bit = (b_pos1 & sel_pos1) | (b_pos2 & sel_pos2) | (b_neg1 & sel_neg1) | (b_neg2 & sel_neg2)
                    
        
                w = base_w + t
                if w < out_bits:      # discard columns beyond output width
                    cols[w].append(emit_bit)
        
            # two's-complement +1 correction at the block’s LSB when neg
            if base_w < out_bits:
                cols[base_w].append(neg)

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

            # append test vector
            vecs.append((f"{va}*{vb}", {"a": va, "b": vb}, {"y": va * vb}))

        spec = {"a": UInt(self.a_w), "b": UInt(self.b_w), "y": UInt(self.y_w)}
        return spec, vecs, None


def gen_spirehdl_module(class_instance: MultiplierCompressorTree) -> Module:
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
    signed = False
    mult = MultiplierCompressorTree(a_w=n_bits, b_w=n_bits, signed_a=signed, signed_b=signed)
    m = gen_spirehdl_module(mult)
    # get size in # t transistors
    print(m.to_verilog())
    
    tc = get_yosys_transistor_count(m, n_iter_optimizations=10)
    print(f"Yosys-reported transistor count: {tc}")

    specs, vecs, dec = MultiplierTestVectors(a_w=n_bits, b_w=n_bits, num_vectors=16, tb_sigma=5.0, signed_a=signed, signed_b=signed).generate()
    specs2 = gen_spec(mult)
    run_vectors(m, vecs, decoder=dec, use_signed=True)


if __name__ == "__main__":
    main()
