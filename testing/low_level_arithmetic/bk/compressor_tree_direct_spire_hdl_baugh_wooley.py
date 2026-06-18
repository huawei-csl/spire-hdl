import abc
from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Dict, Literal, Optional, Tuple, List
import numpy as np
from spire.helpers import get_yosys_transistor_count, run_vectors
from spire.component import gen_spec
from spire.expr import Bool, Concat, Const, Expr, SInt, Signal, UInt
from spire.component import Module


# abstract Component Class
class Component(abc.ABC):

    io: dataclass

    @abc.abstractmethod
    def elaborate(self):
        pass


class MultiplierCompressorTree(Component):

    def __init__(self, a_w: int, b_w: int, signed_a: bool = False, signed_b: bool = False, optim_type: Literal["area", "speed"] = "area"):

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
            a=Signal(typ=base_typ_a(self.n_bits), kind="input"),
            b=Signal(typ=base_typ_b(self.n_bits), kind="input"),
            y=Signal(typ=base_type_y(2 * self.n_bits), kind="output"),
        )

        # build the module
        self.elaborate()

    def elaborate(self):

        def generate_partial_products() -> defaultdict:

            use_precompute = False
            use_precompute_v2 = False  # taken into account when use_precompute is True
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

            # ordinary partial products
            for i in range(wa - 1):
                for j in range(wb - 1):
                    w = i + j
                    if w >= out_bits:
                        continue  # ignore bits beyond output width, in case of sign extension
                    p = a[i] & b[j]
                    cols[w].append(p)

            # sign row (i = wa-1)
            for j in range(wb - 1):
                i = wa - 1
                p = a[i] & b[j]
                cols[i + j].append(~p)

            # sign column (j = wb-1)
            for i in range(wa - 1):
                j = wb - 1
                p = a[i] & b[j]
                cols[i + j].append(~p)

            # sign corner (i = wa-1, j = wb-1)
            i = wa - 1
            j = wb - 1
            p = a[i] & b[j]
            cols[i + j].append(p)  # double inversion cancels out

            # correction bits
            cols[wa - 1 + wb - 1 + 1].append(Const(True, Bool()))  # top correction
            cols[wa - 1].append(Const(True, Bool()))  # compensate sign-row inversion
            cols[wb - 1].append(Const(True, Bool()))  # compensate sign-column inversion

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

        full_adder = full_adder_low_area if self.optim_type == "area" else full_adder_fast

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

    def __init__(self, a_w: int, b_w: int, num_vectors: int = 64, tb_sigma: Optional[float] = None, signed_a: bool = False, signed_b: bool = False):
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


def gen_spire_module(class_instance: MultiplierCompressorTree) -> Module:
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
    m = gen_spire_module(mult)
    # get size in # t transistors
    # print(m.to_verilog())

    tc = get_yosys_transistor_count(m, n_iter_optimizations=10)
    print(f"Yosys-reported transistor count: {tc}")

    specs, vecs, dec = MultiplierTestVectors(a_w=n_bits, b_w=n_bits, num_vectors=16, tb_sigma=None, signed_a=signed, signed_b=signed).generate()
    specs2 = gen_spec(mult)
    run_vectors(m, vecs, decoder=dec, use_signed=True, print_on_pass=True)


if __name__ == "__main__":
    main()
