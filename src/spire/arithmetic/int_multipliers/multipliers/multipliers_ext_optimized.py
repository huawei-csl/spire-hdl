import abc
from collections import defaultdict
from dataclasses import dataclass
import os
import importlib.resources as resources
from pathlib import Path
import re
import tempfile
from typing import Callable, ClassVar, DefaultDict, Dict, Iterable, List, Literal, Optional, Tuple, Type

from aigverse import read_aiger_into_aig, write_aiger
import numpy as np

from spire.arithmetic.int_multipliers.multipliers.multiplier_stage_core import CompressorTreeAccumulator, FinalStageAdderBase, StageMultiplierConfig, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplierBasic, StageBasedMultiplierIO

from spire.arithmetic.int_multipliers.multipliers.multipliers_ext import StageBasedMultiplierBase
from spire.arithmetic.int_multipliers.stages.ppa_fsa_util import OutputConfig, compressor_sum
from spire.arithmetic.int_multipliers.stages.ppa_stages import CarrySaveAccumulator
from spire.arithmetic.int_multipliers.eval.testvector_generation import Encoding, MultiplierTestVectors, from_encoding, to_encoding
from spire.aig.aig_aigerverse import _get_aag_sym, file_to_lines
from spire.helpers import get_aig_stats, get_yosys_metrics, get_yosys_transistor_count, optimize_aag, run_vectors
from spire.aiger import AigerImporter
from spire.component import Component, ImportedComponent
from spire.io_record import IORecord, Input, Output
from spire.expr import Bool, Concat, Const, Expr, Signal, SInt, UInt, mux, mux_if
from spire.component import Netlist

from spire.aig.aig_to_aag import aig_file_to_aag_lines
from spire.aig.aig_yosys import verilog_to_aag_lines_via_yosys, verilog_to_aag_via_yosys
from spire.aig.aig_yosys import aig_file_to_aag_lines_via_yosys

# Package data locations for precomputed multipliers
RESOURCE_PACKAGE = "spire.arithmetic.int_multipliers.data"
RESOURCE_BASE = Path("optimized")
AIG_FILENAME = "out_aiger.aig"
MAP_FILENAME = "aiger_map_cleaned.map"

DEFAULT_MULTIPLIER_RESOURCE_ROOTS: Dict[Tuple[Encoding, Encoding, int], Path] = {
    (Encoding.unsigned, Encoding.unsigned, 4): RESOURCE_BASE / "unsigned_4b",
    (Encoding.unsigned, Encoding.unsigned, 3): RESOURCE_BASE / "unsigned_3b",
    (Encoding.twos_complement, Encoding.twos_complement, 4): RESOURCE_BASE / "signed_4b",
    (Encoding.twos_complement, Encoding.twos_complement, 3): RESOURCE_BASE / "signed_3b",
    (Encoding.unsigned, Encoding.unsigned, 8): RESOURCE_BASE / "unsigned_8b",
    (Encoding.twos_complement, Encoding.twos_complement, 8): RESOURCE_BASE / "signed_8b",
}

STRONG_UNSIGNED_4B_RESOURCE_ROOT = RESOURCE_BASE / "unsigned_4b_strong"


def _encoding_label(enc: Encoding) -> str:
    return getattr(enc, "name", str(enc))


def _load_aag_lines_from_resources(aig_rel: Path, map_rel: Path, desc: str) -> List[str]:
    # Allow users to point to their own asset directory via env var
    env_base = os.environ.get("SPIRE_OPT_MULT_DIR")
    if env_base:
        base_path = Path(env_base).expanduser()
        aig_path = base_path / aig_rel
        map_path = base_path / map_rel

        missing_env: List[str] = []
        if not aig_path.is_file():
            missing_env.append(aig_path.as_posix())
        if not map_path.is_file():
            missing_env.append(map_path.as_posix())

        if missing_env:
            raise FileNotFoundError(
                f"Missing optimized multiplier asset(s) for {desc} using SPIRE_OPT_MULT_DIR={base_path}: "
                f"{', '.join(missing_env)}. "
                f"Expected layout: {base_path}/{aig_rel.parent}."
            )

        return aig_file_to_aag_lines(str(aig_path), map_file=str(map_path))

    # Fall back to packaged resources
    base = resources.files(RESOURCE_PACKAGE)
    aig_resource = base.joinpath(aig_rel)
    map_resource = base.joinpath(map_rel)

    missing: List[str] = []
    if not aig_resource.is_file():
        missing.append(aig_rel.as_posix())
    if not map_resource.is_file():
        missing.append(map_rel.as_posix())

    if missing:
        raise FileNotFoundError(
            f"Missing optimized multiplier asset(s) for {desc}: {', '.join(missing)}. "
            "Add the files under 'src/spire/arithmetic/int_multipliers/data/optimized/' "
            "or set SPIRE_OPT_MULT_DIR to point to your own asset directory."
        )

    with resources.as_file(aig_resource) as aig_path, resources.as_file(map_resource) as map_path:
        return aig_file_to_aag_lines(str(aig_path), map_file=str(map_path))
    
    # other options
    # aag_lines = verilog_to_aag_lines_via_yosys(verilog_file_name, top="mydesign_comb", embed_symbols=True, no_startoffset=True) # works but higher aig count


# ----- precomputed optimized multipliers stored as AIGs: ffile locations need to be adopted -----

def get_aag_lines_default(aw: int, bw: int, a_encoding: Encoding, b_encoding: Encoding) -> List[str]:

    root = DEFAULT_MULTIPLIER_RESOURCE_ROOTS.get((a_encoding, b_encoding, aw))
    if root is None or aw != bw:
        raise NotImplementedError(
            f"No precomputed AIG packaged for aw={aw}, bw={bw}, encodings ({_encoding_label(a_encoding)}, {_encoding_label(b_encoding)}). "
            "Provide 'f_aag_lines' or add the expected AIG files under 'spire/arithmetic/int_multipliers/data/'."
        )

    return _load_aag_lines_from_resources(
        root / AIG_FILENAME,
        root / MAP_FILENAME,
        f"{_encoding_label(a_encoding)}/{_encoding_label(b_encoding)} {aw}x{bw}",
    )


def get_optimized_aag_lines_strong(aw: int, bw: int, a_encoding: Encoding, b_encoding: Encoding) -> List[str]:
    if not(a_encoding == Encoding.unsigned and b_encoding == Encoding.unsigned and aw == bw and aw == 4):
        raise NotImplementedError("Only optimized 4-bit unsigned multiplier is supported")
    return _load_aag_lines_from_resources(
        STRONG_UNSIGNED_4B_RESOURCE_ROOT / AIG_FILENAME,
        STRONG_UNSIGNED_4B_RESOURCE_ROOT / MAP_FILENAME,
        "strong optimized 4-bit unsigned multiplier",
    )


# ----- optimized multiplier classes -----


class OptimizedMultiplier(StageBasedMultiplierBase):

    def __init__(self, *args, f_aag_lines: Optional[Callable[[int, int, Encoding, Encoding], list[str]]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.a_encoding in (Encoding.unsigned, Encoding.twos_complement), "Only unsigned or two's complement encoding is supported"
        assert self.b_encoding in (Encoding.unsigned, Encoding.twos_complement), "Only unsigned or two's complement encoding is supported"
        y_encoding = Encoding.twos_complement if (self.a_encoding == Encoding.twos_complement or self.b_encoding == Encoding.twos_complement) else Encoding.unsigned

        def get_type(enc: Encoding,) -> Type:
            if enc == Encoding.unsigned:
                return UInt
            elif enc == Encoding.twos_complement:
                return SInt

        self.io: StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Input(get_type(self.a_encoding)(self.aw)),
            b=Input(get_type(self.b_encoding)(self.bw)),
            y=Output(get_type(y_encoding)(self.aw + self.bw)),
        )

        self.f_aag_lines = f_aag_lines if f_aag_lines is not None else get_aag_lines_default

        self.elaborate()

    def elaborate(self) -> None:

        m = AigerImporter(self.f_aag_lines(self.aw, self.bw, self.a_encoding, self.b_encoding)).get_spire_module()

        this_class = self

        class LoadedMultiplier(ImportedComponent):
            # Import shell: logic is reinjected via from_module(); no elaborate().
            def __init__(self):
                self.io = IORecord(
                    operand_a_i=Input(UInt(this_class.aw)),
                    operand_b_i=Input(UInt(this_class.bw)),
                    result_o=Output(UInt(this_class.aw + this_class.bw)),
                )

        self.mult = LoadedMultiplier()
        self.mult.from_module(m, group=True)

        # todo: test from_verlog and from_aig_file

        # use the specified multiplier
        self.mult.io.operand_a_i <<= self.io.a
        self.mult.io.operand_b_i <<= self.io.b
        self.io.y <<= self.mult.io.result_o


class OptimizedSignMagnitudeMultiplier(StageBasedMultiplierBase):

    def __init__(self, *args, f_aag_lines: "Optional[Callable[..., List[str]]]" = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.a_encoding == Encoding.sign_magnitude and self.b_encoding == Encoding.sign_magnitude, "Only sign-magnitude encoding is supported"
        assert self.ppg_cls is None, "PPG must be None"
        assert self.ppa_cls is None, "PPA must be None"
        assert self.fsa_cls is None, "FSA must be None"

        self.io : StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Input(UInt(self.aw)),
            b=Input(UInt(self.bw)),
            y=Output(UInt(self.aw + self.bw - 1)),
        )
        
        self.f_aag_lines = f_aag_lines

        self.elaborate()

    def elaborate(self) -> None:

        # instantiate an unsigned multiplier for the magnitudes
        mult: StageBasedMultiplierBasic = OptimizedMultiplier(
            a_encoding=Encoding.unsigned,
            b_encoding=Encoding.unsigned,
            a_w=self.aw - 1,
            b_w=self.bw - 1,
            ppg_cls=self.ppg_cls,
            ppa_cls=self.ppa_cls,
            fsa_cls=self.fsa_cls,
            f_aag_lines=self.f_aag_lines,
        )

        self.mult = mult

        if self.aw != self.bw:
            raise ValueError(f"{type(self).__name__} expects square operands (aw == bw); "
                             f"got {self.aw}x{self.bw} — b's sign/magnitude split is width-hardcoded")
        W = self.aw

        sa = self.io.a[W - 1]
        sb = self.io.b[W - 1]
        mag_a = self.io.a[0 : W - 1]  # make magnitude unsigned
        mag_b = self.io.b[0 : W - 1]  # make magnitude unsigned

        # mag_y = mag_a * mag_b
        # use the specified multiplier
        mult.io.a <<= mag_a
        mult.io.b <<= mag_b
        mag_y = mult.io.y

        # sign
        sy = sa ^ sb

        # make sure sign is positive if value is zero
        is_zero = mux(mag_y == 0, Const(True, Bool()), Const(False, Bool()))
        sy = mux(is_zero, Const(False, Bool()), sy)

        self.io.y <<= Concat([mag_y[0 : 2 * W - 2], sy])  # sign + magnitude (drop overflow bit)


class OptimizedMultiplierFrom4BitBlocks(StageBasedMultiplierBase):

    def __init__(self, *args, f_aag_lines: "Optional[Callable[..., List[str]]]" = None, use_compressor_tree=True, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # assign default PPA/FSA if not provided
        if use_compressor_tree:
            if self.ppa_cls is None:
                self.ppa_cls = CarrySaveAccumulator  # smaller depth same aig count for 8 bit multiplier
            if self.fsa_cls is None:
                self.fsa_cls = RippleCarryFinalAdder

        # Compose four 8x8 optimized unsigned multipliers into a 16x16->32 multiplier
        assert self.a_encoding == Encoding.unsigned and self.b_encoding == Encoding.unsigned, "Only unsigned encoding is supported"
        # aw and bw must be equal (the p1/p3 column shifts are hardcoded for square operands;
        # a custom f_aag_lines with aw != bw would silently mis-weight partial products),
        # >= 4, and powers of two.
        assert self.aw == self.bw, "This composite multiplier expects square operands (aw == bw)"
        assert self.aw >= 4 and self.bw >= 4 and ((self.aw & (self.aw - 1)) == 0) and ((self.bw & (self.bw - 1)) == 0), "This composite multiplier expects input widths >=4 and powers of two"

        self.io: StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Input(UInt(self.aw)),
            b=Input(UInt(self.bw)),
            y=Output(UInt(self.aw + self.bw)),
        )

        # check class attribut f_aag_lines exists
        if hasattr(self, 'f_aag_lines'):
            if f_aag_lines is not None:
                raise ValueError("f_aag_lines class attribute exists, do not pass f_aag_lines per argument")                
        else:    
            self.f_aag_lines = f_aag_lines

        self.use_compressor_tree = use_compressor_tree # results are better with compressor tree

        self.elaborate()

    def elaborate(self) -> None:

        if self.aw == 4 and self.bw == 4:
            multiplier = OptimizedMultiplier(
                a_encoding=Encoding.unsigned,
                b_encoding=Encoding.unsigned,
                a_w=self.aw,
                b_w=self.bw,
                ppg_cls=None,
                ppa_cls=None,
                fsa_cls=None,
                f_aag_lines=self.f_aag_lines,
            )

            # wire inputs/outputs
            multiplier.io.a <<= self.io.a
            multiplier.io.b <<= self.io.b
            self.io.y <<= multiplier.io.y
            return
        else:

            # Split operands into 8-bit halves: a = a_hi<<8 + a_lo; b = b_hi<<8 + b_lo
            a_lo = self.io.a[0 : self.aw//2]
            a_hi = self.io.a[self.aw//2 : self.aw]
            b_lo = self.io.b[0 : self.bw//2]
            b_hi = self.io.b[self.bw//2 : self.bw]

            # Instantiate four optimized 8x8 multipliers
            multipliers: List[StageBasedMultiplierBase] = []
            mult_cls = OptimizedMultiplierFrom4BitBlocks if (self.aw//2 > 4 and self.bw//2 > 4) else OptimizedMultiplier

            for _ in range(4):
                multipliers.append(
                    mult_cls(
                        a_encoding=Encoding.unsigned,
                        b_encoding=Encoding.unsigned,
                        a_w=self.aw//2,
                        b_w=self.bw//2,
                        ppg_cls=None,
                        ppa_cls=None,
                        fsa_cls=None,
                        f_aag_lines=self.f_aag_lines,
                    )
                )
            m_ll, m_lh, m_hl, m_hh = multipliers

            # Wire inputs
            m_ll.io.a <<= a_lo
            m_ll.io.b <<= b_lo

            m_lh.io.a <<= a_lo
            m_lh.io.b <<= b_hi

            m_hl.io.a <<= a_hi
            m_hl.io.b <<= b_lo

            m_hh.io.a <<= a_hi
            m_hh.io.b <<= b_hi

            # Partial products (each 16 bits wide)
            p0 = m_ll.io.y  # a_lo * b_lo
            p1 = m_lh.io.y  # a_lo * b_hi
            p2 = m_hl.io.y  # a_hi * b_lo
            p3 = m_hh.io.y  # a_hi * b_hi

            if not self.use_compressor_tree:

                # Zero-extend to 2*aw bits and align via left shifts using Concat
                p0 = Concat([p0, Const(0, UInt(self.aw))])  # no shift

                p1 = Concat([p1, Const(0, UInt(self.aw))])
                p1_sh8 = Concat([Const(0, UInt(self.aw//2)), p1[0:self.aw*3//2]])  # (p1 << aw//2)

                p2 = Concat([p2, Const(0, UInt(self.aw))])
                p2_sh8 = Concat([Const(0, UInt(self.aw//2)), p2[0:self.aw*3//2]])  # (p2 << aw//2)

                p3_sh16 = Concat([Const(0, UInt(self.aw)), p3])  # (p3 << aw)

                # Sum partial products
                self.io.y <<= p0 + p1_sh8 + p2_sh8 + p3_sh16

            else:

                # use of compression tree
                # generate dict with Dict[int, Expr] with int indicating bit weight
                cols: DefaultDict[int, List[Expr]] = defaultdict(list)
                for i in range(p0.typ.width):
                    cols[i].append(p0[i])
                for i in range(p1.typ.width):
                    cols[i + self.aw//2].append(p1[i])
                for i in range(p2.typ.width):
                    cols[i + self.aw//2].append(p2[i])
                for i in range(p3.typ.width):
                    cols[i + self.aw].append(p3[i])

                config = StageMultiplierConfig(
                    a_width=self.aw,
                    b_width=self.bw,
                    signed_a=False,
                    signed_b=False,
                    optim_type=self.optim_type,
                )

                ppa = self.ppa_cls(config=config)
                ppa_cols = ppa.accumulate(cols)
                fsa = self.fsa_cls(config=config)
                fsa_bits = fsa.resolve(ppa_cols)

                # another option is to use compressor_sum for the summation
                # # Zero-extend to 2*aw bits and align via left shifts using Concat
                # p0 = Concat([p0, Const(0, UInt(self.aw))])  # no shift

                # p1 = Concat([p1, Const(0, UInt(self.aw))])
                # p1_sh8 = Concat([Const(0, UInt(self.aw//2)), p1[0:self.aw*3//2]])  # (p1 << aw//2)

                # p2 = Concat([p2, Const(0, UInt(self.aw))])
                # p2_sh8 = Concat([Const(0, UInt(self.aw//2)), p2[0:self.aw*3//2]])  # (p2 << aw//2)

                # p3_sh16 = Concat([Const(0, UInt(self.aw)), p3])  # (p3 << aw)

                # fsa_bits = compressor_sum(
                #     config=OutputConfig(
                #         out_width=self.aw + self.bw,
                #         optim_type=self.optim_type,
                #     ),
                #     partials=[p0, p1_sh8, p2_sh8, p3_sh16],
                #     ppg_cls=CarrySaveAccumulator,
                #     fsa_cls=RippleCarryFinalAdder,
                # )

                self.io.y <<= Concat(fsa_bits)


class OptimizedMultiplierFrom4BitBlocksStrong(OptimizedMultiplierFrom4BitBlocks):
    f_aag_lines = staticmethod(get_optimized_aag_lines_strong)


def test_multiplier_ext_optimized() -> None:
    n_bits = 8
    signed = False

    c = OptimizedMultiplier(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=to_encoding(signed),
        b_encoding=to_encoding(signed),
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
        optim_type="area",
    )
    mod = c.to_module("multiplier_ext_optimized")
    print(mod)

    module = mod
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
    yosys_metrics = get_yosys_metrics(module)
    aig_gates = get_aig_stats(module)
    print(f"Yosys-reported transistor count: {transistor_count}")
    print(f"Yosys-reported metrics: {yosys_metrics}")
    print(f"AIG-reported gate count: {aig_gates}")

    vecs = MultiplierTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        y_w=2 * n_bits,
        num_vectors=16,
        tb_sigma=None,
        a_encoding=to_encoding(signed),
        b_encoding=to_encoding(signed),
        y_encoding=to_encoding(signed),
    ).generate()

    run_vectors(module, vecs)

    # sign magnitude version

    n_bits = 4

    from spire.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import MultiplierOption, encoding_for_multiplier

    encodings = encoding_for_multiplier(MultiplierOption.STAGE_BASED_SIGN_MAGNITUDE_MULTIPLIER.value)[0]

    c = OptimizedSignMagnitudeMultiplier(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=encodings.a,
        b_encoding=encodings.b,
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
    )
    mod = c.to_module("multiplier_ext_optimized_sign_magnitude")
    print(mod)
    module = mod
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
    yosys_metrics = get_yosys_metrics(module)
    aig_gates = get_aig_stats(module)
    print(f"Yosys-reported transistor count: {transistor_count}")
    print(f"Yosys-reported metrics: {yosys_metrics}")
    print(f"AIG-reported gate count: {aig_gates}")

    vecs = MultiplierTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        y_w=2 * n_bits - 1,
        num_vectors=16,
        tb_sigma=None,
        a_encoding=encodings.a,
        b_encoding=encodings.b,
        y_encoding=encodings.y,
    ).generate()

    run_vectors(module, vecs)

    # assembled from 4 bit blocks

    n_bits = 8

    c = OptimizedMultiplierFrom4BitBlocks(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned,
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
    )
    mod = c.to_module("multiplier_ext_optimized_8x8_from_4x4")
    print(mod)
    module = mod
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
    yosys_metrics = get_yosys_metrics(module)
    aig_gates = get_aig_stats(module)
    print(f"Yosys-reported transistor count: {transistor_count}")
    print(f"Yosys-reported metrics: {yosys_metrics}")
    print(f"AIG-reported gate count: {aig_gates}")

    vecs = MultiplierTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        y_w=2 * n_bits,
        num_vectors=16,
        tb_sigma=None,
        a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned,
        y_encoding=Encoding.unsigned,
    ).generate()
    run_vectors(module, vecs)

    # use a different file

    n_bits = 8
    signed = False

    c = OptimizedMultiplierFrom4BitBlocksStrong(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=to_encoding(signed),
        b_encoding=to_encoding(signed),
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
        optim_type="area",
    )

    mod = c.to_module("multiplier_ext_optimized")
    print(mod)

    module = mod
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
    yosys_metrics = get_yosys_metrics(module)
    aig_gates = get_aig_stats(module)
    print(f"Yosys-reported transistor count: {transistor_count}")
    print(f"Yosys-reported metrics: {yosys_metrics}")
    print(f"AIG-reported gate count: {aig_gates}")

    vecs = MultiplierTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        y_w=2 * n_bits,
        num_vectors=16,
        tb_sigma=None,
        a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned,
        y_encoding=Encoding.unsigned,
    ).generate()
    run_vectors(module, vecs)
    
    n_bits = 16
    signed = False
    
    c = OptimizedMultiplierFrom4BitBlocksStrong(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=to_encoding(signed),
        b_encoding=to_encoding(signed),
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
        optim_type="area",
    )
    
    mod = c.to_module("multiplier_ext_optimized")
    print(mod)
    
    module = mod
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
    yosys_metrics = get_yosys_metrics(module)
    aig_gates = get_aig_stats(module)
    print(f"Yosys-reported transistor count: {transistor_count}")
    print(f"Yosys-reported metrics: {yosys_metrics}")
    print(f"AIG-reported gate count: {aig_gates}")
    
    vecs = MultiplierTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        y_w=2 * n_bits,
        num_vectors=16,
        tb_sigma=None,
        a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned,
        y_encoding=Encoding.unsigned,
    ).generate()
    run_vectors(module, vecs)


if __name__ == "__main__":
    test_multiplier_ext_optimized()
