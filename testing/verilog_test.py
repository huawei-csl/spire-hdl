import tempfile

from aigverse import equivalence_checking, read_verilog_into_aig
from spirehdl.aig.aig_aigerverse import _get_aag_sym, conv_aag_into_aig, conv_aig_into_aag, file_to_lines, read_aag_into_aig
from spirehdl.spirehdl import UInt
from spirehdl.spirehdl_aiger import AigerExporter, AigerImporter
from spirehdl.arithmetic.floating_point.spire_hdl_float_mult import run_vectors_aby
from spirehdl.arithmetic.floating_point.spire_hdl_float_mult_sn import build_fp_mul_sn
from spirehdl.spirehdl_module import IOCollector

# import pyosys as ys, tempfile, os
import os
from pyosys import libyosys as ys

from spirehdl.arithmetic.floating_point.fp_encoding import fp_decode
from spirehdl.arithmetic.floating_point.fp_mul_testvectors import build_fp_vectors



def main():
    ew=5
    subnormals = False
    # fw = 16 - ew - 1
    fw = 3
    bits_tot = 1 + ew + fw

    m = build_fp_mul_sn("F16Mul", EW=ew, FW=fw, subnormals=subnormals)

    # export to verilog to tempfile file (virtual file)
    # with tempfile.NamedTemporaryFile(delete=True) as tmp:
    #     tmp.write(m.to_verilog().encode())
    #     tmp.flush()
    #     print(f"Verilog module written to {tmp.name}")

    #     aig3 = read_verilog_into_aig(tmp.name)

    verilog = m.to_verilog()

    aag = AigerExporter(m).get_aag()
    aig = conv_aag_into_aig(aag)

    if False:
        # from aig
        aag = AigerExporter(m).get_aag()
        aig = conv_aag_into_aig(aag)

        aag_sym = _get_aag_sym(aag)
        m2 = AigerImporter(aag[:-2]+aag_sym).get_spirehdl_module()

        aag2 = AigerExporter(m2).get_aag()
        aig2 = conv_aag_into_aig(aag2)

        collector = IOCollector()
        collector.group(m2, {
            "a": UInt(bits_tot),
            "b": UInt(bits_tot),
            "y": UInt(bits_tot),
        })

    # print(m.to_verilog())
    # verilog = m.to_verilog()

    file_name = "fxmul.v"

    # export to verilog file
    with open(file_name, "w") as f:
        f.write(verilog)

    # export to verilog file
    # with open("f16mul_aig.v", "w") as f:
    #      f.write(m2.to_verilog())

    def verilog_to_aig_pyosys(verilog_path, top=None):
        fd, aag_path = tempfile.mkstemp(suffix=".aag"); os.close(fd)
        ys.run_pass(f"read_verilog {verilog_path}")
        ys.run_pass("hierarchy -check " + ("-auto-top" if top is None else f"-top {top}"))
        ys.run_pass("synth -flatten")
        ys.run_pass("aigmap")
        ys.run_pass(f"write_aiger -ascii {aag_path}")
        return aag_path

    def verilog_to_aig_pyosys(
        verilog_path: str,
        top: str | None = None,
        *,
        mapped_verilog_path: str | None = None,  # e.g. "mapped.v" to write the aigmapped netlist
        aag_out_path: str | None = None,         # optionally choose your .aag path
        tie_undriven: str | None = None          # one of {"zero","one","random"} if you want setundef
    ):
        """
        Convert Verilog -> (aigmap) -> AIGER, and optionally dump the AIG-mapped netlist.
    
        Returns (aig_obj, aag_path, mapped_v_path_or_None).
        """
        # Choose output AAG path if not provided
        if aag_out_path is None:
            fd, aag_out_path = tempfile.mkstemp(suffix=".aag")
            os.close(fd)

        if mapped_verilog_path is None:
            fd, mapped_verilog_path = tempfile.mkstemp(suffix=".v")
            os.close(fd)

        # Load and prepare
        ys.run_pass(f"read_verilog {verilog_path}")
        ys.run_pass("hierarchy -check " + ("-auto-top" if top is None else f"-top {top}"))
        ys.run_pass("synth -flatten")

        # # Optional: resolve undriven signals deterministically to avoid $anyseq in AIGER
        # if tie_undriven in {"zero", "one", "random"}:
        #     flag = {"zero": "-zero", "one": "-one", "random": "-random"}[tie_undriven]
        #     ys.run_pass(f"setundef -undriven {flag}")

        # Map to AIG (AND/INV)
        ys.run_pass("aigmap")

        # Optional: dump the aig-mapped netlist (gate-level Verilog with $_AND_/$_NOT_)
        if mapped_verilog_path:
            ys.run_pass(f"write_verilog -noexpr -attr2comment {mapped_verilog_path}")

        # Emit AIGER and load with py-aiger
        ys.run_pass(f"write_aiger -ascii {aag_out_path}")

        return aag_out_path, (mapped_verilog_path or None)

    def verilog_to_aig_pyosys(
        verilog_path: str,
        top: str | None = None,
        *,
        aag_out_path: str | None = None,
        tie_undriven: str | None = None,      # {"zero","one","random"} or None
        map_out_path: str | None = None,      # e.g. "design.aigmap"  (write_aiger -map)
        vmap_out_path: str | None = None,     # e.g. "design.vmap"    (write_aiger -vmap)
        embed_symbols: bool = True,          # write_aiger -symbols (names inside .aag, at the end of the file)
        no_startoffset: bool = True          # write_aiger -no-startoffset (0-based idx)
    ):
        """
        Convert Verilog -> AIGER, optionally writing an AIGER map with IO names.
    
        Returns (aig_obj, aag_path, map_path, vmap_path).
        """
        if aag_out_path is None:
            fd, aag_out_path = tempfile.mkstemp(suffix=".aag");
            os.close(fd)

        # if mapped_verilog_path is None:
        #     fd, mapped_verilog_path = tempfile.mkstemp(suffix=".v")
        #     os.close(fd)

        if map_out_path is None:
            fd, map_out_path = tempfile.mkstemp(suffix=".map");
            os.close(fd)

        # if vmap_out_path is None:
        #     fd, vmap_out_path = tempfile.mkstemp(suffix=".vmap");
        #     os.close(fd)

        ys.run_pass(f"read_verilog {verilog_path}")
        ys.run_pass("hierarchy -check " + ("-auto-top" if top is None else f"-top {top}"))
        ys.run_pass("synth -flatten")

        if tie_undriven in {"zero", "one", "random"}:
            flag = {"zero": "-zero", "one": "-one", "random": "-random"}[tie_undriven]
            ys.run_pass(f"setundef -undriven {flag}")

        ys.run_pass("aigmap")

        # Build write_aiger command with requested options
        opts = ["-ascii"]
        if embed_symbols:
            opts.append("-symbols")
        if no_startoffset:
            opts.append("-no-startoffset")
        if map_out_path:
            opts += ["-map", map_out_path]
        if vmap_out_path:
            opts += ["-vmap", vmap_out_path]

        ys.run_pass(f"write_aiger {' '.join(opts)} {aag_out_path}")

        return aag_out_path, map_out_path, vmap_out_path

    aag_path, map_out_path, vmap_out_path = verilog_to_aig_pyosys(file_name, top="F16Mul")

    # aag to aig
    aag_back = file_to_lines(aag_path)
    aig_back = read_aag_into_aig(aag_path)

    m_back = AigerImporter(aag_back).get_spirehdl_module()
    collector = IOCollector()
    spirehdl_collected = collector.group(
        m_back,
        {
            "a": UInt(bits_tot),
            "b": UInt(bits_tot),
            "y": UInt(bits_tot),
        },
    )

    # read back temp verilog file into AIG
    # aig3 = read_verilog_into_aig(file_name)

    # Print the size of the unoptimized and optimized AIGs
    print(f"Original AIG Size:  {aig.size()}")
    print(f"Back AIG Size: {aig_back.size()}")

    # simulations:
    run_vectors_aby(m, build_fp_vectors(ew, fw), label=f"float{ew+fw+1} normal cases", decoder=lambda b: fp_decode(b, ew, fw))
    run_vectors_aby(m_back, build_fp_vectors(ew, fw), label=f"float{ew+fw+1} normal cases", decoder=lambda b: fp_decode(b, ew, fw))

    assert equivalence_checking(aig, aig_back), "AIGs are not equivalent after conversion!"


if __name__ == "__main__":
    main()