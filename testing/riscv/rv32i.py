# Minimal RV32I (single-cycle) core in SpireHDL + tiny assembler + simple simulation.

from typing import List, Tuple, Dict

from spire.helpers import get_yosys_transistor_count
from spire.component import Module
from spire.expr import UInt, Bool, Const, mux, cat, fit_width
from spire.simulator import Simulator



# -----------------------
# Helpers: immediates
# -----------------------
def _sext_cat(msb, bits_expr, total_w):
    """Sign-extend 'bits_expr' to 'total_w' via concatenating copies of 'msb'."""
    need = total_w - bits_expr.typ.width
    if need <= 0:
        return bits_expr
    return cat(bits_expr, *([msb] * need))

def imm_i32(instr):
    # I-immediate: instr[20:32], sign-extended to 32
    imm12 = instr[20:32]
    s = instr[31]
    return _sext_cat(s, imm12, 32)

def imm_s32(instr):
    # S-immediate: instr[7:12] (low) | instr[25:32] (high), sign-extended to 32
    bits = cat(instr[7:12], instr[25:32])
    s = instr[31]
    return _sext_cat(s, bits, 32)

def imm_b32(instr):
    # B-immediate: imm[12|10:5|4:1|11|0]
    #   12 = instr[31], 11 = instr[7], 10:5 = instr[25:31], 4:1 = instr[8:12], 0 = 0
    s = instr[31]
    bits13 = cat(Const(0, UInt(1)), instr[8:12], instr[25:31], instr[7], instr[31])
    return _sext_cat(s, bits13, 32)


# -----------------------
# Helpers: regfile read
# -----------------------
def read_reg(regs: List, idx_expr):
    """32→1 mux over regs[0..31]."""
    acc = Const(0, UInt(32))
    for i in range(32):
        acc = mux(idx_expr == i, regs[i], acc)
    return acc


# -----------------------
# Core builder
# -----------------------
def build_rv32i_simple(name="RV32I_Simple"):
    """
    Ports:
      in  instr[31:0]            -- instruction at imem_addr (combinational)
      in  dmem_rdata[31:0]       -- read word from data memory @ dmem_addr (combinational)
      out imem_addr[31:0]        -- PC (byte address)
      out dmem_addr[31:0]        -- byte address for load/store
      out dmem_wdata[31:0]       -- store data
      out dmem_we                -- store enable
      out halt                   -- 1 on EBREAK
    """
    m = Module(name, with_clock=True, with_reset=True)

    # ---- Ports ----
    instr       = m.input(UInt(32), "instr")
    dmem_rdata  = m.input(UInt(32), "dmem_rdata")

    imem_addr   = m.output(UInt(32), "imem_addr")
    dmem_addr   = m.output(UInt(32), "dmem_addr")
    dmem_wdata  = m.output(UInt(32), "dmem_wdata")
    dmem_we     = m.output(Bool(),   "dmem_we")
    halt        = m.output(Bool(),   "halt")

    # ---- State ----
    pc = m.reg(UInt(32), "pc", init=0)

    # 32 general-purpose registers (x0 is hard-wired to zero by construction)
    regs = [m.reg(UInt(32), f"x{i}", init=0) for i in range(32)]

    # ---- Decode fields ----
    opcode = instr[0:7]          # instr[0:7]
    rd     = instr[7:12]         # instr[7:12]
    funct3 = instr[12:15]        # instr[12:15]
    rs1    = instr[15:20]        # instr[15:20]
    rs2    = instr[20:25]        # instr[20:25]
    funct7 = instr[25:32]        # instr[25:32]

    # Opcodes
    is_rtype  = (opcode == 0x33)
    is_itype  = (opcode == 0x13)
    is_load   = (opcode == 0x03)  # LW (funct3=010)
    is_store  = (opcode == 0x23)  # SW (funct3=010)
    is_branch = (opcode == 0x63)  # BEQ/BNE
    is_system = (opcode == 0x73)  # EBREAK/ECALL/CSR
    is_ebreak = (instr == 0x00100073)  # exact encoding

    # Read sources
    rs1_val = read_reg(regs, rs1)
    rs2_val = read_reg(regs, rs2)

    # Immediates
    i_imm = imm_i32(instr)
    s_imm = imm_s32(instr)
    b_imm = imm_b32(instr)

    # ALU (minimal ops: ADD, SUB; ADDI; plus SLTU for simple comparisons)
    is_sub  = (is_rtype & (funct3 == 0) & (funct7 == 0x20))
    is_slt  = (is_rtype & (funct3 == 2) & (funct7 == 0x00))  # SLT (treated as unsigned here)
    is_sltu = (is_rtype & (funct3 == 3) & (funct7 == 0x00))  # SLTU
    # operand B: for I-type arith use immediate, else rs2
    opB = mux(is_itype, i_imm, rs2_val)

    add_res = fit_width(rs1_val + opB, UInt(32))
    sub_res = fit_width(rs1_val - rs2_val, UInt(32))
    # Less-than result as 0/1 extended to 32 bits
    lt_bit  = (rs1_val < rs2_val)
    lt_res  = mux(lt_bit, Const(1, UInt(32)), Const(0, UInt(32)))
    alu_sub_add = mux(is_sub, sub_res, add_res)
    alu_res = mux((is_slt | is_sltu), lt_res, alu_sub_add)

    # Data memory interface
    addr_imm = mux(is_store, s_imm, i_imm)           # S uses s_imm; I (LW) uses i_imm
    daddr    = fit_width(rs1_val + addr_imm, UInt(32))

    dmem_addr <<= daddr
    dmem_wdata <<= rs2_val
    dmem_we <<= is_store

    # Instruction memory address = PC
    imem_addr <<= pc

    # Branch decision (BEQ/BNE only)
    beq = (funct3 == 0) & (rs1_val == rs2_val)
    bne = (funct3 == 1) & (rs1_val != rs2_val)
    take_branch = is_branch & (beq | bne)

    pc_plus4   = fit_width(pc + 4, UInt(32))
    pc_br_tgt  = fit_width(pc + b_imm, UInt(32))
    pc_next    = mux(is_ebreak, pc, mux(take_branch, pc_br_tgt, pc_plus4))
    pc         <<= pc_next

    # Register writeback:
    wb_from_mem = is_load
    wb_data     = mux(wb_from_mem, dmem_rdata, alu_res)
    reg_we      = (is_rtype | is_itype | is_load) & (~is_system)

    for i in range(32):
        if i == 0:
            # x0 is hard-wired zero
            regs[i] <<= Const(0, UInt(32))
        else:
            we_i = reg_we & (rd == i)
            regs[i] <<= mux(we_i, wb_data, regs[i])

    # Halt signal on EBREAK
    halt <<= is_ebreak

    return m


# -----------------------
# Encoders (machine words)
# -----------------------
def ADDI(rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (0 << 12) | (rd << 7) | 0x13

def ADD(rd, rs1, rs2):
    return (0x00 << 25) | (rs2 << 20) | (rs1 << 15) | (0 << 12) | (rd << 7) | 0x33

def SUB(rd, rs1, rs2):
    return (0x20 << 25) | (rs2 << 20) | (rs1 << 15) | (0 << 12) | (rd << 7) | 0x33

def LW(rd, rs1, imm):
    # funct3=010 for LW
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (2 << 12) | (rd << 7) | 0x03

def SLT(rd, rs1, rs2):
    # R-type, funct3=010
    return (0x00 << 25) | (rs2 << 20) | (rs1 << 15) | (2 << 12) | (rd << 7) | 0x33

def SLTU(rd, rs1, rs2):
    # R-type, funct3=011
    return (0x00 << 25) | (rs2 << 20) | (rs1 << 15) | (3 << 12) | (rd << 7) | 0x33

def SW(rs2, rs1, imm):
    # funct3=010 for SW; imm split into [11:5] and [4:0]
    imm12 = imm & 0xFFF
    imm11_5 = (imm12 >> 5) & 0x7F
    imm4_0  = imm12 & 0x1F
    return (imm11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (2 << 12) | (imm4_0 << 7) | 0x23

def BEQ(rs1, rs2, bytes_off):
    off = bytes_off
    imm12  = (off >> 12) & 0x1
    imm10_5= (off >> 5) & 0x3F
    imm4_1 = (off >> 1) & 0xF
    imm11  = (off >> 11) & 0x1
    return (imm12 << 31) | (imm10_5 << 25) | (rs2 << 20) | (rs1 << 15) | (0 << 12) | (imm4_1 << 8) | (imm11 << 7) | 0x63

def BNE(rs1, rs2, bytes_off):
    off = bytes_off
    imm12  = (off >> 12) & 0x1
    imm10_5= (off >> 5) & 0x3F
    imm4_1 = (off >> 1) & 0xF
    imm11  = (off >> 11) & 0x1
    return (imm12 << 31) | (imm10_5 << 25) | (rs2 << 20) | (rs1 << 15) | (1 << 12) | (imm4_1 << 8) | (imm11 << 7) | 0x63

def EBREAK():
    return 0x00100073


# -----------------------
# Tiny assembler
# -----------------------
_REG_ALIAS: Dict[str, int] = {
    # base names
    **{f"x{i}": i for i in range(32)},
    # canonical aliases
    "zero":0,"ra":1,"sp":2,"gp":3,"tp":4,
    "t0":5,"t1":6,"t2":7,"s0":8,"s1":9,
    "a0":10,"a1":11,"a2":12,"a3":13,"a4":14,"a5":15,"a6":16,"a7":17,
    "s2":18,"s3":19,"s4":20,"s5":21,"s6":22,"s7":23,"s8":24,"s9":25,"s10":26,"s11":27,
    "t3":28,"t4":29,"t5":30,"t6":31,
    # extra alias
    "fp":8,
}

def _parse_reg(tok: str) -> int:
    t = tok.strip().lower()
    if t in _REG_ALIAS:
        return _REG_ALIAS[t]
    raise ValueError(f"Unknown register '{tok}'")

def _parse_int(tok: str) -> int:
    tok = tok.strip().lower()
    # allow +/- decimal, hex 0x, binary 0b
    if tok.startswith(("+0x","-0x","0x","+0b","-0b","0b")):
        return int(tok, 0)
    # plain int (decimal)
    return int(tok, 0)

def _strip_comment(line: str) -> str:
    # strip // or # or ; comments
    for sep in ("//", "#", ";"):
        p = line.find(sep)
        if p != -1:
            line = line[:p]
    return line.strip()

def _split_operands(opstr: str) -> List[str]:
    # Split by commas, trim spaces
    # e.g. "x1, x2, 0(x3)" -> ["x1","x2","0(x3)"]
    return [t.strip() for t in opstr.split(",") if t.strip()]

def _parse_mem(operand: str) -> Tuple[int, int]:
    # offset(rs) syntax, e.g. "16(x2)" or "-4(x10)"
    s = operand.replace(" ", "")
    if "(" not in s or not s.endswith(")"):
        raise ValueError(f"Bad memory operand '{operand}', expected off(rs)")
    off_str, rs_str = s.split("(", 1)
    rs_str = rs_str[:-1]
    off = _parse_int(off_str)
    rs = _parse_reg(rs_str)
    return off, rs

def assemble(lines: List[str], *, origin: int = 0) -> List[int]:
    """
    Two-pass assembler for a tiny RV32I subset.
    Returns a list of machine words (ints).
    """
    # Pass 1: collect labels
    labels: Dict[str, int] = {}
    pc = origin
    items: List[Tuple[int, str]] = []  # (pc, normalized_inst)
    for raw in lines:
        s = _strip_comment(raw)
        if not s:
            continue
        while True:
            # support multiple labels on the same line: L1: L2: ADDI x1,x0,1
            if ":" in s:
                head, tail = s.split(":", 1)
                label = head.strip()
                if label:
                    if label in labels:
                        raise ValueError(f"Duplicate label '{label}'")
                    labels[label] = pc
                    s = tail.strip()
                    if not s:
                        break
                    continue
            break
        if s:
            items.append((pc, s))
            pc += 4

    # Pass 2: encode
    words: List[int] = []
    for pc, s in items:
        # separate mnemonic and operands
        parts = s.split(None, 1)
        mnem = parts[0].upper()
        ops  = _split_operands(parts[1]) if len(parts) > 1 else []

        def beq_off(tok: str) -> int:
            # label or immediate (bytes). RISC-V branch adds immediate to current PC.
            if tok in labels:
                return labels[tok] - pc
            v = _parse_int(tok)
            return v

        if mnem == "NOP":
            words.append(ADDI(0, 0, 0))
        elif mnem in ("EBREAK","HALT"):
            words.append(EBREAK())

        elif mnem == "ADDI":
            if len(ops) != 3: raise ValueError("ADDI rd, rs1, imm")
            rd  = _parse_reg(ops[0]); rs1 = _parse_reg(ops[1]); imm = _parse_int(ops[2])
            if not (-2048 <= imm <= 2047): raise ValueError("ADDI imm out of 12-bit signed range")
            words.append(ADDI(rd, rs1, imm))

        elif mnem == "ADD":
            if len(ops) != 3: raise ValueError("ADD rd, rs1, rs2")
            rd  = _parse_reg(ops[0]); rs1 = _parse_reg(ops[1]); rs2 = _parse_reg(ops[2])
            words.append(ADD(rd, rs1, rs2))

        elif mnem == "SUB":
            if len(ops) != 3: raise ValueError("SUB rd, rs1, rs2")
            rd  = _parse_reg(ops[0]); rs1 = _parse_reg(ops[1]); rs2 = _parse_reg(ops[2])
            words.append(SUB(rd, rs1, rs2))

        elif mnem == "SLT":
            if len(ops) != 3: raise ValueError("SLT rd, rs1, rs2")
            rd  = _parse_reg(ops[0]); rs1 = _parse_reg(ops[1]); rs2 = _parse_reg(ops[2])
            words.append(SLT(rd, rs1, rs2))

        elif mnem == "SLTU":
            if len(ops) != 3: raise ValueError("SLTU rd, rs1, rs2")
            rd  = _parse_reg(ops[0]); rs1 = _parse_reg(ops[1]); rs2 = _parse_reg(ops[2])
            words.append(SLTU(rd, rs1, rs2))

        elif mnem == "LW":
            if len(ops) != 2: raise ValueError("LW rd, offset(rs1)")
            rd = _parse_reg(ops[0])
            off, rs1 = _parse_mem(ops[1])
            if not (-2048 <= off <= 2047): raise ValueError("LW offset out of 12-bit signed range")
            words.append(LW(rd, rs1, off))

        elif mnem == "SW":
            if len(ops) != 2: raise ValueError("SW rs2, offset(rs1)")
            rs2 = _parse_reg(ops[0])
            off, rs1 = _parse_mem(ops[1])
            if not (-2048 <= off <= 2047): raise ValueError("SW offset out of 12-bit signed range")
            words.append(SW(rs2, rs1, off))

        elif mnem in ("BEQ","BNE"):
            if len(ops) != 3: raise ValueError(f"{mnem} rs1, rs2, label|imm")
            rs1 = _parse_reg(ops[0]); rs2 = _parse_reg(ops[1]); off = beq_off(ops[2])
            if (off & 0x1) != 0:
                raise ValueError("Branch offset must be 2-byte aligned")
            if not (-4096 <= off <= 4094):
                raise ValueError("Branch offset out of range (~±4KB)")
            words.append(BEQ(rs1, rs2, off) if mnem == "BEQ" else BNE(rs1, rs2, off))

        else:
            raise ValueError(f"Unknown mnemonic '{mnem}' in: {s}")

    return words


# -----------------------
# Simple simulation (now with assembly program)
# -----------------------
def simulate_demo():
    if Simulator is None:
        raise RuntimeError("Simulator class not found. Import your SpireHDL simulator and retry.")
    
    print("Running simple RV32I simulation demo...")

    # Program (assembly):
    prog_asm = [
        "ADDI x1, x0, 2",         # x1 = 2
        "ADDI x2, x0, 3",         # x2 = 3
        "ADD  x3, x1, x2",        # x3 = 5
        "SW   x3, 0(x0)",         # mem[0] = 5
        "LW   x4, 0(x0)",         # x4 = 5
        "BEQ  x3, x4, done",      # if equal, skip next
        "ADDI x5, x0, 99",        # would be skipped
        "done: EBREAK",           # halt
    ]
    prog = assemble(prog_asm)

    # Instantiate core
    m = build_rv32i_simple("RV32I_Simple")
    #transistor_count = get_yosys_transistor_count(m, n_iter_optimizations=0)
    #print("Estimated transistor count:", transistor_count)
    sim = Simulator(m)

    # Memories
    dmem = bytearray(256)  # tiny data memory
    max_cycles = 100

    # Reset: assert then deassert
    sim.reset(True).eval()
    sim.deassert_reset().eval()

    def imem_fetch(pc_val: int) -> int:
        idx = (pc_val >> 2)
        return prog[idx] if 0 <= idx < len(prog) else 0

    # Run
    print("Assembled program 1:")
    for i, w in enumerate(prog):
        print(f"  {4*i:04x}: 0x{w:08x}   | {prog_asm[i] if i < len(prog_asm) else ''}")

    for cycle in range(max_cycles):
        # Phase A: drive instr for current PC
        pc_val = sim.get("pc")
        inst = imem_fetch(pc_val)
        sim.set("instr", inst)

        # First eval to compute dmem address/control
        sim.eval()

        # Phase B: drive data memory read combinationally
        addr = sim.get("dmem_addr") & ~0x3
        rword = int.from_bytes(dmem[addr:addr+4], "little")
        sim.set("dmem_rdata", rword)

        # Evaluate again to let LW data propagate into writeback
        sim.eval()

        # Snapshot some signals
        we   = sim.get("dmem_we")
        wdat = sim.get("dmem_wdata")
        halt = sim.get("halt")

        # Rising edge: commit store (use values from this cycle)
        if we:
            dmem[addr:addr+4] = (wdat & 0xFFFFFFFF).to_bytes(4, "little")

        # Advance state
        sim.step(1)

        # Print a brief trace
        x1 = sim.get("x1"); x2 = sim.get("x2"); x3 = sim.get("x3"); x4 = sim.get("x4")
        print(f"t={cycle:02d}  PC=0x{pc_val:08x}  instr=0x{inst:08x}  x1={x1} x2={x2} x3={x3} x4={x4}  we={we} addr={addr} w=0x{wdat&0xffffffff:08x}")
        if halt:
            break

    print(f"Simulation ended at cycle {cycle}, PC=0x{sim.get('pc'):08x}, halt={halt}")
    
    # Final checks and simple assertions
    # - x1=2, x2=3, x3=x1+x2=5, store/load path works (DMEM[0]=5, x4=5)
    # - BEQ taken to label 'done' so ADDI x5,99 is skipped (x5 remains 0)
    # - PC should point at the EBREAK instruction (index 7 -> byte addr 28)
    final_word = int.from_bytes(dmem[0:4], "little")
    x1 = sim.get("x1"); x2 = sim.get("x2"); x3 = sim.get("x3"); x4 = sim.get("x4"); x5 = sim.get("x5")
    pc_final = sim.get("pc")

    # Print a short summary for visibility when run as a script
    print(f"DMEM[0..3] = {list(dmem[0:4])}  (word={final_word})")
    print("Registers:", {f"x{i}": sim.get(f'x{i}') for i in range(8)})

    # Assertions (will raise if any basic behavior regresses)
    assert x1 == 2, f"Expected x1=2 after ADDI, got {x1}"
    assert x2 == 3, f"Expected x2=3 after ADDI, got {x2}"
    assert x3 == 5, f"ADD failed: expected x3=5, got {x3}"
    assert final_word == 5, f"Store/Load failed: DMEM[0]=5 expected, got {final_word}"
    assert x4 == 5, f"Load failed: expected x4=5, got {x4}"
    assert x5 == 0, f"Branch not taken? x5 should remain 0 (skipped), got {x5}"
    assert pc_final == 7*4, f"Unexpected final PC: expected 28 (EBREAK), got {pc_final}"
    
    print("All assertions passed.")

def run_all_demos():
    simulate_demo()

    # --- Additional simulations ---
    def run_program(prog_asm, *, verbose=False, max_cycles=5000):
        if Simulator is None:
            raise RuntimeError("Simulator class not found. Import your SpireHDL simulator and retry.")

        prog = assemble(prog_asm)
        m = build_rv32i_simple("RV32I_Simple")
        sim = Simulator(m)

        dmem = bytearray(1024)  # data memory

        # Reset
        sim.reset(True).eval()
        sim.deassert_reset().eval()

        def imem_fetch(pc_val: int) -> int:
            idx = (pc_val >> 2)
            return prog[idx] if 0 <= idx < len(prog) else 0

        if verbose:
            print("Assembled program:")
            for i, w in enumerate(prog):
                asm = prog_asm[i] if i < len(prog_asm) else ""
                print(f"  {4*i:04x}: 0x{w:08x}   | {asm}")

        for cycle in range(max_cycles):
            pc_val = sim.get("pc")
            inst = imem_fetch(pc_val)
            sim.set("instr", inst)
            sim.eval()

            addr = sim.get("dmem_addr") & ~0x3
            rword = int.from_bytes(dmem[addr:addr+4], "little")
            sim.set("dmem_rdata", rword)
            sim.eval()

            we   = sim.get("dmem_we")
            wdat = sim.get("dmem_wdata")
            halt = sim.get("halt")
            if we:
                dmem[addr:addr+4] = (wdat & 0xFFFFFFFF).to_bytes(4, "little")
            sim.step(1)

            if verbose:
                print(f"t={cycle:04d} PC=0x{pc_val:08x} instr=0x{inst:08x} we={we} addr={addr} w=0x{wdat&0xffffffff:08x}")
            if halt:
                break

        return dmem, sim

    def simulate_fibonacci(n: int = 10):
        print(f"\nSimulating Fibonacci sequence for n={n}")
        # First n Fibonacci numbers starting at F0=0, F1=1 stored to DMEM words
        prog_asm = [
            # init: a=x1=0, b=x2=1, ptr=x7=0, cnt=x3=n
            "ADDI x1, x0, 0",
            "ADDI x2, x0, 1",
            "ADDI x7, x0, 0",
            f"ADDI x3, x0, {n}",
            "loop: BEQ  x3, x0, done",   # if cnt==0 -> done
            "SW   x1, 0(x7)",
            "ADDI x3, x3, -1",
            "ADD  x4, x1, x2",          # t = a+b
            "ADD  x1, x2, x0",          # a = b
            "ADD  x2, x4, x0",          # b = t
            "ADDI x7, x7, 4",           # ptr += 4
            "BEQ  x0, x0, loop",
            "done: EBREAK",
        ]

        dmem, sim = run_program(prog_asm, verbose=False, max_cycles=2000)
        # Validate
        fib = [0, 1]
        for _ in range(max(0, n-2)):
            fib.append(fib[-1] + fib[-2])
        exp = fib[:n]
        got = [int.from_bytes(dmem[4*i:4*i+4], "little") for i in range(n)]
        assert got == exp, f"Fibonacci mismatch: expected {exp}, got {got}"
        print("Fibonacci OK:", got)

    def simulate_pi_digits(m: int = 10):
        print(f"\nSimulating Pi digits (22/7) for m={m}")
        # Approximate pi using 22/7. Store integer part at DMEM[0], then m digits at DMEM[4..]
        prog_asm = [
            # r=x1=22, d=x2=7, q0=x3=0, ptr=x7=0, m=x9=m
            "ADDI x1, x0, 22",
            "ADDI x2, x0, 7",
            "ADDI x3, x0, 0",
            "ADDI x7, x0, 0",
            f"ADDI x9, x0, {m}",
            # q0 loop: while r >= d: r-=d; q0++
            "q0_loop: SLTU x8, x1, x2",   # x8 = (r<d)
            "BNE  x8, x0, q0_done",
            "SUB  x1, x1, x2",
            "ADDI x3, x3, 1",
            "BEQ  x0, x0, q0_loop",
            "q0_done: SW   x3, 0(x7)",    # store integer part
            "ADDI x7, x7, 4",
            # digits loop
            "digits_check: BEQ  x9, x0, done",   # if m==0 -> done
            # t = r * 10 via repeated addition
            "ADDI x4, x0, 0",             # t=0
            "ADDI x5, x0, 10",            # c=10
            "mul10: BEQ  x5, x0, after_mul10",
            "ADD  x4, x4, x1",            # t += r
            "ADDI x5, x5, -1",
            "BEQ  x0, x0, mul10",
            "after_mul10: ADDI x6, x0, 0",# q=0
            # divide t by d: while t >= d: t -= d; q++
            "div_loop: SLTU x8, x4, x2",  # x8=(t<d)
            "BNE  x8, x0, div_done",
            "SUB  x4, x4, x2",
            "ADDI x6, x6, 1",
            "BEQ  x0, x0, div_loop",
            "div_done: ADD  x1, x4, x0",  # r = t
            "SW   x6, 0(x7)",             # store digit
            "ADDI x7, x7, 4",
            "ADDI x9, x9, -1",            # m--
            "BEQ  x0, x0, digits_check",
            "done: EBREAK",
        ]

        dmem, sim = run_program(prog_asm, verbose=False, max_cycles=20000)

        # Expected digits for 22/7
        n = 22
        d = 7
        q0 = n // d
        r = n % d
        digits = []
        for _ in range(m):
            r *= 10
            digits.append(r // d)
            r = r % d

        got_q0 = int.from_bytes(dmem[0:4], "little")
        got_digits = [int.from_bytes(dmem[4*i+4:4*i+8], "little") for i in range(m)]
        assert got_q0 == q0, f"pi int-part mismatch: expected {q0}, got {got_q0}"
        assert got_digits == digits, f"pi digits mismatch: expected {digits}, got {got_digits}"
        digits_str = "".join(str(d) for d in digits)
        print(f"Pi(22/7) OK: {q0}.{digits_str}")

    # Run the extra demos with assertions
    simulate_fibonacci(10)
    simulate_pi_digits(5)
    
if __name__ == "__main__":
    run_all_demos()
