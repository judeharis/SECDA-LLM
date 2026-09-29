#!/usr/bin/env python3
"""Prepare an RTL replay of one bfpp_acc MUL_MAT call (see README.md).

  prepare.py <trace.txt> <HLS solution dir> <out dir>

- splits a SECDA_DMA_TRACE log into per-phase stimulus files (p<phase>_c<dma>.hex)
  and the expected outputs (expected.hex);
- copies the packaged IP's VHDL core with every signal that has no initial value
  given one of 0, which is how the FPGA powers up (without it, HLS registers
  such as weightloader_A start as 'U' in simulation and the design misreads
  its input stream);
- writes bfp_acc_top_sim.vhd: the IP top with its 1-bit core ports declared
  std_logic (Vivado synthesis accepts the std_logic_vector(0 downto 0)
  mismatch; xsim does not).
"""
import glob, os, re, sys

trace, sol, out = sys.argv[1:4]
hdl = os.path.join(sol, "impl", "ip", "hdl", "verilog")
os.makedirs(os.path.join(out, "core_sim"), exist_ok=True)

phases, outs = [], []
for line in open(trace):
    t = line.split()
    if t[0] == "SEND":
        if not phases or phases[-1]["t"] != int(t[3]):
            phases.append({"t": int(t[3]), "ch": {}})
        phases[-1]["ch"][int(t[1])] = []
    elif t[0] == "W":
        ch = int(t[1])
        for p in reversed(phases):
            if ch in p["ch"]:
                p["ch"][ch].append(int(t[2]) & 0xFFFFFFFF)
                break
    elif t[0] == "R":
        outs.append((int(t[2]) & 0xFFFFFFFF, int(t[3])))
for i, p in enumerate(phases):
    for ch, ws in p["ch"].items():
        with open(os.path.join(out, f"p{i}_c{ch}.hex"), "w") as f:
            f.write("\n".join("%08x" % w for w in ws) + "\n")
        print(f"phase {i}: dma {ch}, {len(ws)} words")
with open(os.path.join(out, "expected.hex"), "w") as f:
    f.write("\n".join("%08x %d" % o for o in outs) + "\n")
print(f"{len(outs)} expected outputs")

n = 0
for path in glob.glob(os.path.join(hdl, "*.vhd")):
    name = os.path.basename(path)
    if name in ("bfp_acc_top.vhd", "BFP_Acc_reset_if.vhd"):
        continue
    text = open(path).read()
    def init(m):
        global n
        decl = m.group(0)
        if ":=" in decl:
            return decl
        n += 1
        if re.search(r"STD_LOGIC_VECTOR", decl, re.I):
            return decl[:-1] + " := (others => '0');"
        return decl[:-1] + " := '0';"
    text = re.sub(r"(?im)^\s*signal\s+\w+\s*:\s*STD_LOGIC(?:_VECTOR\s*\([^)]*\))?\s*;", init, text)
    open(os.path.join(out, "core_sim", name), "w").write(text)
print(f"{n} signals given a power-up value of 0")

top = open(os.path.join(hdl, "bfp_acc_top.vhd")).read()
ci = top.index("component BFP_Acc is"); ce = top.index("end component;", ci)
comp = top[ci:ce]
names = re.findall(r"(\w+)\s*:\s*(?:in|out)\s+std_logic_vector\(1 - 1 downto 0\)", comp)
top = top[:ci] + re.sub(r"(:\s*(?:in|out)\s+)std_logic_vector\(1 - 1 downto 0\)", r"\1std_logic", comp) + top[ce:]
m = re.search(r"BFP_Acc_U\s*:\s*component BFP_Acc\s+port map\s*\(", top)
pi = m.end(); pe = top.index(");", pi)
pm = top[pi:pe]
for nm in names:
    pm = re.sub(r"(\b%s\s*=>\s*)(\w+)" % nm, r"\1\2(0)", pm)
open(os.path.join(out, "bfp_acc_top_sim.vhd"), "w").write(top[:pi] + pm + top[pe:])
