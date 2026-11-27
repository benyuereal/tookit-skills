#!/usr/bin/env python3
r"""PMC 采集的**负载提供者**：造一份生产口径的输入、把指定实现跑 3 次，让 hipprof 采到 L1/L2/发射计数器。

分工（三层，别混）：
  - `hipprof --pmc --pmc-type 3`：真正采计数器的那一层，产出 CSV；
  - `op-profiling/SKILL.md`：口径的**唯一来源**（稳态行怎么选、TA_* 怎么跨 SE 求和、CU 数从哪来、
    哪些计数器不存在）；解析照抄该文档 §六 的参考实现，只出机械事实、不下结论；
  - 本脚本：只解决「本算子的输入长什么样、怎么把变体跑起来」。不解析、不计时、不判对错，**输出的 CSV 归上两层读**。

用法：
  HIP_VISIBLE_DEVICES=<idle> hipprof --pmc --pmc-type 3 -o /tmp/pmcout/g_tl \
      python3 bench/pmc_collect.py tl 32 128 2048 131072 0      # tilelang（第二参数=block_m 改写）
  HIP_VISIBLE_DEVICES=<idle> hipprof --pmc --pmc-type 3 -o /tmp/pmcout/g_fl \
      python3 bench/pmc_collect.py fl 0 128 2048 327680 2       # flash_mla 基线
  HIP_VISIBLE_DEVICES=<idle> hipprof --pmc --pmc-type 3 -o /tmp/pmcout/g_hip \
      python3 bench/pmc_collect.py hip dev/libsparse_mla_hip_glm.so 128 2048 327680 2
  args: variant(tl|fl|hip) [block_m | .so路径] S topk SKV pad(2=512 吸收式【生产】/1=576 补零尾/0=576 真尾)
  汇总：按 op-profiling/SKILL.md §六 的参考实现写个小脚本，读 /tmp/pmcout/*.csv 并排

手写 HIP 交付物是裸 `.so`（非 torch op、非 tilelang 模块），只能 ctypes 直调 `hip_sparse_mla_fwd`，
所以 `hip` 分支把 `.so` 路径放在 block_m 的位置；只有 512 吸收式一档（GLM5.3）。

口径：真随机索引池（生产口径，不是 `_common.build_inputs` 的增长窗口），H=64，三个实现输入逐位相同，
      可同表对照。每个实现连跑 3 次：**PMC 下首次 launch 的 GRBM 虚高 ~40%**（实测 1.83M vs 稳态 1.29M），
      取稳态行的规则只在 `op-profiling/SKILL.md §六` 里，别在这里重写；
      判据（MFMA 发射率 / (VALU−MFMA)/MFMA / LDS/MFMA）见 SKILL.md §四。
"""
import os, sys, ctypes, torch
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # skill 根
TL_ROOT = os.environ.get("TILELANG_HYGON_ROOT", "/workspace/tilelang-hygon")
for p in (HERE, os.path.join(HERE, "_dcu-toolkit-shared"), TL_ROOT):
    if os.path.isdir(p):
        sys.path.insert(0, p)
import importlib.util, tempfile
from pathlib import Path

SNAP = os.path.join(TL_ROOT, "perf/sparse_mla/sparse_mla_fwd.py")
# 2026-10-27: 生产文件改成了 `>= 64`（见 ledger §二 16 的落地记录），这里同步换锚点，
# 否则 bm 变体全部 assert 失败。bm=0 表示按生产原样跑。
OLD = "max_block_m = 32 if head_kv >= 64 else 16"
NEW = "max_block_m = %d if head_kv >= %d else 16"

def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def variant(bm):
    src = Path(SNAP).read_text()
    if str(bm) == "maxn32":
        sys.path.insert(0, "/tmp/alt")
        from xform import maxn
        src = maxn(src, 32)
    elif bm:
        assert OLD in src
        src = src.replace(OLD, NEW % (bm, bm))
    d = Path(tempfile.mkdtemp(prefix="pmc_"))
    p = d / "fwd.py"; p.write_text(src)
    return load(p, "pmc_%s" % (bm or "o"))

if len(sys.argv) < 7 or sys.argv[1] in ("-h", "--help"):
    raise SystemExit(__doc__)

name, bm, S, topk, SKV, pad = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
if name not in ("tl", "fl", "hip"):
    raise SystemExit("未知变体 %r（可选 tl / fl / hip）" % name)
H = 64
torch.manual_seed(0)
# pad=2：GLM5.3 吸收式 MLA 的生产形状（dim=512, tail_dim=0）。
# 2026-10-27 补：原来的 0/1 两档都是 576 维带 rope 尾的老形状，生产已经不走那条路。
D = 512 if pad == "2" else 576
q = torch.zeros((S, H, D), dtype=torch.bfloat16, device="cuda")
kv = torch.zeros((SKV, 1, D), dtype=torch.bfloat16, device="cuda")
q[..., :512] = torch.randn((S, H, 512), dtype=torch.bfloat16, device="cuda")
kv[..., :512] = torch.randn((SKV, 1, 512), dtype=torch.bfloat16, device="cuda")
if pad == "0":
    q[..., 512:] = torch.randn((S, H, 64), dtype=torch.bfloat16, device="cuda")
    kv[..., 512:] = torch.randn((SKV, 1, 64), dtype=torch.bfloat16, device="cuda")
idx = torch.empty((S, 1, topk), dtype=torch.int32, device="cuda")
for t in range(S):
    idx[t, 0] = torch.randperm(SKV, device="cuda")[:topk]
sc = 1.0 / (576 ** 0.5) if pad == "0" else 1.0 / (512 ** 0.5)
if name == "fl":
    import flash_mla
    for _ in range(3):
        flash_mla.flash_mla_sparse_fwd(q, kv, idx, sc, d_v=512)
elif name == "hip":
    assert pad == "2", "手写 HIP 交付物只有 512 吸收式档（GLM5.3，tail_dim=0）"
    out = torch.empty((S, H, D), dtype=torch.bfloat16, device="cuda")
    fn = ctypes.CDLL(bm).hip_sparse_mla_fwd          # bm 位置放 .so 路径（该分支没有 block_m）
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for _ in range(3):
        fn(q.data_ptr(), kv.data_ptr(), idx.data_ptr(), out.data_ptr(), S, SKV, ctypes.c_float(sc), stream)
else:
    mod = variant(None if bm == "0" else int(bm))
    for _ in range(3):
        mod.tilelang_sparse_fwd(q, kv, idx, sc, d_v=512)
torch.cuda.synchronize()
