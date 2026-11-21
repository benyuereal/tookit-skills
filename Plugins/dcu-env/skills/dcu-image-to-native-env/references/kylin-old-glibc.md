# Kylin V10 / 老 glibc 系统移植（2026-09-03 在 <<internal-ip>> 端到端验证通过）

> 把 ubuntu22.04 镜像（glibc 2.35）复刻到 **Kylin V10（内核 4.19，系统 glibc 2.28）** 物理机。
> 与 Ubuntu 22.04 主机（见 SKILL.md 主流程）相比，**核心新增问题是 glibc 版本不匹配**。
> ✅ 状态：最终方案已端到端验证——sglang 服务起、Qwen3.6-27B 模型推理正确（123×456=56088）。

## 0. 先做 glibc 预检（高效，别先拷 28GB）

镜像 .so 若编译针对 glibc 2.35，拷到 2.28 系统会加载失败。**先扫描，再决定方案**：

```bash
# 系统 libc 导出的最高 GLIBC 符号版本（Kylin 2.28 只到 GLIBC_2.28）
objdump -T /lib64/libc.so.6 | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sort -Vu | tail -1

# 镜像内全量扫 .so，找需求 > 系统版本的（在探针容器里跑）
cd /usr/local/lib/python3.10/dist-packages
for so in $(find . -name "*.so" -o -name "*.so.*"); do
  max=$(objdump -T "$so" 2>/dev/null | grep -oE "GLIBC_[0-9]+\.[0-9]+" | sort -Vu | tail -1)
  ver=$(echo "$max" | grep -oE "[0-9]+\.[0-9]+$")
  [ -z "$ver" ] && continue
  major=${ver%.*}; minor=${ver#*.}
  if [ "$major" -gt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -gt 28 ]; }; then
    echo "NEEDS_$ver  $so"
  fi
done
```

实测（sglang0.5.12 镜像，850 个 .so）只有 **7 个**需求 > 2.28：

| .so | 需求 | 是否在 sglang 核心推理路径 |
|---|---|---|
| `sgl_kernel/common_ops...so` | GLIBC_2.32（`__libc_single_threaded`） | **是，核心算子库，关键** |
| `sglang/srt/grpc/_core...so` | 2.34 | 否（仅 grpc_mode） |
| `sglang_router/...rs.abi3.so` | 2.34 | 否（仅多机路由） |
| `cryptography/..._rust.abi3.so` | 2.34 | 否（sglang 不 import cryptography） |
| `st_attn_cuda...so` / `vsa_cuda...so` | 2.32 | 否（稀疏注意力，可选） |
| `runai_model_streamer/...so` | 2.29 | 否（模型流式加载，可选） |

**结论**：真正卡脖子的只有 `sgl_kernel/common_ops.so`（GLIBC_2.32）。其余 6 个不在核心路径。

## 1. 已实测失败的方案（别再试）

| 方案 | 结果 | 原因 |
|---|---|---|
| LD_PRELOAD shim 提供 `__libc_single_threaded` 符号 | **无效** | 报错是 `version 'GLIBC_2.32' not found`（**版本节点 VERNEED** 缺失），不是 `undefined symbol`。符号预加载解决不了版本节点问题。（早期"shim 有效"是假阳性：libgflags 错误发生在依赖解析阶段，glibc 版本检查还没执行） |
| `conda install glibc`（conda-forge） | **不可用** | conda-forge linux-64 无 glibc 包；conda python 仍链接系统 libc 2.28 |
| patchelf 改 **conda** python 的 PT_INTERP → glibc 2.35 的 ld-linux | **间歇性 segfault** | conda python 是 2.28 编译的（最高符号需求 GLIBC_2.28），用 2.35 的 ld 加载：`import sgl_kernel`/`torch`/matmul 全 OK，但 `import sglang`/`launch_server` 间歇 segfault（不可靠） |
| LD_PRELOAD glibc 2.35 的 libc.so.6 | **segfault**（exit 139） | 同上，2.28 编译的 python 用 2.35 libc 在复杂 import 时崩 |

## 2. 最终方案（端到端验证通过）

**核心思路：镜像 python（2.35 编译）+ patchelf 自包含 + libgalaxyhip 统一用原生 DTK。**

### 2.1 python 自包含（patchelf）

镜像 python3.10 是 2.35 编译、静态链接 libpython（5.7M 自包含），与所有 .so 同 glibc 基线——这是根本解法（conda python 是 2.28 编译的，怎么换 ld 都 segfault）。

```bash
# 1. 提取镜像 python + stdlib + 头文件
docker cp img:/usr/bin/python3.10 <dcu-env>/imgpython/python3.10
docker exec img tar cf - -C /usr/lib python3.10 | tar xf - -C <dcu-env>/imgpython/lib/
# 头文件（sglang JIT 编译 hip_utils 需要 Python.h）
docker exec img tar cf - -C /usr/include python3.10 | tar xf - -C <dcu-env>/imgpython/include/
# ⚠ 坑：tar 解包 /usr/include/python3.10 会覆盖 imgpython/python3.10（二进制）！
#   头文件要解到 include/python3.10/，二进制单独放 imgpython/python3.10

# 2. 拷镜像完整系统库（glibc 2.35，1.1G）
docker exec img tar cf - -C /lib x86_64-linux-gnu | tar xf - -C <dcu-env>/imglib/

# 3. patchelf（用 conda 自带的 patchelf，dnf 源没有）
PE=<conda>/bin/patchelf
IMG=<dcu-env>/imglib/x86_64-linux-gnu
PY=<dcu-env>/imgpython/python3.10
NDTK=/opt/dtk-26.04   # 原生 DTK（2.28 编译）
EXTRA=<dcu-env>/extra-libs
$PE --set-interpreter $IMG/ld-linux-x86-64.so.2 $PY
$PE --set-rpath "$IMG:$NDTK/hip/lib:$NDTK/lib:$NDTK/lib64:$NDTK/llvm/lib:$NDTK/dcc/gcvm/lib:$NDTK/dushmem/lib:$NDTK/opencl/lib:/opt/hyhal/lib:/opt/hyhal/lib64:$EXTRA" --force-rpath $PY
```

**关键**：RPATH 指向**原生 DTK**（不是镜像 DTK），为了统一 libgalaxyhip 实例（见 2.2）。

**RPATH vs LD_LIBRARY_PATH 分工（ELF 加载语义，关键，别只设一个）**：
- **RPATH** 只作用于**主可执行文件（python）的直接依赖**，**不**作用于 .so 的传递依赖。所以 python 直接依赖的 glibc 2.35（libc/libm）走 RPATH 就能解析。
- **LD_LIBRARY_PATH** 作用于**进程内所有库**（含 .so 的传递依赖），且**传播给子进程**（sglang spawn 的 TP worker 继承）。所以 torch 的 `libgfortran.so.5`（`libtorch_hip.so` 的传递依赖，python RPATH 覆盖不到）必须靠 LD_LIBRARY_PATH。
- 这就是 wrapper 要**两者都设**的原因：RPATH 给 python 自包含 glibc 2.35，LD_LIBRARY_PATH 给 torch 的传递依赖 + 传给 worker。
- 只设 RPATH 不设 LD_LIBRARY_PATH → `ImportError: libgfortran.so.5: cannot open shared object file`（实测踩过，见 2.3）。

### 2.2 libgalaxyhip 实例统一（关键，最隐蔽的坑）

**现象**：torch 能 matmul，但 sglang 起服务时 Triton kernel 报
`ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)`。

**根因**：torch 和 Triton launcher 加载了**两份不同的 libgalaxyhip**（不同 DTK 目录、不同字节数），跨实例的 HIP 指针互不认——torch 用 A 实例分配的 GPU 指针，Triton launcher 用 B 实例的 `hipPointerGetAttribute` 查询，返回 `hipErrorInvalidValue`。

- torch 加载哪份：走 python **RPATH**（patchelf 设的）
- Triton launcher 链接哪份：走 `HIP_PATH`/`LD_LIBRARY_PATH`（driver.py 的 `_get_path_to_hip_runtime_dylib`）
- 两者不一致 → 双实例

**修法**：torch + Triton launcher + rocminfo **全用原生 DTK**（/opt/dtk-26.04，2.28 编译）。
原理：原生 DTK 的 libgalaxyhip 是 2.28 编译，既能被 image ld 2.35 加载（torch），也能被系统 ld 2.28 加载（rocminfo）——一份库通吃。镜像 DTK 只用来补 aillvm（Triton 编译用）。

wrapper 里四个变量分工：
```bash
export LD_LIBRARY_PATH="$NDTK/hip/lib:$NDTK/lib:...:$EXTRA:/opt/hyhal/lib:..."  # 原生 DTK 优先
export DTKROOT=$IDTK ROCM_PATH=$IDTK   # 镜像 DTK：Triton 找 aillvm clang
export HIP_PATH=$NDTK                   # 原生 DTK：Triton launcher 链接 libgalaxyhip
export PATH="$NDTK/bin:...:$IDTK/bin:..."  # 原生 DTK 优先：rocminfo 用原生版
```

**验证**：`python -c "import torch; x=torch.randn(8,device='cuda:0'); print([l.split()[-1] for l in open('/proc/self/maps') if 'libgalaxyhip' in l][0])"`——必须输出**原生 DTK** 路径（/opt/dtk-26.04/...），不是镜像 DTK。

### 2.3 libgfortran/libquadmath（extra-libs）

torch 需要 `libgfortran.so.5`（系统没有），镜像的是 2.35 编译。建 extra-libs 软链：
```bash
mkdir -p <dcu-env>/extra-libs
for lib in libgfortran.so.5 libgfortran.so.5.0.0 libquadmath.so.0 libquadmath.so.0.0.0; do
  ln -sf <dcu-env>/imglib/x86_64-linux-gnu/$lib <dcu-env>/extra-libs/$lib
done
```
加进 RPATH + LD_LIBRARY_PATH。

**⚠ 关键**：extra-libs **只放 torch 专用库**（libgfortran/libquadmath），**绝不能放 glibc 核心**（libc/libm/libpthread/libtinfo/libncurses）——否则系统工具（cat/sh/rocminfo）继承 LD_LIBRARY_PATH 后加载 2.35 版会崩（`relocation error: symbol __errno_location version GLIBC_2.2.5 not defined in file libpthread.so.0`）。

### 2.4 pyconfig.h（multiarch wrapper 坑）

镜像 `/usr/include/python3.10/pyconfig.h` 是 **4K multiarch wrapper**（`#include <x86_64-linux-gnu/python3.10/pyconfig.h>`），真正的 48K 版在 multiarch 路径。只拷 wrapper → sglang JIT 编译 hip_utils 时 gcc 报 `x86_64-linux-gnu/python3.10/pyconfig.h: 没有那个文件或目录`。

修法：替换为 48K 真版：
```bash
docker cp img:/usr/include/x86_64-linux-gnu/python3.10/pyconfig.h <dcu-env>/imgpython/include/python3.10/pyconfig.h
```

### 2.5 wrapper（dcu_python）

```bash
#!/bin/bash
# DCU python wrapper — Kylin V10 最终方案
# python glibc 2.35 自洽: patchelf INTERP+RPATH(imglib)
# libgalaxyhip 统一: torch + Triton launcher + rocminfo 全用原生 DTK(2.28)
NDTK=/opt/dtk-26.04
IDTK=/data1/dtk
EXTRA=/data1/dcu-env/extra-libs
SP=/data1/miniconda3/envs/sglang0512/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$NDTK/hip/lib:$NDTK/lib:$NDTK/lib64:$NDTK/llvm/lib:$NDTK/dcc/gcvm/lib:$NDTK/dushmem/lib:$NDTK/opencl/lib:$EXTRA:/opt/hyhal/lib:/opt/hyhal/lib64:/data1/dcu-env/opt/ucx/lib:/data1/dcu-env/opt/mpi/lib:/data1/dcu-env/opt/hwloc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$SP${PYTHONPATH:+:$PYTHONPATH}"
export DTKROOT=$IDTK ROCM_PATH=$IDTK HIP_PATH=$NDTK HYHAL_PATH=/opt/hyhal DTK_VERSION=2604
export AMDGPU_TARGETS="gfx906;gfx926;gfx928;gfx936;gfx938" MIOPEN_FIND_MODE=3
export PATH="$NDTK/bin:$NDTK/llvm/bin:$NDTK/hip/bin:/opt/hyhal/bin:$IDTK/bin:$IDTK/llvm/bin:$IDTK/hip/bin:$PATH"
# 关键: imgpython 是 glibc 2.35 自洽（patchelf 指向 imglib 的 ld-linux），
# 绝不能继承 LD_PRELOAD 里的 glibc_shim.so（那是给 conda python/glibc2.28 补符号的）。
# 若用户先 source activate_dcu_env.sh（会 export LD_PRELOAD=glibc_shim.so）再跑 dcu_python，
# shim 被预加载进 glibc2.35 进程 → __libc_early_init 段错误（EXIT 139）。
unset LD_PRELOAD
exec /data1/dcu-env/imgpython/python3.10 "$@"
```

**关键**：LD_LIBRARY_PATH **只含原生 DTK + extra-libs**（无 imglib 的 glibc 2.35），所以系统工具（cat/sh/rocminfo）继承它不会崩。python 的 glibc 2.35 来自 patchelf INTERP+RPATH，不靠 LD_LIBRARY_PATH。

**⚠ LD_PRELOAD 继承陷阱（2026-09-03 实测，wrapper 必须带 `unset LD_PRELOAD`）**：`activate_dcu_env.sh`（conda python 激活脚本）里有 `export LD_PRELOAD=<dcu-env>/glibc_shim.so`——shim 是给 **conda python（glibc 2.28 编译）** 补 `__libc_single_threaded` 等符号用的。但 imgpython 是 **glibc 2.35 自洽**，不需要 shim。用户先 `source activate_dcu_env.sh` 再跑 `./dcu_python`，LD_PRELOAD 被继承 → shim 预加载进 glibc 2.35 进程 → 启动即段错误（EXIT 139，dmesg `segfault ... in libc.so.6`，backtrace `__libc_early_init`，即 main 之前的动态链接阶段）。**这就是"我跑正常、别人跑崩"的原因**：非交互 ssh（`ssh host 'cmd'`）不读 .bashrc/激活脚本，环境干净能跑；交互式登录（`ssh host` 敲命令）source 了激活脚本 → LD_PRELOAD 被设 → 崩。修法：wrapper 的 exec 前加 `unset LD_PRELOAD`（自防御）。诊断流程与 core 证据见 §5。

### 2.6 sglang 启动参数（Qwen3.6-27B，Qwen3_5 架构，TP2）

```bash
export HIP_VISIBLE_DEVICES=0,1
dcu_python -m sglang.launch_server \
  --model-path /data/models/Qwen3.6-27B \
  --tp 2 --host 127.0.0.1 --port 30999 \
  --attention-backend fa3 --page-size 64 \
  --context-length 32768 --mem-fraction-static 0.85 \
  --disable-radix-cache
```

Kylin 特有参数（Ubuntu 22.04 不需要）：
- `--disable-radix-cache`：Qwen3_5 是混合 Mamba/线性注意力架构，radix cache + Mamba no_buffer 模式**强制 page_size=1**（日志：`requires page_size=1 ... but got 64. Automatically setting page_size=1`），与 fa3 要求 paged KV block=64 矛盾 → `RuntimeError: Paged KV cache block size must be 64`。禁 radix cache 让 page_size 保持 64。
- Qwen3.6 `text_config.rope_scaling=null`，**不传** `--json-model-override-args`（那是 Qwen3.8 的 yarn 参数）。
- **CUDA graph 不用禁（2026-09-03 复测纠正）**：标准 decode CUDA graph 在 v2 wrapper 下**正常跑**——默认 max_bs=256、capture 27 个 batch size（bs [1,2,4,8,12,16,24,...,179]）无 static TLS 报错、推理正确（56088）、decode 日志 `cuda graph: True`、吞吐 **22 tok/s**（关 CUDA graph 时 12.9，提速 70%）。之前"static TLS 耗尽必须 --disable-cuda-graph"是误判。
- **piecewise CUDA graph 已与镜像一致（2026-09-03 对照镜像容器确认，无需额外操作）**：DCU 是 HIP 平台，sglang `server_args._handle_piecewise_cuda_graph()` 里 `if is_hip(): self.disable_piecewise_cuda_graph = True` **自动禁用** piecewise。对照目标镜像容器（`lcq-qwen3.8`，镜像正是 `...custom:sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620`）实际 server_args：`disable_cuda_graph=False`（CUDA graph 开）+ `disable_piecewise_cuda_graph=True`（piecewise 关）+ `enforce_piecewise_cuda_graph=False`——**与 <device-B> 原生完全一致**。容器命令里那个 `--enable-piecewise-cuda-graph` 是**已废弃 flag**（启动日志明确提示 deprecated），最终 `enforce=False` 被 `is_hip()` 覆盖，**根本没生效**。**不要**加 `--enforce-piecewise-cuda-graph` 强行开 piecewise——那反而与镜像不一致（镜像里 piecewise 也是关的）。

### 2.7 验证结果（2026-09-03 实测）

- torch 2.9.0，8x BW100，matmul OK
- sgl_kernel / sglang / lightop / triton 3.5.1 import OK
- rocminfo CU=64（子进程正常，不崩）
- 系统工具（cat/sh/head）未污染
- sglang 服务起：`The server is fired up and ready to roll!`
- 模型推理：123×456=**56088**（finish:stop，正确）、五言绝句、流式 131 chunk、并发 4/4
- **CUDA graph（2026-09-03 复测）**：标准 decode CUDA graph 默认 max_bs=256、capture 27 个 batch size 成功、无 static TLS 报错、decode 日志 `cuda graph: True`、吞吐 **22 tok/s**（关 CUDA graph 时 12.9，提速 70%）。**piecewise CUDA graph 已与镜像一致**（DCU 是 HIP 平台，sglang `is_hip()` 自动禁用 piecewise；对照目标镜像容器 server_args 也是 `disable_piecewise_cuda_graph=True`，无需额外操作，详见 2.6）

## 3. Kylin 特有问题清单（最终版）

1. **glibc 2.28 vs 2.35 不匹配**（7 个 .so，核心 sgl_kernel/common_ops.so）→ 镜像 python + patchelf INTERP
2. **原生 DTK 缺 aillvm**（只有 llvm/clang-17）→ 拷镜像完整 DTK（17G）+ 软链目录名 dtk
3. **conda python 2.28 编译**，patchelf/LD_PRELOAD 换 glibc 都 segfault → 用镜像 python
4. **镜像 python 缺 libffi.so.8**（Kylin 只有 libffi.so.6）→ 从镜像拷（libffi.so.8 是指向 libffi.so.8.1.0 的符号链接，要拷真文件）
5. **libgalaxyhip 双实例**（torch 用镜像 DTK、Triton launcher 用原生 DTK，指针互不认 → "Pointer argument cannot be accessed from Triton"）→ 统一用原生 DTK
6. **static TLS 耗尽**（老 glibc 2.28 + 老内核 4.19，`libgomp.so.1 cannot allocate memory in static TLS block`）→ **2026-09-03 复测重新归因**：标准 decode CUDA graph（默认 max_bs=256）在 v2 wrapper 下**不触发**，之前失败是旧 wrapper 双实例 / piecewise 层导致。piecewise 层在 DCU（HIP）上被 sglang `is_hip()` 自动禁用，与镜像一致，无需处理。若复现：先确认 wrapper v2；`--disable-cuda-graph` 是最后手段（decode 吞吐掉 ~40%）
7. **pyconfig.h multiarch wrapper**（镜像 pyconfig.h 是 4K wrapper，#include <x86_64-linux-gnu/...> 找不到 → gcc 编译 hip_utils 失败）→ 替换 48K 真版
8. **Qwen3_5 Mamba page_size 冲突**（radix cache + Mamba no_buffer 强制 page_size=1，与 fa3 要求 64 矛盾）→ --disable-radix-cache
9. **docker cp 的 `/.` 合并语法静默失败**（某些 docker 版本 exit 0 但只拷部分）→ 先拷全新临时目录再合并
10. **老内核 4.19**（Kylin V10）——命令行为与 Ubuntu 5.15 差异大

## 4. <device-B> 已就位资产（最终版）

- `/data1/miniconda3/envs/sglang0512`（conda py3.10 + 镜像 599 包 11G）
- `/data1/dtk-26.04-full`（镜像完整 DTK 17G，含 aillvm）+ `/data1/dtk` 软链
- `/data1/dcu-env/imglib`（镜像完整系统库 1.1G，glibc 2.35）
- `/data1/dcu-env/imgpython`（镜像 python3.10 + stdlib + include，已 patchelf）
- `/data1/dcu-env/extra-libs`（libgfortran/libquadmath 软链）
- `/data1/dcu-env/dcu_python`（wrapper，最终方案）
- `/data1/dcu-env/opt/{ucx,hwloc}`
- 启动脚本 `/tmp/launch_sglang_242.sh`（Qwen3.6-27B TP2 fa3）

## 5. dcu_python 段错误诊断（LD_PRELOAD 继承陷阱，2026-09-03 实测）

**现象**：别人跑 `./dcu_python` 启动即段错误（EXIT 139），但自己 ssh 跑完全正常。dmesg 连续多条 `python3.10 ... segfault ... error 7 in libc.so.6`。

**为什么"我跑正常、别人跑崩"**：
- 非交互 ssh（`ssh host 'cmd'`）是**非交互 shell**，不读 `.bashrc`/激活脚本 → 环境干净 → 能跑
- 交互式登录（`ssh host` 敲命令）会 source `.bashrc`/激活脚本 → 若其中 `source activate_dcu_env.sh`（export 了 `LD_PRELOAD=glibc_shim.so`）→ 再跑 `./dcu_python` 就崩

**根因**：`glibc_shim.so` 是给 **conda python（glibc 2.28 编译）** 补 `__libc_single_threaded` 等符号的；imgpython 是 **glibc 2.35 自洽**（patchelf 指向 imglib 的 ld-linux），不需要 shim。shim 被 LD_PRELOAD 预加载进 glibc 2.35 进程后，在 `__libc_early_init`（main 之前的动态链接阶段）写未映射页 → 段错误。

**诊断流程（可复用，按序做）**：
```bash
# 1. dmesg 找段错误记录（确认是 python3.10、崩在 libc.so.6）
dmesg -T | grep -iE "segfault|core dump" | tail -20
# 2. coredumpctl 看 backtrace（确认崩在 __libc_early_init = 动态链接最早期）
coredumpctl list | tail -15
coredumpctl info <PID>          # 看 Stack trace + Command Line
# 3. 提取 core（coredumpctl extract/dump 不可用时直接 lz4 解压）
CORE=$(ls -t /var/lib/systemd/coredump/core.python3*<PID>* | head -1)
lz4 -d -f "$CORE" /tmp/core_<PID>
# 4. gdb 看实际加载了哪些库（关键：确认 glibc_shim.so 被加载了）
gdb -batch -ex "info proc mappings" <imgpython> /tmp/core_<PID> | grep -E "libc|ld-linux|shim"
gdb -batch -ex "bt 20" <imgpython> /tmp/core_<PID>
# 5. 复现（正确抓退出码，别用管道——管道会掩盖真实 EXIT）
LD_PRELOAD=<dcu-env>/glibc_shim.so ./dcu_python -c "print(1)" > /tmp/o.txt 2>&1; echo "EXIT=$?"   # 期望 139
./dcu_python -c "print(1)" > /tmp/o.txt 2>&1; echo "EXIT=$?"                                       # 期望 0
```

**修复**：`dcu_python` 和 `pyimg` 两个 wrapper 的 `exec` 前各加一行 `unset LD_PRELOAD`（自防御，无论用户之前 source 过什么都不会串）。改前备份 `cp -a dcu_python dcu_python.bak_$(date +%Y%m%d_%H%M%S)`。

**验证**：带 LD_PRELOAD（模拟 source activate 后）+ 干净环境都跑通；端到端 `dcu_python -c "import torch; x=torch.randn(1024,1024,device='cuda:0'); print((x@x).sum().item())"` 正常出数。

**注意**：`activate_dcu_env.sh` 里的 `LD_PRELOAD=glibc_shim.so` 对 **conda python（路径 A）** 是需要的，不要删——只在 wrapper（路径 B 的 imgpython）侧 unset。
