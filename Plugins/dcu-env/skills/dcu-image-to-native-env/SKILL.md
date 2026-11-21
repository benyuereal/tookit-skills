---
name: dcu-image-to-native-env
description: 把 DCU 镜像复刻为物理机原生 conda 环境（非 Dockerfile 重建）。
version: 1.0.0
author: DCU Engineer
license: MIT
---

# DCU 镜像环境 → 物理机原生 conda 环境复刻

给定一个 DCU docker 镜像（如 `<internal_registry>/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620`），在物理机上构建**不依赖容器、可直接运行**的等价环境（conda env）。

**与 `dcu-image-reverse-rebuild` 的区分（重要）**：那个 skill 产出 Dockerfile 重建镜像，其用户硬约束**禁止**把包目录拷出容器；本 skill 的目标是物理机原生环境，核心做法**恰恰是**把镜像的包目录 + DTK 拷出容器。两个任务类约束相反，不要混用。

## When to Use
- 用户说"依据这个镜像环境在物理机的 anaconda 环境下构建一个相同的环境，能够在物理机下直接运行"。
- 物理机不想/不能跑容器，但需要与某 DCU 镜像完全一致的软件栈。

## 路径选择（30 秒判断，先做这个）

| 物理机系统 | glibc | 路径 | 关键差异 |
|---|---|---|---|
| Ubuntu 22.04 | 2.35（与镜像同） | **路径 A**（简单） | 直接用 conda python |
| Kylin V10 / CentOS / 老系统 | 2.28 或更低 | **路径 B**（镜像 python + patchelf + wrapper） | glibc 不匹配，必须用镜像 python |

判断命令：`objdump -T /lib64/libc.so.6 | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sort -Vu | tail -1`
- 输出 `GLIBC_2.35`（≥ 镜像需求）→ **路径 A**
- 输出 `GLIBC_2.28`（或更低）→ **路径 B**（详见 `references/kylin-old-glibc.md`）

## 完整复现流程（一步一步照做，2026-09-02/03 在 <<internal-ip>>/<<internal-ip>> 端到端验证）

> 从零到"服务起 + 模型推理正确"。按顺序做，每步有**关键命令 + 验证点**。
> 标 **[B]** 的步骤仅路径 B（Kylin/老 glibc）需要；路径 A（Ubuntu 22.04）跳过。
> 每步的详细坑见下方"坑速查"和 `references/kylin-old-glibc.md`。

### 步骤 0. 摸底（镜像 + 物理机两侧）
- 起探针容器：`docker run -d --name img_probe --network host -v /opt/hyhal:/opt/hyhal:ro <image> sleep 3600`
- 镜像侧：`pip list` 全量存文件（后面 diff 用）、`env | grep -iE 'dtk|hip|path|ld_lib'` 记录 DTK 环境变量、`du -sh` 看包/DTK 大小
- 物理机侧：**先 `df -h` 查磁盘**（环境放大盘如 /public）、`ls -d /opt/dtk*` 验证 DTK 目录完整性、`hy-smi` 看空闲卡（VRAM% 0% 为空闲）、查可用模型
- **验证**：磁盘够（DTK ~17G + 包 ~11G + imglib ~1.1G）、空闲卡够

### 步骤 1. glibc 预检 [B]
- `objdump -T /lib64/libc.so.6 | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sort -Vu | tail -1` 看系统最高符号版本
- 探针容器里全量扫镜像 .so 找需求更高的（脚本见 `references/kylin-old-glibc.md` §0）
- **验证**：定位卡脖子的 .so（通常只有 `sgl_kernel/common_ops.so` 需 GLIBC_2.32）

### 步骤 2. Miniconda + conda env
- 装到大盘：`bash Miniconda3-py310_*.sh -b -p /public/miniconda3`
- `conda create -y -n <env> -c <清华源> python=3.10`（python 版本对齐镜像）
- **验证**：`conda activate <env> && python --version` 输出 3.10.x

### 步骤 3. 拷镜像 Python 包（核心）
- `docker cp img_probe:/usr/local/lib/python3.10/dist-packages <env>/lib/python3.10/site-packages`
- **拍平嵌套目录**：`cp -an site-packages/dist-packages/. site-packages/ && rm -rf site-packages/dist-packages`
- **补 Ubuntu 系统包**（six 等在 /usr/lib/python3/dist-packages）：`docker exec img_probe tar cf - -C /usr/lib/python3/dist-packages <纯python包列表> | tar xf - -C <site-packages>/`
- **验证**：`diff <(docker exec img_probe pip list --format=freeze | sort) <(pip list --format=freeze | sort)`——剩余差异应**全是 `<` 行**（镜像多出的 Ubuntu 系统包），无 `>` 行

### 步骤 4. 拷 DTK + 配套组件
- `docker cp img_probe:/opt/dtk /public/dtk-26.04`（完整 DTK ~17G）
- `docker cp img_probe:/opt/{ucx,mpi,hwloc} /public/dcu-env/opt/`
- **验证 DTK 完整**（别信目录存在）：hip/hipblas/hipdnn/llvm/rccl 目录 + env.sh 都在
- **建软链保证目录名为 dtk**（Triton 靠目录名选 clang）：`ln -sfn /public/dtk-26.04 /public/dtk && ln -sfn /public/dtk /opt/dtk`
- **验证**：`ls /public/dtk/{hip,hipblas,hipdnn,llvm,rccl}` 全存在

### 步骤 5. Python 运行时（分叉点）
- **路径 A（Ubuntu 22.04）**：直接用 conda python（glibc 与镜像一致），跳到步骤 6
- **路径 B（Kylin/老 glibc）**：镜像 python + patchelf + wrapper（详见 `references/kylin-old-glibc.md` §2）：
  1. 提取镜像 python3.10 + stdlib + 头文件（**头文件别覆盖二进制**，见坑 20）
  2. 拷镜像完整系统库（glibc 2.35，imglib ~1.1G）
  3. patchelf：`--set-interpreter` 指向镜像 ld-linux、`--set-rpath` 指向 imglib + 原生 DTK + extra-libs
  4. **libgalaxyhip 统一用原生 DTK**（torch + Triton launcher + rocminfo 同一份，见坑 16）
  5. 写 wrapper（dcu_python）：LD_LIBRARY_PATH 只含原生 DTK + extra-libs（**绝不放 glibc 核心**）
  6. pyconfig.h 换 48K 真版（见坑 18）
- **验证**：`dcu_python -c 'import torch; print(torch.cuda.device_count())'` 输出卡数；`python -c "import torch; ..."` 确认 libgalaxyhip 加载的是**原生 DTK** 路径

### 步骤 6. 命令行入口
- `docker cp img_probe:/usr/local/bin /tmp/img_local_bin`，全部脚本拷到 `<env>/bin/`（跳过 python*/pip*/conda*）
- 修 shebang：`sed -i "1s|^#!.*python[0-9.]*|#!<env>/bin/python3|"`
- **验证**：入口脚本 shebang 指向正确 python

### 步骤 7. 激活脚本 / wrapper
- 路径 A：写 `activate_dcu_env.sh`（模板见 `templates/activate_dcu_env.sh`）：DTK 环境变量 + 完整 LD_LIBRARY_PATH/PATH + `conda activate <env>`
- 路径 B：用 wrapper `dcu_python`（见 `references/kylin-old-glibc.md` §2.5）
- **验证**：`source activate_dcu_env.sh`（或 `dcu_python`）后 `which python` / `echo $LD_LIBRARY_PATH` 正确

### 步骤 8. 验证（"能用"的定义）
- DCU 实际计算：`python -c 'import torch; x=torch.randn(1024,1024,device="cuda:0"); print((x@x).sum().item())'`
- 核心框架包逐个 import（sglang/vllm/transformers/triton）
- Triton clang 路径：`python -c 'from triton.backends.amd.compiler_hcu import HIPBackend; print(HIPBackend.path_to_rocm_clang())'`——必须是 `aillvm/bin/clang-18`（见坑 8）
- **验证**：以上全过

### 步骤 9. 启动服务 + 模型实测
- 起 sglang 服务（参数见下方"启动服务"段；路径 B 加 `--disable-radix-cache`，见坑 19）
- 等日志出现 `The server is fired up and ready to roll!`
- 接口测试：`curl /health`（200）、`curl /v1/chat/completions` 发"计算 123*456"期望 56088
- **验证**：模型返回正确结果（56088）、finish:stop

### 步骤 10. 验证"与镜像完全一致"（对照**有效** server_args，不是命令行）
- 找到跑**目标镜像**的容器（`docker ps` 看 Image 列，**别认错镜像**——同机常有多个不同镜像的容器）
- 从该容器日志抓**有效** `ServerArgs(...)` dump：`docker logs <c> 2>&1 | grep "server_args=ServerArgs"`
- 从原生环境日志抓同样的 `ServerArgs(...)` dump
- 逐字段对比（`disable_cuda_graph` / `disable_piecewise_cuda_graph` / `cuda_graph_max_bs` 等）
- **验证**：字段一致（tp 不同导致的 `cuda_graph_max_bs` 差异属正常）

## 详解（每步深入，2026-09-02 在 <<internal-ip>> 实测，sglang0.5.12-dtk26.04-py3.10 镜像）

### 0. 摸底（镜像 + 物理机两侧）
镜像侧（`docker run -d --name img_probe --network host -v /opt/hyhal:/opt/hyhal:ro <image> sleep 3600` 后）：
- `pip list` 全量存文件（后面 diff 用）
- `env | grep -iE 'dtk|hip|path|ld_lib'` 记录 DTK 环境变量全集
- `du -sh /usr/local/lib/python3.10/dist-packages /usr/lib/python3/dist-packages /opt/dtk`
- `ls /usr/local/bin | wc -l`（命令行入口数量）
- `readelf -d /opt/dtk/lib/libamdhip64.so | grep -i runpath`（确认 DTK 库是否写死 RUNPATH=/opt/dtk）

物理机侧：
- `df -h` **先查磁盘**：/ 盘常接近满，环境放 /public 这类大数据盘（实测机：/ 剩 15G，/public 剩 222G）
- `ls -d /opt/dtk*` 并**验证目标 DTK 版本目录完整性**（见坑 4）
- `hy-smi` 看卡数/空闲卡（VRAM% 0% 为空闲）
- 查可用模型（最终实测用）

### 1. Miniconda + conda env
- 装到大盘：`bash Miniconda3-py310_*.sh -b -p /public/miniconda3`
- `conda create -y -n <env> -c <清华源> python=3.10`（python 版本对齐镜像）

### 2. 拷镜像 Python 包（核心）
- `docker cp img_probe:/usr/local/lib/python3.10/dist-packages <env>/lib/python3.10/site-packages`
  - **坑**：docker cp 会生成嵌套 `site-packages/dist-packages/` 子目录——必须拍平：`cp -an site-packages/dist-packages/. site-packages/ && rm -rf site-packages/dist-packages`
- **坑**：镜像 `pip list` 含 Ubuntu 系统包（six/pyparsing/zipp/distro/httplib2/importlib_metadata/more_itertools/oauthlib 等），它们在 `/usr/lib/python3/dist-packages`（不在 /usr/local/.../dist-packages）——只拷前者会漏 → `ModuleNotFoundError: No module named 'six'`（dateutil 依赖它）。修法：`docker exec img_probe tar cf - -C /usr/lib/python3/dist-packages <纯python包列表> | tar xf - -C <site-packages>/`；跳过 dbus-python/PyGObject/python-apt 这类依赖系统库的包。
- **坑**：`docker cp` 整个目录时遇到坏符号链接（如 `lsb_release.py -> ../../../share/pyshared/...`）会报 "invalid symlink" 中断——改用 `docker exec tar cf - | tar xf -` 管道。
- 版本验证：`diff <(docker exec img_probe pip list --format=freeze | sort) <(pip list --format=freeze | sort)`——剩余差异应**全部是 `<` 行**（镜像多出的 Ubuntu 系统包），无 `>` 行（物理机多出）。

### 3. 拷 DTK + 配套组件
- `docker cp img_probe:/opt/dtk /public/dtk-26.04`（完整 DTK ~17G）
- `docker cp img_probe:/opt/{ucx,mpi,hwloc} /public/dcu-env/opt/`
- **坑**：物理机 `/opt/dtk-26.04` 可能不完整（实测机只有 250M 的 onnxruntime，缺 hip/hipblas/hipdnn/llvm）——不要信目录存在，验证完整布局（hip/hipblas/hipdnn/llvm/rccl 目录 + env.sh）。
- **坑**：镜像 DTK 库 RUNPATH 写死 `/opt/dtk`，物理机该软链指向旧版（25.04）——激活脚本必须显式 `LD_LIBRARY_PATH`（优先级高于 RUNPATH）。

### 4. 命令行入口
- `docker cp img_probe:/usr/local/bin /tmp/img_local_bin`，全部脚本拷到 `<env>/bin/`（跳过 python*/pip*/conda* 避免覆盖 conda 自带）
- 修 shebang：`sed -i "1s|^#!.*python[0-9.]*|#!<env>/bin/python3|"`（镜像入口指向 /usr/bin/python3）

### 5. 激活脚本
- 写 `/public/dcu-env/activate_dcu_env.sh`（模板见 `templates/activate_dcu_env.sh`）：DTK 环境变量（DTKROOT/ROCM_PATH/HIP_PATH/HYHAL_PATH/AMDGPU_TARGETS/MIOPEN_FIND_MODE + 完整 LD_LIBRARY_PATH/PATH/C_INCLUDE_PATH，全部指向 /public/dtk-26.04）+ `conda activate <env>`
- 用法：`source /public/dcu-env/activate_dcu_env.sh`

### 6. 验证（"能用"的定义）
- `python -c 'import torch; print(torch.__version__, torch.cuda.device_count()); x=torch.randn(1024,1024,device="cuda:0"); print((x@x).sum().item())'`——DCU 实际计算
- 核心框架包逐个 import（sglang/vllm/transformers/triton）
- pip freeze 与镜像 diff（见步骤 2）
- **Triton clang 路径**：`python -c 'from triton.backends.amd.compiler_hcu import HIPBackend; print(HIPBackend.path_to_rocm_clang())'`——必须是 `aillvm/bin/clang-18`（见坑 8）
- **模型实际运行测试**（用户硬要求）：起推理服务 + 发请求拿到生成结果，才算交付完成

### 6b. 验证"与镜像完全一致"（对照**有效** server_args，不是命令行）
用户硬要求"跟镜像里完全一致"时，**别信启动命令行**——命令行可能含已废弃/被框架覆盖的 flag（如 `--enable-piecewise-cuda-graph` 在 DCU 上被 `is_hip()` 覆盖，实际没生效）。正确做法：
1. 找到跑**目标镜像**的容器（`docker ps` 看 Image 列，**别认错镜像**——同机常有多个不同镜像的容器，如 `<internal_registry>/jenkins/...` 测试镜像 vs `<internal_registry>/custom:...` 目标镜像）
2. 从该容器日志抓**有效** `ServerArgs(...)` dump：`docker logs <c> 2>&1 | grep "server_args=ServerArgs"`——这是框架解析 + 自动覆盖后的**真实运行态**
3. 从原生环境日志抓同样的 `ServerArgs(...)` dump
4. 逐字段对比（`disable_cuda_graph` / `disable_piecewise_cuda_graph` / `enforce_piecewise_cuda_graph` / `cuda_graph_max_bs` 等）
5. 差异归因：tp 不同导致的 `cuda_graph_max_bs` 差异属正常；其余应一致
**本会话实例**：镜像容器命令行带 `--enable-piecewise-cuda-graph`，但有效 server_args 是 `disable_piecewise_cuda_graph=True`（`is_hip()` 自动禁用）——按命令行会误判"镜像开了 piecewise"，按有效 server_args 才确认"镜像 piecewise 也是关的，与原生一致"。

### 7. 启动服务 + 模型实测（sglang 0.5.12 实测参数）
以 Qwen3.8-27B（52G）TP8 为例（2026-09-02 在 <<internal-ip>> 实测通过）：

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MARLIN_W16A16_MOE=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_ROCM_USE_AITER_MOE=False
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_ENABLE_SPEC_V2=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -u -m sglang.launch_server \
  --numa-node 0 0 0 0 0 0 0 0 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 64 \
  --kv-cache-dtype fp8_e5m2 \
  --model-path <model> \
  --host 127.0.0.1 --port 30999 \
  --served-model-name <name> \
  --mm-attention-backend fa3 \
  --enable-piecewise-cuda-graph \
  --tp-size 8 --pp-size 1 \
  --chunked-prefill-size 8192 \
  --watchdog-timeout 3000 \
  --attention-backend fa3 \
  --json-model-override-args '{"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}' \
  --context-length 1000000 \
  --page-size 64 \
  --mem-fraction-static 0.83 \
  --keep-mm-feature-on-device \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-cache-report
```

关键参数说明：
- `--attention-backend fa3` + `--page-size 64`：默认 aiter backend 在 sglang 0.5.12 下缺 `paged_attention_ragged` 等函数（镜像同样缺，aiter 顶层无这些函数），会 NameError + HSA VMFault；fa3 要求 KV block size 64（见坑 9）
- `--kv-cache-dtype fp8_e5m2`：BW150/BW1000/BW100 用 e5m2，BW1100 用 e4m3（性能差异大）
- `--json-model-override-args` 的 rope 参数：Qwen3.8-27B（Qwen3_5 架构）必需
- 启动成功标志：日志出现 `The server is fired up and ready to roll!`

接口测试（交付前必跑）：
```bash
curl http://127.0.0.1:30999/health          # 200 即健康（空 body 正常）
curl http://127.0.0.1:30999/v1/models       # 模型列表
curl http://127.0.0.1:30999/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<name>","messages":[{"role":"user","content":"计算 123*456 等于多少，只回答数字"}],"max_tokens":512,"temperature":0}'
# 期望 content 返回 56088
```

## 已验证修复（2026-09-02 全部实测通过）

### accelerate 循环导入（.pth 预导入修复）
现象：`python -m sglang.launch_server` 报 `ImportError: cannot import name 'dispatch_model' from partially initialized module 'accelerate.big_modeling'`（accelerate 1.13.0 内部循环：`__init__`→`accelerator`→`big_modeling`→`hooks`→`utils`→`bnb`→`big_modeling` 未初始化完）。镜像内同样 import 链正常（导入顺序不同：镜像里 `accelerate.utils` 先于顶层 `accelerate` 被完整导入）。

**已验证修复**：在 site-packages 放 `.pth` 预导入：
```bash
echo 'import accelerate.utils' > <site-packages>/_dcu_accelerate_preimport.pth
```
`.pth` 在 site 初始化阶段执行，早于任何业务 import，打破循环。实测 `python -m sglang.launch_server` 正常启动。

诊断方法（复现时）：`python -v -c '<失败import>' 2>&1 | grep -n "import 'accelerate"` 对比两侧导入顺序。

## Kylin V10 / 老 glibc 系统（2026-09-03 在 <<internal-ip>> 端到端验证通过）

把 ubuntu22.04 镜像（glibc 2.35）复刻到 **Kylin V10（内核 4.19，系统 glibc 2.28）** 时，**核心新增问题是 glibc 版本不匹配**——镜像 .so 若编译针对 2.35，拷到 2.28 系统加载失败。

**先做 glibc 预检（高效，别先拷 28GB）**：`objdump -T /lib64/libc.so.6 | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sort -Vu | tail -1` 看系统最高符号版本；再在探针容器里全量扫镜像 .so 找需求 > 系统版本的。实测 sglang0.5.12 镜像 850 个 .so 只有 7 个需求 > 2.28，**真正卡脖子的只有 `sgl_kernel/common_ops.so`（GLIBC_2.32，sglang 核心算子库）**，其余 6 个（cryptography/grpc_router/st_attn/vsa 等）不在核心推理路径。

**已实测失败、别再试**：① LD_PRELOAD shim 提供 `__libc_single_threaded` 符号——无效（报错是 `version 'GLIBC_2.32' not found` 版本节点缺失，不是 undefined symbol）；② `conda install glibc`——conda-forge linux-64 无此包；③ patchelf 改 **conda** python 的 PT_INTERP 到 glibc 2.35 的 ld——conda python 是 2.28 编译的，`import sglang` 间歇 segfault；④ LD_PRELOAD glibc 2.35 的 libc——segfault。

**最终方案（端到端验证通过，服务起 + 模型推理正确）**：
1. **镜像 python3.10**（2.35 编译、静态链接 libpython、自包含）+ **patchelf**：`--set-interpreter` 指向镜像 glibc 2.35 的 ld-linux，`--set-rpath` 指向 imglib（镜像系统库）+ **原生 DTK** + extra-libs。python 的 glibc 2.35 自洽，不靠 LD_LIBRARY_PATH。
2. **libgalaxyhip 实例统一用原生 DTK**（/opt/dtk-26.04，2.28 编译）：torch（走 RPATH）+ Triton launcher（走 HIP_PATH）+ rocminfo（走 PATH）必须同一份，否则跨实例指针互不认 → Triton kernel 报 `Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)`。原生 DTK 的 libgalaxyhip 2.28 编译，image ld 2.35 和系统 ld 2.28 都能加载，一份通吃。镜像 DTK 只用来补 aillvm（ROCM_PATH 指向它，Triton 编译用）。
3. **wrapper（dcu_python）**：LD_LIBRARY_PATH 只含原生 DTK + extra-libs（libgfortran/libquadmath，**绝不放 glibc 核心**，否则系统工具 cat/sh/rocminfo 继承后加载 2.35 版崩溃）+ PYTHONPATH + DTKROOT/ROCM_PATH=镜像 DTK + HIP_PATH=原生 DTK + PATH 原生 DTK 优先。
4. **pyconfig.h 换 48K 真版**（镜像的是 4K multiarch wrapper，`#include <x86_64-linux-gnu/...>` 找不到 → sglang JIT 编译 hip_utils 失败）。
5. **sglang 启动加 Kylin 特有参数**：`--disable-radix-cache`（Qwen3_5 Mamba 架构 radix cache 强制 page_size=1，与 fa3 要求 64 矛盾）。
   - **CUDA graph 不用禁（2026-09-03 复测纠正）**：标准 decode CUDA graph 在 v2 wrapper（libgalaxyhip 统一原生 DTK）下**正常跑**——默认 max_bs=256、capture 27 个 batch size 无 static TLS 报错、推理正确（56088）、decode 日志 `cuda graph: True`、吞吐 22 tok/s（关 CUDA graph 时 12.9，提速 70%）。之前"static TLS 耗尽必须 --disable-cuda-graph"是误判（旧 wrapper 双实例 / piecewise 层导致）。
   - **piecewise CUDA graph 已与镜像一致（2026-09-03 对照镜像容器确认，无需额外操作）**：DCU 是 HIP 平台，sglang `server_args._handle_piecewise_cuda_graph()` 里 `if is_hip(): self.disable_piecewise_cuda_graph = True` **自动禁用** piecewise。对照目标镜像容器（`lcq-qwen3.8`，镜像正是 `...custom:sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620`）实际 server_args：`disable_cuda_graph=False`（CUDA graph 开）+ `disable_piecewise_cuda_graph=True`（piecewise 关）+ `enforce_piecewise_cuda_graph=False`——**与 <device-B> 原生完全一致**。容器命令里那个 `--enable-piecewise-cuda-graph` 是**已废弃 flag**（启动日志明确提示 deprecated），最终 `enforce=False` 被 `is_hip()` 覆盖，**根本没生效**。**不要**加 `--enforce-piecewise-cuda-graph` 强行开 piecewise——那反而与镜像不一致（镜像里 piecewise 也是关的）。

其他 Kylin 差异：<device-B> 原生 DTK 26.04 缺 aillvm（拷镜像完整 DTK 解决）；缺 /opt/ucx、/opt/hwloc（从镜像拷）；docker cp 的 `/.` 合并语法某些版本静默失败（先拷全新临时目录再合并）；模型 rope 参数按 config 匹配（Qwen3.6 rope_scaling=null 不能套 Qwen3.8 的 yarn override）。

**完整方案 + 已就位资产 + 验证结果见 `references/kylin-old-glibc.md`。**

## 坑速查
1. docker cp dist-packages → 嵌套子目录，必须拍平
2. /usr/lib/python3/dist-packages 的 Ubuntu 系统包会漏 → six 缺失
3. docker cp 遇坏符号链接中断 → 用 tar 管道
4. 物理机 /opt/dtk-26.04 可能不完整 → 验证完整布局
5. DTK 库 RUNPATH=/opt/dtk 写死 → LD_LIBRARY_PATH 必须显式
6. 入口 shebang 指向 /usr/bin/python3 → 必须修
7. 先查磁盘 → 环境放大盘（/public）
8. **DTK 目录名决定 Triton 选哪个 clang**：`path_to_rocm_llvm()` 靠 `rocm_path.name=="dtk"` 判断——目录名 `dtk` 用 `aillvm/bin/clang-18`（正确），目录名 `dtk-26.04` 用 `llvm/bin/clang-17`（错误，Triton kernel 编译报 `HSACOError: File operation failed: No such file or directory: /tmp/tmp*.amdgcn`）。修法：建软链保证目录名为 dtk（`ln -sfn /public/dtk-26.04 /public/dtk`，再 `ln -sfn /public/dtk /opt/dtk`），ROCM_PATH 指向它
9. **aiter attention backend 缺函数**：sglang 0.5.12 默认 aiter backend 缺 `paged_attention_ragged`/`mha_batch_prefill_func` 等（aiter 顶层无这些函数，镜像同样缺）→ NameError + HSA VMFault。修法：`--attention-backend fa3` + `--page-size 64` + `--kv-cache-dtype fp8_e5m2`
10. **推理模型 content 为 null**：Qwen3.8-27B 等推理模型 max_tokens 太小时答案全在 `reasoning_content` 字段、`content` 为 null（finish_reason=length）。调用时 max_tokens ≥512
11. **pkill 自杀陷阱**：`pkill -f 'sglang.launch_server'` 会匹配到 SSH 命令行自身（命令行含该字符串）→ 杀掉自己的 shell → SSH 255、新进程没起来。修法：`pgrep -f 'sglang[.]launch_server'`（正则 `[.]` 不匹配命令行里的字面量）
12. **老 glibc 系统（Kylin V10 2.28）镜像 .so 加载失败**：先 `objdump -T /lib64/libc.so.6` 看系统最高 GLIBC 符号版本，再扫镜像 .so 找需求更高的。报错 `version 'GLIBC_2.3x' not found` 是**版本节点**缺失，LD_PRELOAD shim 提供符号**无效**；conda python 是系统 glibc 编译的，patchelf 改 ld / LD_PRELOAD 换 libc 都会间歇 segfault。修法：用镜像自带 python（同 glibc 基线）+ 镜像 glibc 的 ld-linux 启动。详见 `references/kylin-old-glibc.md`
13. **docker cp 的 `/.` 合并语法静默失败**：某些 docker 版本 `docker cp img:/dir <existing_dir>` 只拷部分（exit 0 但 599 项只拷 34 项）。修法：先 `docker cp` 到全新临时目录再合并，或 `rm -rf` 目标后让 docker cp 直接创建
14. **模型 rope 参数要按 config 匹配**：Qwen3.6-27B（Qwen3_5 架构）`text_config.rope_scaling=null`，不能套用 Qwen3.8 的 `--json-model-override-args` yarn rope override。用前查 `config.json` 的 `text_config.rope_scaling`，null 就不传 override
15. **老 DTK 原生安装缺 aillvm**：物理机原生 DTK 26.04 可能只有 llvm/clang-17、缺 aillvm/clang-18（镜像 DTK 有）→ triton 编译失败。修法：拷镜像完整 DTK 到独立目录 + 软链保证目录名为 `dtk`（同坑 8）
16. **libgalaxyhip 双实例（Kylin 特有，最隐蔽）**：torch 和 Triton launcher 加载了**两份不同** libgalaxyhip（torch 走 python RPATH、Triton launcher 走 HIP_PATH/LD_LIBRARY_PATH，指向不同 DTK 目录）→ 跨实例 HIP 指针互不认 → Triton kernel 报 `ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)`（`hipPointerGetAttribute` 返回 `hipErrorInvalidValue`）。诊断：`python -c "import torch; x=torch.randn(8,device='cuda:0'); print([l.split()[-1] for l in open('/proc/self/maps') if 'libgalaxyhip' in l][0])"` 看 torch 实际加载哪份。修法：torch + Triton launcher + rocminfo **全用原生 DTK**（2.28 编译，image ld 2.35 和系统 ld 2.28 都能加载，一份通吃）——python RPATH 指向原生 DTK、wrapper 的 HIP_PATH=原生 DTK、PATH 原生 DTK 优先；镜像 DTK 只用来补 aillvm（ROCM_PATH 指向它）
17. **static TLS 耗尽（Kylin 老 glibc 2.28 + 老内核 4.19 特有，2026-09-03 复测重新归因）**：现象 `OSError: /usr/lib64/libgomp.so.1: cannot allocate memory in static TLS block`。**复测结论**：标准 decode CUDA graph（默认 max_bs=256、27 个 batch size）在 v2 wrapper 下**不触发**——之前失败是旧 wrapper（libgalaxyhip 双实例）/ piecewise 层导致，不是标准 capture 本身。sglang 0.5.12 有**两层** cuda graph（标准 + piecewise）：标准层已验证可跑；**piecewise 层在 DCU（HIP）上被 sglang 自动禁用**（`is_hip()` → `disable_piecewise_cuda_graph=True`），与镜像行为一致（镜像容器 server_args 也是 `disable_piecewise_cuda_graph=True`），**无需处理**。若复现该报错：先确认 wrapper 是 v2（libgalaxyhip 统一原生 DTK）；`--disable-cuda-graph` 是最后手段（decode 吞吐掉 ~40%）
18. **pyconfig.h multiarch wrapper（Kylin 特有）**：镜像 `/usr/include/python3.10/pyconfig.h` 是 **4K multiarch wrapper**（`#include <x86_64-linux-gnu/python3.10/pyconfig.h>`），真正的 48K 版在 multiarch 路径。只拷 wrapper → sglang JIT 编译 hip_utils 时 gcc 报 `x86_64-linux-gnu/python3.10/pyconfig.h: 没有那个文件或目录`。修法：`docker cp img:/usr/include/x86_64-linux-gnu/python3.10/pyconfig.h <imgpython>/include/python3.10/pyconfig.h`（替换为 48K 真版）
19. **Qwen3_5 Mamba page_size 冲突（Kylin 特有，模型架构相关）**：Qwen3.5/3.6 是混合 Mamba/线性注意力架构，radix cache + Mamba no_buffer 模式**强制 page_size=1**（日志：`requires page_size=1 ... but got 64. Automatically setting page_size=1`），与 fa3 要求 paged KV block=64 矛盾 → `RuntimeError: Paged KV cache block size must be 64`。修法：`--disable-radix-cache`（让 page_size 保持 64）
20. **tar 解包 /usr/include/python3.10 覆盖 python 二进制（Kylin 特有）**：`docker exec img tar cf - -C /usr/include python3.10 | tar xf - -C <imgpython>/` 会把头文件目录解到 `<imgpython>/python3.10/`，**覆盖** `<imgpython>/python3.10`（python 二进制）→ wrapper 报 `is a directory`。修法：头文件解到 `<imgpython>/include/python3.10/`，二进制单独放 `<imgpython>/python3.10`（先备份 `.bak`，被覆盖后从 `.bak` 恢复 + 重新 patchelf）
21. **dcu_python 继承 LD_PRELOAD=glibc_shim.so 启动即段错误（Kylin 特有，2026-09-03 实测）**：现象 `./dcu_python`（或 pyimg）启动即段错误（EXIT 139，dmesg `segfault ... in libc.so.6`，backtrace `__libc_early_init`），但自己 ssh 跑正常。根因：`activate_dcu_env.sh` 有 `export LD_PRELOAD=<dcu-env>/glibc_shim.so`（shim 是给 **conda python/glibc 2.28** 补 `__libc_single_threaded` 等符号的），imgpython 是 **glibc 2.35 自洽**不需要 shim；用户先 `source activate_dcu_env.sh` 再跑 `./dcu_python`，shim 被预加载进 glibc 2.35 进程 → `__libc_early_init`（main 之前的动态链接阶段）写未映射页段错误。修法：wrapper 的 exec 前加 `unset LD_PRELOAD`（自防御，见 `references/kylin-old-glibc.md` §2.5/§5）。诊断：非交互 ssh 不读 .bashrc 不复现、交互式登录才复现；`dmesg -T | grep segfault` → `coredumpctl info <PID>` → `lz4 -d` core → `gdb -batch -ex "info proc mappings"` 看实际加载了哪个 libc/shim

## 支撑文件
- templates/activate_dcu_env.sh — 已验证的激活脚本模板（DTK 26.04 经 /opt/dtk 软链指向 /public/dtk-26.04 + conda activate；路径按实际拷贝位置调整）。
- references/kylin-old-glibc.md — Kylin V10 / 老 glibc 系统移植（**端到端验证通过**）：glibc 预检方法、7 个高版本 .so 清单、4 个已实测失败方案、最终方案（镜像 python + patchelf + libgalaxyhip 统一原生 DTK + wrapper + pyconfig.h 真版 + Kylin 启动参数）、<device-B> 已就位资产、验证结果。
- 配套 `.pth`：`<site-packages>/_dcu_accelerate_preimport.pth`，内容一行 `import accelerate.utils`（打破 accelerate 1.13.0 循环导入，见"已验证修复"段）。
