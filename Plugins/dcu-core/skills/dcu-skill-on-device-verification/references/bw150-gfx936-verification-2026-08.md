# DCU 技能真机验证 — <device-A> 设备实测（2026-08）

一次完整的「技能库真机验证」落点记录：命令脚手架、设备快照、各技能对照结论。
下次验证可照此模板复制。

## 设备快照（<device-A>）

- OS: Ubuntu 22.04, kernel 5.15.0-25, hostname <internal-host>
- CPU: Hygon C86 7490 64-core ×2 (128 核, 8 NUMA nodes: 每 node 16 核)
- DCU: 8 卡 HCU (C-3000 系)，`lspci -nn` ID **1d94:6320**（= BW150/BW1000 系，**非凭证标的 BW100**）
- 驱动: 6.3.30-V1.4.1a; hy-smi v1.24.0
- DTK: /opt/dtk + 24.04.1/2/3, 25.04(.1), 26.04 多套
- kernel cmdline: `iommu=off`；`/sys/kernel/iommu_groups` 空；DCU 无 ACS 行（视为关闭）
- Docker root: /public/docker（133G 可用）；根分区 / 仅 6.4G（99%）
- 可用镜像（跨 sglang/vllm/pytorch/migraphx 各版本）→ 验证容器选轻量 pytorch dtk26.04

## SSH 辅助封装（python，凭证入 SSHPASS 不落盘）

```python
import json, os, subprocess, shlex
DEV_KEY = "<internal_device_ip>-BW100X8"   # 凭证 key 是 IP-机型 格式，用 IP 前缀匹配（真实 IP 在 credentials.json 的 key 里）
def run(cmd, timeout=120, host="<internal_device_ip>"):
    pw = json.load(open(os.path.expanduser('~/.config/dcu-toolkit/credentials.json')))[DEV_KEY]['password']
    env = dict(os.environ); env['SSHPASS'] = pw
    p = subprocess.run(
        f"sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@{host} {shlex.quote(cmd)}",
        shell=True, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr
```

写文件到远端：
```python
env['SSHPASS'] = pw
subprocess.run(f"sshpass -e ssh -o StrictHostKeyChecking=no root@{host} \"cat > /path\"",
               shell=True, env=env, input=content, capture_output=True, text=True, timeout=60)
```

## 起验证容器（常驻，非 --rm）

```bash
docker run -itd --name dcuval --network=host \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro --shm-size=2g \
  -v /tmp/v:/work \
  harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.11 bash
# 执行一律 docker exec <c> bash -ilc '...'（-ilc 加载 DTK 环境）
```

验证完 `docker rm -f dcuval; rm -rf /tmp/v`（只删自己的）。

## 逐技能验证命令与结果

### dcu-env → check_iommu_acs.sh
传脚本进容器实跑，输出 `{iommu:disabled, acs:disabled, all_disabled:true}`，exit=0，stderr 提示「可以跑通信测试」。
与技能「基线合格」判定一致。

### dcu-hy-smi
```
/opt/hyhal/bin/hy-smi --showtoponuma --json   # 8卡全 node0（单NUMA）✅
/opt/hyhal/bin/hy-smi --showtopotype          # 表格（非json）✅ 印证仅 showtoponuma 支持 json
/opt/hyhal/bin/hy-smi --showtopohops          # 组内1hop/跨组2hop ✅
/opt/hyhal/bin/hy-smi --showpids              # No KFD PIDs
```
hop/linktype 印证拓扑卡0-3 HSL、卡4-7 HSL、跨组 PCIE。

### dcu-dtk-layout（容器内）
```
source /opt/dtk/env.sh
cat /opt/dtk/.dtk_version                 # DTK-26.04-rc4
echo $AMDGPU_TARGETS                      # gfx906;gfx926;gfx928;gfx936;gfx938
readlink -f /opt/dtk/bin/hipcc            # .../dcc/bin/dcc
readlink -f /opt/dtk/bin/rocm-smi         # .../hy-smi-wrapper
which hipcc rocminfo rocm-smi gpurun hipgdb clang-ocl extractkernel
```
全部就位；gpurun 在 `/opt/dtk/llvm/bin/gpurun`（技能写 dcc/bin，实为 llvm/bin 同源，非错误）。

### dcu-topology → collect_topology.sh -j
输出：8 卡、8 NUMA、128 核、dcu_source=hy-smi、cross_numa=false。正确用 co-processor 抓 8 真卡（无 230 假 DCU）。

### dcu-cpu-affinity → gen_affinity.sh
单 NUMA → 8 卡全绑 node0、标注「保险式绑定」，默认 dry-run 未执行 numactl。

### dcu-kernel-dev → dcu_detect_arch.sh + dcu_build_kernel.sh
```
bash dcu_detect_arch.sh        # gfx936（与技能 §13.5 记录的 <device-A>=gfx936 一致）
bash dcu_build_kernel.sh hip_gemm_tiled.cpp -o gemm_demo
./gemm_demo
# METRIC max_diff=0.000977
# METRIC errors=0
# METRIC status=PASS   ← 与技能 2026-08-06 首批记录的数字完全一致
```

### dcu-cuda-hip-port → hipify + hipcc
```
/opt/dtk/hip/bin/hipify/hipify-perl vecadd.cu > vecadd_hip.cpp   # HIPIFY_RC=0, cudaMalloc残留0
/opt/dtk/bin/hipcc --offload-arch=gfx936 vecadd_hip.cpp -o vecadd_demo
./vecadd_demo   # METRIC max_err=0.000000 status=PASS
```

## 结论

6 个技能全部生效、内容准确、无错误 → 技能库「地基」可信。可作为后续验证的对照基线。

---

## 扩展验证（第 2-3 轮，累计 17 技能）

首轮 6 个之后，按 A→B→C 依次又验证 11 个，全部生效。新增对照点：

### dcu-rccl-test（通信）
- 物理机 `rocminfo`（带 ROCM_PATH/LD_LIBRARY_PATH，用 25.04 的）确认 **gfx936**（26.04 路径无 rocminfo 可执行）。
- 起 vllm0.18.1 dtk26.04 容器，容器内 8 卡可见、RCCL/MPI 就位、NCCL_NCHANNELS_PER_PEER 未预置。
- 按技能编译法（commit `5e838ad` + `make ROCM_HOME=/opt/dtk NCCL_HOME=/opt/dtk/rccl CUSTOM_RCCL_LIB=... MPI=1 MPI_HOME=/opt/mpi -j8`）编译 rccl-tests 成功。
- 8 卡 all_reduce / alltoall 跑通：设备识别 **BW150**，#wrong 全 0、Out of bounds 0 OK。完整原始日志按技能规范不裁剪/不均值/不分析。

### dcu-vllm-deploy（部署）
- TP=8 全卡起 Qwen3.6-27B（bfloat16/enforce-eager/max-model-len 8192）`Application startup complete`，推理"1+1=2"正确。
- `--tensor-parallel-size`/`--enforce-eager`/`--kv-cache-dtype`(含 fp8_e5m2/fp8_e4m3)/`--enable-prefix-caching` 均确认存在。
- ⚠ 发现 `--device rocm` 参数在 0.18.1 无（见差异 1）。

### dcu-sglang-source（源码索引）
- 用技能来源镜像 `dsv4-flash-k100ai-sglang0.5.12-20260728` 验证：`is_dcu()` 命中、`dcu_mla_backend.py` 存在、server_args 含 `dcu_mla` → 技能记录 0.5.12 准确。
- 用更新的 0.5.15 镜像验证：发现术语演进（见差异 2）。

### dcu-perf-profiling / dcu-hipprof（性能）
- 用 GEMM demo 实跑 `hipprof --pmc --pmc-type 3`：采集到真实 kernel 数据，计数器 **SQ_INSTS_MMOP=0 / SQ_LDS_BANK_CONFLICT / wave_size=64** 与技能映射表吻合。
- `hipprof --hip-trace --stats`：出 `trace_out.hipkernel.csv`/`.hiptrace.csv`/`.db`（与 §8.7 命名一致）；`--leak-check` 正常介入。
- ⚠ 工具栈发现：DTK-26.04 容器只有 hipprof，rocprof/rocprofv2 在宿主 `/opt/dtk-25.04/rocprofiler/bin/`（见差异 3）。

### dcu-das-hub-api（包源，纯网络）
- 根域名 SPA 200；无 `X-Das-Source` 头 → `Access Denied: Invalid Client Source`（逐字吻合）；带头返回 repos。
- 列版本 API 返回 sglang 多版本，内网 URL 格式 `http://<internal_ip_pypi_storage>:9929/...` 与 §3 一致。
- **内网→公网 URL 改写下载：实际 sha256 与 API 期望值完全一致**（HTTP 200, 7.7MB wheel 校验通过）。

### dcu-llm-benchmark（benchmark CLI）
- `vllm bench serve --help=all`：`--served-model-name`(有)/`--served-model`(无裸名)、`--dataset-name` 枚举与 §2.2 的 0.18.1 集完全一致（含 custom_mm、无 speed_bench）、`--backend` 含 vllm-pooling。
- `sglang.bench_serving --help`：`--served-model-name`/`--random-range-ratio`/`--disable-ignore-eos`/`--flush-cache`/`--warmup-requests`/`--tokenize-prompt`/`--fake-prefill` 全存在，与 §3.2 吻合。

### dcu-migraphx-perf / dcu-ort-perf（引擎）
- migraphx-driver 5.2.0，路径 `/opt/dtk/bin/migraphx-driver` 与 §1.2 一致；子命令（compile/onnx/op/params/perf/read/roctx/run/tf/time/verify）与 §10 速查表完全吻合。
- ORT 1.19.2 在 DCU 可用，providers 含 MIGraphXExecutionProvider/ROCMExecutionProvider/CPUExecutionProvider（paddle 镜像）。注：该镜像 ORT 推理因 numpy 版本冲突（_ARRAY_API not found）跑不通，属镜像环境问题，非技能内容错误。

## 差异定性（双版本对照法）

用户硬要求：「有差异看一下是不是通用差异还是特定版本的差异，再更新」。定性手段 = **同时起技能来源版本 + 一个更新版本镜像，对同一断言各跑一遍**：

| 技能 | 差异 | 双版本对照证据 | 定性 | 处理 |
|---|---|---|---|---|
| dcu-vllm-deploy | `--device rocm` 无 | 0.15.1 和 0.18.1 的 `vllm serve --help=all` 都无 `--device`；envs.py 有 `VLLM_TARGET_DEVICE` 环境变量 | **通用**（vLLM platform plugin 机制替代，跨版本都成立） | 标「已废弃」 |
| dcu-sglang-source | is_dcu→is_hcu | 0.5.12 镜像有 `is_dcu()`/`dcu_mla_backend.py`/`dcu_mla`；0.5.15 镜像无 is_dcu，改 `is_hcu()`/`hcu_mla_backend.py`/`HCU_ATTENTION_BACKEND_CHOICES` 含 `hcu_mla` | **版本特定**（海光 fork 术语演进，技能记录 0.5.12 本身没错） | 补版本演进对照表，保留旧版 |
| dcu-perf-profiling | rocprof 定位 | 26.04 容器 `/opt/dtk/bin` 只有 hipprof；宿主 `/opt/dtk-25.04/rocprofiler/bin/` 有 rocprof/rocprofv2 | **版本/镜像特定** | 工具栈表补版本说明 |

三处差异已回填对应技能并 commit+push（e26a0fd → origin master）。
