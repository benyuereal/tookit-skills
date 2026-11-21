---
name: dcu-ontrack-experience
description: DCU 故障排查速查：按报错关键词反查根因与规避手段（容器/驱动/可见性、vLLM/SGLang 推理、RCCL 集合通信、库缺失与编译、硬件 RMA 提示、驱动 dmesg 规则引擎）。触发场景：拿着一段报错日志定位根因、多机训练卡住、容器里设备异常。不适用：进程崩溃后的 coredump 分析走 dcu-vmfault-coredump；分布式 hang 的系统化定位走 debug-distributed-hang；精度问题走 dcu-vllm-precision / dcu-train-precision。
version: 1.2.3
author: DCU Engineer
license: MIT
metadata:
  toolkit:
    tags: [dcu, hygon, troubleshooting, ontrack, fault, driver, vllm, rccl, compile]
    related_skills: [dcu-driver, dcu-docker, dcu-vllm-deploy, dcu-rccl-test, dcu-env, dcu-perf-profiling]
  source:
    provenance: "海光 DCU 故障排查经验速查（按报错关键词反查）"
---

# DCU 故障排查经验速查（dcu-ontrack-experience）

> **命名说明**：ontrack 为内部工单系统名。本 skill 是**脱敏后的故障经验库**（报错关键词 → 根因 → 规避手段），不含任何工单号、内部链接或内部人名。

## 触发条件
遇到以下任一类问题、且不确定根因时，先在本手册按关键词 grep：
- 容器里 `hy-smi` 看不到卡 / DCU 不可见 / 驱动加载失败
- vLLM / SGLang 启动崩溃、精度乱码、toolcall 失效、长上下文报错
- RCCL/NCCL 多机多卡通信卡死、带宽低、allreduce 异常
- 推理/训练报 `libxxx.so` 缺失、hip/nvcc 编译失败、numpy/transformers 版本冲突
- 掉卡、VMFault、内核切换后驱动 down
- **`dmesg` 出现 XID 错误码（如 XID 190/301）** → 查 `references/xid_sxid_errors.md`（正向错误码速查：错误码→含义/原因/处理）
- **HySwitch 互联报 SXID 错误码（如 SXID 207）** → 查 `references/xid_sxid_errors.md`（SXID 列表 + HySwitch HW/HFM 处理策略）

## 纪律（必读）
- 本手册**只沉淀踩坑经验与可复用规避手段**，不写性能数值、不写逐行日志、不写客户名/行业/项目名。
- 内网镜像仓库域统一记为 `<internal_registry>`（落地时替换为实际 harbor/image 域名）。
- 每条格式：**现象关键词 → 设备型号 → 根因 → 规避/修复手段**。
- 硬件损坏类（UE 错误、ubb 未供电、PCIe 卡槽故障、锁频 RMA）不在本手册——那类走换卡/报修，非技术可避。

## ⚠️ 使用须知（务必先读）
1. **本手册仅供参考，不保证完全修复**。所列经验来自历史工单沉淀，实际环境（驱动版本、DTK、机型批次、固件）千差万别，同样报错可能根因不同。使用前先核对**设备型号 + 驱动/DTK 版本 + 报错原文**是否匹配，不要无脑套 env。
2. **涉及高危操作务必谨慎**：
   - 重启物理机 / 重装驱动 / 刷 vbios / 改 BIOS（关 IOMMU、ACS、NUMA）/ 拔插电源线 / 改内核启动参数 等操作，可能影响整机业务或导致设备离线。
   - 高危操作前：**先确认是否在产线/客户环境**（避免在业务运行中机器直接操作）、**确认有带外管理（BMC/IPMI）可恢复通道**、**必要时应先与硬件/产线团队确认**。
   - 本手册中带"重启物理机""重装驱动""刷 vbios""改 BIOS""拔电源"字样的条目，执行前多想一步，优先在测试机/空闲节点验证。
3. **先观测后动手**：能用 `hy-smi` / `dmesg` / `lspci` / 日志定位的，先定位再改，不要靠试错堆 env。
4. 若按手册操作后仍异常，说明非该经验覆盖场景，回到正常排障流程（查日志、提工单、问研发），不要反复堆砌环境变量。

---

## A. 容器 / 驱动 / 可见性

| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| 容器内 hy-smi 看不到卡、DEVCOUNT=0 | 全系 | 未挂 hyhal | 创建容器必挂 `-v /opt/hyhal:/opt/hyhal:ro`；已挂仍异常→容器内重解压 hyhal 包 |
| 容器内起服务报 amdsmi/lib 缺失 | 全系 | 未 source DTK 环境 | 服务启动前 `source /opt/dtk/env.sh`（含 cuda/env.sh） |
| 装驱动后 hy-smi 仍报错 / 驱动 down | K100-AI/BW | 驱动与 vbios 版本不匹配 | 装驱动时各选项选 Y 自动刷 vbios；多卡不一致跑 `vbios_upgrade.sh` |
| 增减桥接卡/换卡后 topo 异常 | K100-AI | 未重刷固件 | 换卡后必须重装驱动重刷 vbios |
| 虚拟机直通 DCU 驱动加载失败 | Z100L/BW | 内核/直通延迟 | 物理机先装驱动刷 vbios，再进虚拟机装（不刷 vbios）；`modprobe hygon` 等映射延迟 |
| 自升内核后驱动编译失败 | 全系 | kernel-devel 不匹配 | 装系统后 `/etc/yum.conf` 加 `exclude=kernel*`；已踩坑则回退内核或装对应 kernel-devel |
| 定制 OS 装驱动失败（凝思 6.0.80 等） | BW | 无适配记录 | 换社区版 openEuler 22.03 SP3 / 凝思 6.0.99 |
| 多卡通信掉卡 / 带宽低 | K100-AI/BW | 多为 PCIe 卡且 BIOS 未关 IOMMU+ACS（**OAM 产品 BW1000/BW1100 基本不受影响，先确认卡形态再处理**） | PCIe 卡：BIOS 关 IOMMU、关 ACS（关 ACS 后 NUMA 需 BIOS 团队确认同步）；OAM 机型排查其他根因，勿盲目改 BIOS |
| `linux-modules-extra` 缺失（IOMMU v2 报错） | 全系 | 未装额外内核模块 | `apt install linux-modules-extra-$(uname -r)` |
| hyqual 容器报 libkmod 缺失 | BW150 | 容器未映射内核模块 | 创建容器加 `-v /lib/modules/$(uname -r):/lib/modules/$(uname -r)` |
| 新支点 OS 脚本无法执行 / tmp 无权限 | 全系 | OS 限制 | `bash xxx.sh` + 改用用户目录 tmp |
| `max_map_count` 不足导致推理失败 | BW | 内存映射限制 | `echo 262144 > /proc/sys/vm/max_map_count`（重启失效） |
| 安全套件与驱动冲突 | 全系 | 北信源等拦截 | 卸载冲突安全套件 |
| `hfmtool`/`hdms`/`hymgr`/`criu` 报 `error while loading shared libraries: libhfmlib.so/libhydmi.so/libhydrm.so` | 全系 | 未加载 hyhal 库路径 | 先 `source /opt/hyhal/env.sh`（或 `LD_LIBRARY_PATH=/opt/hyhal/lib`）再跑；`hy-smi`/`drvdiag`/`mmio_rw`/`hlinkmanager` 不受影响（<device-A> 实测） |
| `hfmtool -t/-b/--get_uuid` 报 `state is -805/-810, data_size ERROR` | PCIe 无 HSW 机型（BW100 等） | 无 HySwitch 可查，**非故障** | 勿据此判驱动异常；OAM+HSW 机型（BW1000/BW1100）才有真实数据（<device-A> 实测） |
| `hycc` 报 `missing clang path` | 全系 | 未设 clang 路径 | `export HIP_CLANG_PATH=<clang路径>`（DTK 环境即 `/opt/dtk/llvm/bin`） |
| `hybandwidth` 报 `Failed to load HIP library` | 裸机 | 宿主机无 HIP 运行时 | 进容器 `source /opt/dtk/env.sh` 后再跑 |
| 驱动工具"找不到"但 `/opt/hyhal/bin` 里有 | 部分机器 | 服务/工具实际在 `/usr/local/hyhal`（与 /opt/hyhal 并存的独立目录） | `ps aux \| grep -E "hfm\|hymgr"` 确认服务真实路径；两目录 bin 清单一致，版本以运行路径为准（<device-A> 实测） |
| BIOS/lspci/i2cdetect 均检测不到 DCU（三层全盲） | 全系 | vbios 异常（与驱动无关，驱动层以下） | 带外（BMC）刷 vbios 恢复 |
| 装 rock-6.3.30-V1.4.1a 驱动后系统 crash | 全系 | 驱动访问 module param `kfd_int_cpus` 触发 crash | 升级驱动版本，或设置/调整 `kfd_int_cpus` 参数规避（<internal-ticket>/12814） |
| 模型推理加载异常缓慢、cuda graph 卡住（整机供货场景） | BW150 + Kylin V11 OS（华三整机） | 未完全定位（内部 Kylin V11 环境未复现，指向整机+OS 组合适配问题） | 现场换 ubuntu22.04 可正常启动；华三整机出货引导客户用 Kylin V10 SP3 而非 V11；根治需整机厂商与 Kylin 适配验证（<internal-ticket>） |
| 掉卡，重装驱动无法解决 | 全系（物理机） | grub 启动参数 `pci=realloc` 导致 PCIe 资源重分配异常 | 启动参数中去掉 `pci=realloc`（改后需重启生效） |
| 驱动 v1.5.0 在 sosreport 收集系统信息时系统挂机 | 全系（驱动 v1.5.0） | v1.5.0 驱动与 sosreport 信息收集流程冲突 | 升级驱动到 1.5.0h 修复版本 |
| 机器定时重启（周期性自动重启） | BW100（整机） | BIOS 版本较低（疑似，当时更新验证中） | 更新 BIOS 版本 |
| K100 AI 虚拟化掉卡 | K100-AI（虚拟化） | 物理机缺内核头文件致驱动无法加载；最终定位为 vbios 问题 | 临时：虚拟机内重启驱动恢复；根治：刷 vbios（已提供临时验证版本） |
| 麒麟 V10SP3 开启内存加密（CSV/SEV）后 DCU 驱动加载报错 | K100 / K100-AI | 内核 cmdline `mem_encrypt=on` 与 DCU 驱动冲突 | cmdline 去掉 `mem_encrypt=on` 或改 `mem_encrypt=off` 可正常加载；但 CSV+DCU 为机密计算方案，需保留加密时须与驱动团队协调（<internal-ticket>） |
| 虚拟机直通 K100 启动后 hy-smi 报 `MKFD file descriptor not valid` / `No hycu Driver loaded` | K100-AI（虚拟化直通） | 驱动/vbios 版本或直通卡映射延迟，内核加载驱动失败 | 升级驱动/刷新同版本 vbios；虚拟机已启动但映射有延迟→等待或手动 `modprobe hygon`（<internal-ticket>） |
| 虚拟机直通运行 vllm0.11+ 丢卡（vllm0.9 正常） | K100-AI（虚拟化直通 + vllm） | 虚拟化直通与 vllm 高版本兼容问题（BIOS 需 SVM/IOMMU/ACS 开启） | 用 vllm0.9；或与驱动/虚拟化团队确认直通补丁（`hycu_pci_fixup_header`）适配（<internal-ticket>） |
| K100_AI 升级驱动后 `device_id` 由 6210 变 6211 | K100-AI | 驱动版本升级改变 did（非故障） | 勿按 did 硬编码判断型号；以 hy-smi/驱动版本为准（<internal-ticket>） |

---

## B. vLLM / SGLang 推理部署（最大头）

### B1. 启动崩溃 / 精度乱码
| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| 多模态/长上下文精度乱码 | K100-AI/BW150 | 默认注意力后端不兼容 | `--mm-encoder-attn-backend TRITON_ATTN --attention-backend TRITON_ATTN` |
| W8A8 路径异常 | K100-AI | 量化实现 | `W8A8_SUPPORT_METHODS=3` + `ROCBLAS_INT8_ENABLE=0` |
| FP8 启动报错 | K100-AI | 融合算子 | `VLLM_USE_FUSED_QA_KVA_GEMM=0` + `VLLM_USE_FUSED_DTBMM=0` |
| PP 分片报错 `len(partitions)!=pp_size` | K100-AI | 残留 PP 环境变量 | `unset VLLM_PP_LAYER_PARTITION` |
| Bert/bge-m3/XLMRoberta 启动 NotImplementedError | 全系 | V1 引擎不支持这些架构 | `VLLM_USE_V1=0` |
| vLLM 0.9.2 v1 精度异常 | K100-AI | V1 注意力 | `VLLM_USE_FLASH_ATTN_PA=0` |
| Gemma3 输出空 | K100-AI | dtype | 指定 `--dtype bf16` |
| 服务仅监听 127.0.0.1 | 全系 | 默认 host | `--host 0.0.0.0` |
| vLLM 启动崩溃，traceback 断在 `await run_server_worker(...)` 且无后续 `Error` 行 | 全系（海光 vllm_hcu 0.18.x + GLM-5.x） | 启动脚本 `--speculative_config` 写了已废弃的 `"method":"deepseek_mtp"`（0.18.x 已改名 `mtp`，海光插件下该旧名初始化即崩） | 改 `"method":"mtp"` + `"num_speculative_tokens":1`；token 数 2→1 无效（已验证），根因是方法名非层数。仍崩则整段删 `--speculative_config` 重启隔离 MTP vs 权重/fp8。详见 `glm-tool-call-debug` skill 的 MTP startup CRASH 段 |
| 容器 hy-smi `failed to check docker container` | 全系 | seccomp | 创建容器加 `--security-opt seccomp=unconfined` |
| 长上下文 VMFault（特定镜像） | K100-AI | 分段实现 | `VLLM_USE_PIECEWISE=1`（仅特定镜像） |
| SGLang 服务启动正常但**第一个请求 prefill 即崩** `RuntimeError: Paged KV cache block size must be 64`（traceback 断在 `vllm_mha_varlen_fwd`） | gfx936（BW1000 类，DAS 定制 flash-attn） | vendor prefill kernel 要求 **page 为 64 的倍数**（报错文案"must be 64"有误导，实测 64/128/256 通过、1/16/32 失败）；SGLang 在 DCU 上默认 `page_size=1` | 启动脚本显式加 `--page-size 64`（生产必需项，非优化项）。decode kernel 无此要求。详见 `dcu-sglang-source` §三·补 |
| 离线算子级 cos_sim=0.25（AWQ 换 aiter w4a16 kernel 后精度全错） | BW10 | GPTQ int32 打包列序 [0..7] 与 aiter `AWQ_REVERSE_ORDER=[0,4,1,5,2,6,3,7]` 解包顺序不一致 | 送入 aiter 前按 `AWQ_INV=[0,2,4,6,1,3,5,7]` 重排列；对称量化（uint4b8）无显式 zp 时造全 8 的 zp 张量（int32=0x88888888）。详见 `dcu-perf-cases` 案例 13 |
| `Half != BFloat16` 崩溃（aiter w4a16 kernel + bf16 模型，compile/cudagraph 下尤甚） | BW10 | aiter kernel 硬编码 fp16 输出 | 启动加 `--dtype float16` 全链路对齐 fp16（零 kernel 改动，精度无损） |
| `RuntimeError: For prefix decode, query and key must have the same dtype`（fp8 KV + flash） | BW10 | fp8 KV 时 K 是 e5m2、Q 是 fp16，flash prefix decode 要求 Q/K 同 dtype | aiter 侧把 Q cast 成 KV dtype 再传 + 附 unit descale（全 1，纯 bit-cast 数值无影响） |
| `'constexpr_type' object has no attribute 'is_ptr'`（fp8 KV prefill 崩） | BW10 | aiter triton 2D attention 在 fp8 KV prefill 有编译 bug（e4m3/e5m2 都崩） | `ATTN_FLASH_PREFILL=1` 让 prefill 走 flash——ROCM_AITER_UNIFIED_ATTN + fp8 KV 路径下 prefill 必须走 flash，**设 0 无退路** |
| `'_OpNamespace' 'aiter' object has no attribute`（切换 w4a16 patch 后启动崩） | BW10 | 旧 torch.compile 缓存被复用 | 清 `torch_compile_cache` + `torchinductor_root` 后重启 |
| TPOT 偏高 5~7ms / 吞吐掉 ~7%（decode 阶段，无报错） | BW10 | DCU 默认按负载动态调频，decode 低负载时降频 | 锁 sclk level 6（760MHz）：`rocm-smi -d $i --setperflevel manual && rocm-smi -d $i --setsclk 6`；容器内锁频可能静默失败（权限不足），用 `rocm-smi --showclocks` 确认（异常：level 0/300MHz 或 Perf 列 auto） |
| TP1 单卡 32GB 装 31B AWQ 模型 OOM | BW10 | repack 成 aiter 格式后原始 qweight+qzeros（~3.5GB/卡）双份占用 | repack 完立即 `replace_parameter(layer, w_q_name, None)` 释放原始权重 |
| 推理/训练 KERNEL VMFault（FA 引起，BW1000 多见） | BW1000/BW10 | FlashAttention 包 bug（<internal-ticket>/<internal-ticket>/<internal-ticket> 等，最终解法都是更新 fa/triton 包） | 优先从内网拉 flash-attention-cutlass **aicc-master-dev** 分支（master 已年久失修）+ cutlass_3.2.1 **feature/16x64-mmac-aicc** 分支自编译 whl 替换：cutlass 解压进 FA 的 csrc/，`export FLASH_ATTN_OPT=1 && python setup.py sdist bdist_wheel`，pip 替换 flash_attn。编译坑：无 aicc 命令→装 AICC（dtk_llvm.run）；`cannot find ROCm device library for gfx92a`→setup.py 去掉 gfx92a（324/989/1142 行）；编译近 1 小时放 tmux。BW10 场景另见 w4a16 config 坑：BK=64/NG=2 触发 VMFault，用 BK=32/NG=1 |
| SGLang 服务中 KERNEL VMFault，fault kernel `paged_attention_kernel`（hylog 特征 `Page not present` + `Address does not belong to a known buffer`，多卡连锁 fault），**EAGLE 投机解码开 + KV 池高水位（≥0.95）时触发**，bf16/fp8 KV 均中招 | gfx936（BW1000 类，DAS 定制 flash-attn 预编译 .so，无本地源码） | 预编译 flash-attn das fork 的 paged attention 在 EAGLE draft/verify 路径读到失效页表项越界（A/B 实证：同负载关 EAGLE 稳定、开 EAGLE 必崩；与请求长度无关，64 token 小请求也崩） | **根治=升级 flash-attn das 构建**（内网 PyPI `<internal_registry>` 的 release/nightly 索引查版本，nightly 常有修复版；换装前备份原 .so，换后用复现脚本验证再上生产；nightly 构建上生产前与海光确认是否进 release 线）。**短期缓解=关 EAGLE**（去掉全部 `--speculative-*` 参数 + `SGLANG_ENABLE_SPEC_V2=0`，A/B 实证同负载稳定；1M 长文本 prefill 占大头，损失有限）。**复现配方**（逼高水位）：64k 输入 × 64 并发 × 256 请求（`max_running_requests=48` 是上限，32k 输入只到 0.87 逼不出 0.99）；崩溃取证先 `grep "kernel name:"` 拿 mangled 符号 demangle（见 `dcu-vmfault-coredump`） |
| SGLang 启动 Qwen3-ASR 报 config 配置错误 | 全系（SGLang + Qwen3-ASR） | `sglang/srt/configs/qwen3_asr.py` 中 `AutoConfig.register` 重复注册报错 | 注册函数加 `exist_ok=True` 参数 |
| SGLang 启动 GLM-ASR 后报 kernel fault | 全系（SGLang + GLM-ASR） | block-size 默认值与 DCU kernel 不匹配 | 调整 `block-size` 为 64 |
| GLM-ASR 不同音频请求后报 `split_with_sizes expects split_sizes to sum exactly to N` | 全系（SGLang + GLM-ASR） | 音频 Encoder 输出长度非 4 的整数倍，SGLang 直接 `reshape(-1,5120)` 不整除 | 改 SGLang 源码：reshape 前先将音频序列截断至 4 的整数倍 |
| 启用 RMSNorm-Quant 融合 + HIP Graph custom all-reduce 后 logits 异常、输出乱码 | BW1000（SGLang） | RMS 量化融合与 HIP Graph custom all-reduce 两特性冲突 | 两者不可同时开启，需二选一或都关 |
| vLLM `-q slimquant_marlin` 输出重复乱码 | 全系（vLLM + Torch 2.10 + slimquant_marlin） | Torch 2.10 下 vLLM 错误地强制覆盖输入（inplace 行为异常） | 将 vLLM 源码中 `inplace=True` 改为判断条件；性能有小幅回落 |
| DeepSeek-V4-Flash-W8A8 复杂指令下精度问题（常规数据集正常） | BW150/BW1000/BW100（SGLang） | lightop 等 whl 包 bug | 使用修复镜像（已覆盖三卡型）；具体方法为替换 lightop 等 whl 包 + 修改 sglang 源码 |
| 双卡推理导致虚拟机宕机 | BW100（虚拟化环境） | custom-pcie-all-reduce 与虚拟化环境冲突 | 禁用 custom-pcie-all-reduce |
| Qwen3.5 PD 分离部署输出乱码 | 万卡集群（PD 分离） | PD 分离下投机解码参数未配置 | 添加 `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`（Qwen3.5 验证值，其他模型需自行验证） |
| DeepSeek-R1-AWQ 推理卡死 | BW100（vLLM + AWQ） | `-q` 参数误用 `awq_marlin`（该量化后端在 BW100 上导致卡死） | `-q` 配置为 `moe_wna16`（<internal-ticket>） |
| Qwen3-235B 使用 KV cache 量化后输出乱码 | 全系（vLLM + KV cache 量化 + fp16） | dtype 为 float16 时 KV cache 量化路径有 bug | `VLLM_USE_OPT_RESHAPE_AND_CACHE=0` 规避；或改用 bfloat16（性能有一定损失） |
| 双机 TP16 部署报 shape error | 全系（vLLM + marlin 量化 + TP>8） | vLLM 对 marlin 量化后端的 TP 上限为 8，TP16 超出限制 | 量化方法改用 `compressed-tensors`（支持更高 TP） |
| K100_AI 上 Qwen3-32B 输出异常乱码 | K100-AI（vLLM 0.9.2） | vLLM 0.9.2 带尾缀版本精度 bug | 使用 vllm-0.9.2 不带尾缀的版本（实测精度正常） |
| GLM5.2 4 机 PD 分离，上下文达 KV cache 临界值（~132K）时 P 节点崩溃（稳定复现，一周多次） | BW1000（多机 PD 分离） | 未定位（KV cache 临界值触发，<internal-ticket> 跟踪中） | 暂无规避；崩溃后需重启所有节点服务；跟踪 <internal-ticket> |
| GLM-5.3 输出乱码 | BW1100 | 未定位（优化中，待同步） | 暂无 |
| 长时间推理内存泄漏（持续增长） | BW150（DeepSeek-V4-0731） | 旧镜像内存管理问题（具体根因未展开） | 部署 deepseek-v4-0731 新镜像 + 修改参数，大幅减轻泄漏（具体镜像版本/参数待补） |
| Qwen3.8-27B 量化推理精度异常（仅 temperature=0 时正常） | K100-AI（SGLang 0.5.12 + DTK26.04 + 量化） | 未定位（研发分析中）；现象特征：仅非贪心解码时异常，指向采样路径而非 prefill 路径 | 临时规避：temperature 设 0（贪心解码）；根治待研发方案 |
| DS-V4-Flash-0731 FP8 精度测评时关键 kernel/算子报错（DS-Park 方案） | BW1100（DS-Park） | 部分关键 kernel/算子在 DCU 上未适配（具体 kernel 名待补） | 基于 TileLang 重写/修复关键 kernel 与算子（详见内部 DS-Park 部署实践文档） |
| 双机推理长输入输出乱码（长序列幻觉） | 全系（vLLM + 长输入） | `max-num-batched` 过大导致长序列幻觉 | ① 调小 `max-num-batched`（主）② 调低 temperature ③ 丰富提示词 |
| BW1100 上 vLLM 只能用 TP4，无法 TP8 | BW1100（vLLM 0.15-0401） | 镜像自带 lightop 库版本与 TP8 不匹配 | 单独替换 lightop 库后即可用 TP8 |
| Qwen3-Coder-480B-A35B FP8-Dynamic 精度异常/乱码 | BW1100 | FP8-Dynamic 量化在 aiter moe 路径精度坍塌 | 替换 triton/vllm/aiter 包；`VLLM_USE_AITER_MOE_FP8_W8A8=1` + `VLLM_USE_PIECEWISE=0` + `VLLM_USE_FUSED_FILL_RMS_CAT=0` + `VLLM_USE_FUSED_RMS_ROPE=0`（<internal-ticket>） |
| Qwen3-Coder-480B-A35B W8A8 精度/性能异常 | BW1100 | W8A8 需正确开关 aiter moe | `VLLM_USE_AITER_MOE_W8A8=1` + `VLLM_USE_PIECEWISE=0`；`--compilation-config fuse_act_quant=false`（<internal-ticket>） |
| GLM-5.x-Channel int4/fp8-w8a8 智能体调用工具乱码/工具调用失败 | BW1100 | tool-call/reasoning parser 与量化路径不匹配 | `--tool-call-parser glm47 --reasoning-parser glm45`；`VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0`（不能和 kvfp8 同开）；`-cc fuse_act_quant=false`（<internal-ticket>/13710） |
| GLM-5.2 INT8 4 机 TP32 报 marlin moe dimension unsupported | BW1000 | slimquant_marlin 量化后端 TP 上限限制（>16 卡维度不支持） | 降到 TP16（实测正常）或换量化后端；TP32 超出 marlin 支持维度（<internal-ticket>） |
| K100_AI sglang FP8 报 triton MLIR `triton.language has no attribute 'popcount'` | K100-AI | Triton 编译器在 K100 上 FP8 内核 MLIR 编译流水线异常（仅 FP8 触发，BF16/INT8 正常） | `SGLANG_DISABLE_TRITON_FP8=1`（关 triton fp8 内核）；或改 AWQ-INT8/BF16 权重；或换镜像（跟 DTK 版本有关）（<internal-ticket>） |
| sglang 高并发报 KERNEL VMFault（DeepGEMM/DeepEP/FlashInfer 均非根因） | BW1100 | CUDA Graph 开启状态与 MoE/A2A/缓冲区生命周期交互（交叉实验排除各 backend） | 加 `--disable-cuda-graph` 可稳定规避；定位重点在 CUDA Graph 而非 deepgepm/deepep（<internal-ticket>/DPS-289） |
| DeepSeek-V4-Flash-VE 长时间运行后崩溃（8 卡连锁 VMFault） | BW1000 | fault kernel=index_select（embedding 查表越界）；索引含 dspark 投机 token（1 真实+3 投机），投机解码注入非法 token id（-1 补齐） | 定位：fault kernel 是 aten index_select 而非 flash-attn；属投机解码 spec token 补齐 bug；避免「长上下文 + 投机解码」组合或升级修复版（<internal-ticket>） |
| Qwen3.5-35B-A3B 关闭思考模式后回答一堆字/答非所问 | K100-AI | 关闭思考模式后 reasoning/parse 路径不稳定 | 保留 `--reasoning-parser qwen3`；核查 `VLLM_MLA_DISABLE`/`VLLM_USE_FLASH_MLA` 设置；精度异常时考虑关 flash mla（<internal-ticket>） |
| sglang 推理 Qwen3.6 输出不稳定（时好时坏/时而返回空） | BW1000 | sglang 配置（EAGLE 投机+piecewise cuda graph）在该模型上不稳，vllm 稳定 | 换 vllm 或调整 sglang EAGLE/piecewise 参数；`SGLANG_ENABLE_SPEC_V2=1` 等（<internal-ticket>） |
| Qwen3.5-397B-W8A8 八卡报错不支持（四卡正常） | BW1100 | 八卡 TP 下 lightop/量化 env 组合不支持 | 用 4 卡，或调整融合算子 env（`USE_FUSED_RMS_QUANT`/`VLLM_USE_LIGHTOP_*` 组合）（<internal-ticket>） |
| Qwen3.5 长上下文拓展失败 `MRotaryEmbedding has no attribute 'truncate'` | K100-AI | 长上下文需 rope 覆写 + 放开 max len 限制 | `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` + `--hf-overrides` 配 rope yarn 参数 + `--disable-custom-all-reduce`（<internal-ticket>） |
| Qwen3.8-27B sglang fp8 kv 长文本崩溃 / 高并发崩溃 | BW1000 | fp8 kv + 长上下文 + EAGLE 投机并发路径不稳 | `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`；调整 kv-cache dtype/`--cuda-graph-max-bs`；高并发需关部分投机/降并发（<internal-ticket>/15145） |
| vllm bench 不指定 `--temperature 0.0` 多并发性能严重下降 | BW1100 | 默认非贪心采样，吞吐特性与贪心完全不同 | bench 命令显式加 `--temperature 0.0`（4 并发吞吐可翻倍）（<internal-ticket>） |
| vllm0.9.2 跑 deepseek w8a8 1P1D 走 roce 推理卡住 | BW1100 | 1P1D PD 分离下 P 侧 KV 发送失败/KeyError/pd_pair 已存在 | 核查 PD 分离配置、`NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` 一致性；P/D 启动命令对应；必要时 `register_pd_to_proxy` 后重启（<internal-ticket>） |
| sglang Qwen3.5-397B INT8 w8a8 部署报错 | BW1000 | cookbook 部署脚本 + sglang env 组合在 INT8 下不兼容 | 按 cookbook 但注意 `SGLANG_USE_FP8_W8A8_MOE=0`/`SGLANG_KV_LAYOUT_DCU_FA=1`；`--attention-backend fa2` 等（<internal-ticket>） |
| Qwen3-8B 加载 lora 权重报 get_punica_wrapper 错误 | K100-AI | vllm0.18.1 lora 路径 punica 兼容问题 | `export VLLM_USE_PUNICA=0` 无效则升级/换 vllm 版本（<internal-ticket>） |
| TRL PPO 训练与 HF PreTrainedModel 接口不兼容（`AttributeError 'tuple' has no attribute 'logits'` / `no attribute 'score'` / 参数未收梯度） | BW1000（torch2.7.1 + trl1.10.0.dev0，Qwen3-0.6B 全参 8 卡） | trl experimental ppo 的 AutoModelForCausalLMWithValueHead 与 HF 接口/参数空梯度不匹配 | 核查 trl 版本兼容；PPO 全参训练报 AttributeError logits/score 及 DDP 规约「参数未收梯度」，属 trl 与 HF 接口适配问题（<internal-ticket>） |
| deepseek-v4-0731 int4 模型降智（生成字数波动大，扩写/缩写偏差 2000+） | BW150（sglang） | sglang 部署 int4 精度/长度控制差；同类题 minimax 无此问题 | 避免过分精细提示词；跑业务场景验证；关注量化与采样路径精度（DPS-370） |
| SGLang 多卡（TP）启动即崩 `RuntimeError: The memory capacity is unbalanced. Some GPUs may be occupied by other processes. pre_model_load_memory=X, local_gpu_memory=Y, ... * 0.9=Z` | 共享机（多卡 TP 部署） | 启动时各 rank 空闲显存不均：某张目标卡仍有残留/别的进程占显存（刚杀的上一个服务未释放完，或共享机上别的容器正占卡）——sglang 的 pre-model-load 均衡检查拒绝启动。非软件 bug，是启动时机/资源问题 | ① 杀完上一个服务后**等所有目标卡显存回落 ~0%** 再重拉（DCU VRAM 异步释放，hy-smi 会短暂停在高位，别急着重拉）② 共享机先查目标卡有无**别人的新容器/进程**占卡（`docker ps` + `hy-smi`），别误杀别人的 ③ 确认各目标卡都 ~0% 空且均衡后再重拉 |

### B2. 高并发 / 超时 / toolcall
| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| 高并发崩溃 / TimeoutError | K100-AI/BW150 | RPC/迭代超时过短 | `VLLM_RPC_TIMEOUT=100000` `VLLM_ENGINE_ITERATION_TIMEOUT_S=600` `--speculative-disable-by-batch-size 33` |
| toolcall 失效 | K100-AI | 解析器/模板 | 启动加 `--tool-call-parser deepseekv32` 并去 jinja；推理 toolcall 失败加 `--chat-template-content-format=string` |
| 思考模式关不掉 | K100-AI | 缺 reasoning parser | `--reasoning-parser qwen3` 或改 chat_template.jinja |
| Ray DAG 超时（>300s） | K100-AI | ray timeout 太小 | `RAY_CGRAPH_get_timeout=3600` 等 |
| 64K 长上下文推理 fail（超时） | 超节点集群 | router 请求超时阈值过短，长 prefill 未完成即被掐断 | 调大 `router-request-timeout-secs` |
| DeepSeek 模型无思考模式（不输出 thinking 内容） | K100-AI | 框架层思考模式默认关闭，需显式设置方可生效 | 在框架层显式开启思考模式开关（具体参数随框架/版本而定，先查所用框架的 reasoning/thinking 相关配置） |
| 双机大并发崩溃 | BW1100（SGLang 双机） | APC 与 MTP 同时开启冲突 | APC 与 MTP 不可同时开启 |
| 大上下文 + 大并发下模型崩溃 | BW1100（SGLang） | MTP 在大上下文大并发下不稳定 | 大上下文大并发场景关闭 MTP |
| PD 分离压测 D 节点宕机 | BW1000（多机 PD 分离） | 对外服务网卡误指定为系统管理网卡（带宽/能力不足） | 改指定 200G 业务网卡 |
| 日志反复刷 `Received output for rid=... but the state was deleted in TokenizerManager` | BW1100（SGLang；上游有同签名 issue） | 僵尸请求：客户端断连/LB 超时/abort 后，前端 TokenizerManager 侧状态已删，但 abort 信号未传到调度器，调度器继续为已删 rid 逐 token 生成，每个输出回来刷一条日志；MTP 投机解码路径易出 race（abort 落在 draft/verify 中间丢停止信号） | 判据：正常断连只刷几条残余；同一 rid 持续刷上百条 = 僵尸请求，占并发槽+KV 显存（高负载下是实打实容量损失）。处置：① LB/网关 access log 查对应时间窗口断连记录，LB read timeout 短于长回答生成时间则调大 ② A/B：重启实例去掉 MTP 投机配置（并确认 APC 未与 MTP 同开），同负载复测看是否消失 ③ 仍复现 = abort 路径 bug，升级最新 fork 复核或提框架团队；`/get_load` 或 scheduler 日志确认僵尸 rid 是否仍占位，不释放则重启服务 |

### B3. 文生图 / 多模态大模型
| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| GLM-Image / Qwen-Image 推理异常 | BW150 | AR 阶段占比大、aiter 接口不稳 | 建议走 vLLM-Omni；sglang 的 aiter flash_attn 需 patch shim（`patch_aiter_flash_shim.py`） |
| sglang 文生图无法走 aiter 接口 | BW150 | aiter 未导出兼容接口 | Patch `aiter.__init__` 导出 `flash_attn_func` 兼容 SGLang |

### B4. 第三方模型 / 多模态 / 语音模型适配
| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| Bagel 多模态模型报 `No module named torch.nn.attention.flex_attention` | K100-AI | Bagel 需 torch≥2.5 | 装 torch-2.5.1+das.opt1.dtk25042 wheel（光源 DAS1.7 直链）（<internal-ticket>） |
| Kimi-Audio 编译 CUDA 组件失败（需 nvcc 编译 cuda 代码） | K100-AI | 内含需 nvcc 编译的 CUDA 代码 | `source /opt/dtk/cuda/env.sh`；或用 FastPT：`source /usr/local/bin/fastpt -C` 实现 CUDA 源码不转码编译（<internal-ticket>） |
| LiveTalking freqencoder/gridencoder 组件编译失败 | K100-AI | CUDA 宏未转 HIP | gridencoder 的 setup.py 把 `__CUDA_NO_HALF_OPERATORS__` 改 `__HIP_NO_HALF_OPERATORS__` 后 `python setup.py install`（<internal-ticket>） |
| ClearerVoice-Studio 报 nvidia-smi 错误 | K100-AI | 官方代码用 subprocess 调 nvidia-smi 取 GPU ID | 改 clearvoice/networks.py，去掉 nvidia-smi 相关调用（<internal-ticket>） |
| Qwen2.5-Omni 多模态推理异常 | K100-AI | web_demo.py 默认只用第 1 张卡 | 修改 web_demo.py 的 device_map 为 'auto' 指定 DCU 设备（<internal-ticket>） |
| marker/surya 框架安装报错 | K100-AI | pyproject 限定了 torch 版本与 DCU 镜像冲突 | 注释掉 surya/marker 的 pyproject.toml 中 torch 和 surya-ocr 版本要求后 `pip install -e .`（<internal-ticket>） |
| whisperx 适配失败 | K100-AI | requirements 限版本 + ctranslate2 需 DCU 版 | 注释掉 requirements 里 torch/torchaudio/faster-whisper 版本；装 DCU 版 ctranslate2 wheel；`numpy=1.24.3`；`CT2_CUDA_ALLOW_FP16=1`（<internal-ticket>） |
| MongkeyOCR 识别 >5 页 PDF 报 KERNEL VMFault | K100-AI | 单卡处理时间太长报错 | 改 magic_pdf/model/custom_model.py 默认单卡调度为多卡（4 卡处理 17 页很快）（<internal-ticket>） |
| Rasa 聊天框架 tf.function 数据格式不匹配报错 | K100-AI | tensorflow2.13.1 的 input_signature 校验过严（NV 上也报） | 改 Rasa 源码 tf.function 的 input_signature 校验；或降回 tensorflow 2.12（<internal-ticket>） |

---

## C. RCCL / NCCL 集合通信

| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| 双机 allreduce 带宽低 / 卡死 | 全系 | topo 文件 busid 不匹配 | `export NCCL_TOPO_FILE=null` 自动探测 |
| 多机通信走内核态、性能差 | K100-AI | IB 网卡降速未用 GDR | 更换 IB 网卡；老驱动暂关 `NCCL_NET_GDR_READ=1` 并升级驱动 |
| 重装驱动后 allreduce 仅 30+ | K100-AI | 驱动状态未生效 | 重启物理机 |
| 双机 ACS 导致通信异常 | 主要为 PCIe 卡（K100-AI/BW/BW10/BW100/BW150 等；**OAM 产品 BW1000/BW1100 基本无影响**） | ACS 未关 | PCIe 卡：BIOS 关 ACS；`NCCL_IB_GID_INDEX` unset；`NCCL_IB_TC=160或162`。OAM 机型先排查其他根因 |
| 默认通道数上限导致多卡慢 | BW150/BW1000/多节点BW1100 | `NCCL_SIMPLE_CHANNELS` 默认 16 | `export NCCL_SIMPLE_CHANNELS=32` |
| `NCCL_RINGS` 配置后通信失败 | 全系 | RINGS 只能从 0 编号 GPU 序列 | `HIP_VISIBLE_DEVICES` 不能跳过编号（如 0,1,2,3 而非 0,2,4,6） |
| dcu_megatron 默认 NCCL 不适配本机 | K100-AI/BW | 框架默认是 BW1000 配置 | 改 `dcu_megatron/requirements/env.sh` 按实际机器配 NCCL |
| 创建 ray 集群前 GPU 不可用 | BW | 容器未设可见设备 | 每容器先 `export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` |
| 双机 SSH 免密/网口 IP 冲突 | K100-AI | 主机名绑定错 IP | 修正 `/etc/hostname` 与 hosts；确保双机互信 + 网口 IP 不冲突 |
| `NET/IB : No device found` + `Using network Socket`（IB 可见但走 Socket） | BW1000（曙光 SHCA IB） | SHCA 需官方 RCCL 网络插件，通用 verbs 桥接不兼容 | `NCCL_NET_PLUGIN=.../librccl-net-shca.so` + 插件目录进 `LD_LIBRARY_PATH`；`NCCL_IB_HCA=shca_0,shca_1,...` **不带 `:端口` 后缀**（该版本 RCCL 不接受） |
| `modify_qp` EINVAL / 偶发挂死 / LID 截断（如 111896→46360） | BW1000（曙光 SHCA IB） | 系统内多个 libibverbs（torch/deep_ep 自带）与 SHCA 驱动 ABI 冲突 | 弃用 LD_PRELOAD 桥接库（只能缓解枚举、数据面仍卡死），改用 SHCA 官方插件 `librccl-net-shca.so`（直调原生 verbs API） |
| `ibv_reg_mr` EINVAL + DeregMr 崩溃 | BW1000（曙光 SHCA IB） | SHCA 不支持 PCIe Relaxed Ordering | `NCCL_IB_PCI_RELAXED_ORDERING=0` + `NCCL_IB_ECE_ENABLE=0` |
| `Message truncated: received 8192 instead of 2048` | 多机（SHCA IB 实测） | 两端 p2pnChannels 自动协商不一致（如一端 8 一端 32） | 两端显式锁死 `NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=16` + `NCCL_NCHANNELS_PER_NET_PEER=4` |
| `Nic dev attr no exit ... cannot be found` | 多机（Ray 后端） | vLLM 把头节点 `NCCL_TOPO_FILE` 广播到 worker，各节点设备 GUID 不同 | 每节点独立生成 topo（`hy-smi --showtopo` 取实际 GUID + 修正 XML vendor 字段）；Ray 启动脚本按 `ip -4 addr show ib0` 的 IP 区分节点动态设置；或改 vLLM `ray_env.py` 的 `non_carry_over_env_vars` 把 `NCCL_TOPO_FILE` 加入排除项 |
| 节点身份误判（用错 topo/IP） | 多机（Ray 后端） | `hostname -I` 最先输出 eth0 IP，条件判断错 | 用 `ip -4 addr show ib0` 取地址判定节点身份；`VLLM_HOST_IP` 设为 ib0 IP |
| 模型加载中途 ActorDied / SIGTERM 疑 OOM（假性） | 多机（Ray 后端） | 外部 `ray stop`/`ray start` 或残留 placement group 占 GPU | 先 `ray stop` 清理所有 Ray 进程再重启集群，区分真 OOM 与运维干扰 |
| GDR 未生效 / 数据面不稳（SHCA 插件） | BW1000（曙光 SHCA IB） | GDR 依赖 DMABUF；`NCCL_NET_GDR_READ` 因插件实现而异 | `NCCL_DMABUF_ENABLE=1` + `unset NCCL_NET_GDR_LEVEL` + `NCCL_NET_GDR_READ=0`（SHCA 插件实测置 0 更稳定）；验证终态看日志 `via NET/IBext_v8/.../GDRDMA` |
| `ibv_reg_mr_iova2 failed with error Invalid argument` + `Test NCCL failure common.cu.cpp:1416 'unhandled system error'`（次生：`Connection reset by peer` / 多进程 segfault） | BW1000（gfx936）双机 16 卡 RoCE（mlx5），DTK 6.3.3 / RCCL 2.22.3-HEAD / 内核 5.15.131-29.cl9 | GDR peer-memory 注册失败：dmesg 反复刷 `failing symbol_get of non-GPLONLY symbol ib_register_peer_memory_client`（hycu 为非 GPL 模块，内核该符号 GPL-only 导出），网卡注册不了 DCU 显存 | 绕过：`-x NCCL_NET_GDR_LEVEL=LOC` 关 GDR 退回 host-staging（能跑，busbw 被 PCIe 拷贝顶住、远低于 GDR 基线，属预期回归）；根治需海光出能成功注册 peer-memory 的驱动（报 dmesg 证据 + 驱动/DTK 版本）。诊断：`dmesg \| grep -iE "peer\|gdr\|dma_buf\|reg_mr"` 看 symbol_get 是否失败；`for d in /sys/class/infiniband/mlx5_*; do echo -n "$(basename $d): "; cat $d/ports/1/rate; done` 顺带查有无混速网卡（200G/100G/25G 混插，GDR 修好后慢卡会成瓶颈） |
| `NCCL_DMABUF_ENABLE=1` 后所有进程启动即 segfault（`Could not open kernel conf file` + 崩在 fgets） | RCCL 2.22.3-HEAD 构建（DTK 6.3.3，gfx936） | 该 RCCL 构建 dma-buf 路径自身 bug（读 kernel conf 文件时空指针） | 勿用 `NCCL_DMABUF_ENABLE=1`；注意去掉 `NCCL_NET_GDR_LEVEL`/`NCCL_NET_GDR_READ` 无效（该构建默认走 GDR，env 只是调参不是开关）；验证 GDR 已关看 channel 日志 `via NET/IB/x`（无 `/GDRDMA` 后缀） |
| 自研/第三方网卡与 DCU 无法通过 GDR 通信 | 全系（自研网卡 + DCU GDR） | 内核未开启 `CONFIG_PCI_P2PDMA` 和 `CONFIG_DMABUF_MOVE_NOTIFY` | 重新编译系统内核并开启这两个配置项（需重编译内核，高危，先测试机验证） |
| IB 网络无法使用：多播失败、网口无法 UP | 全系（IB 网络） | 子网管理器（opensm）版本来源不对：yum install 安装的 opensm 与 IB 环境不兼容 | 必须使用 mlnx 驱动包安装的 opensm，不要用 yum install 的版本 |
| rccl 测试流量不走指定的无损队列（配置的队列 5） | BW1000/BW150 | 容器内 rccl 流量默认走队列 0，未走交换机配置的无损队列 | 配置队列映射 / `NCCL_*` 相关；确认容器内 rccl 用到的 IB 队列与交换机无损队列一致（<internal-ticket>/13266） |
| BW150 四卡桥接 Ring AllReduce busbw 仅 91GB/s 偏低 | BW150 | 四卡桥接形态 + DTK26 之前 8 卡 AllReduce 走 Ring/Tree 都得降级 graph 带宽很低 | 用 DTK26（DTK26 前该问题存在）；大数据量 busbw 91~92GB/s 需查桥接拓扑；`NCCL_BUFFSIZE`/`NCCL_ALGO=Ring`（<internal-ticket>/12212） |

### C2. K8S / 云原生部署

| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| K8S 环境下推理性能远低于裸机/docker（差数倍） | 全系（K8S 部署） | 创建 pod 时缺少 env 里的关键变量（DTK 路径/NCCL 配置等未注入） | 排查法：先跳过 K8S 在 docker 上部署验证性能正常，再逐项检查 pod env 与 docker 环境的差异 |
| K8S 下 DCU 频繁出现 num=0 且不上报 error | 全系（K8S + dcu-plug-in） | dcu-plug-in 版本与 DTK 适配问题（注意：该插件更新只改日期不改版本号，按版本号判断会误判为"已是最新"） | 更新**最新日期**的 dcu-plug-in（看日期不看版本号） |
| K8S 分配 2 卡做 TP2 出现 nccl error | 全系（K8S + dcu-plug-in + DTK） | dcu-plug-in 与 DTK 适配问题（当时待 DTK26.04 发版根治） | 临时：分配 4 张 DCU 卡、用 TP2 方式推理（多分 2 卡规避） |
| K8s 单机同时部署 dcu 和 vdcu 插件冲突 | K100-AI（K8s） | 两插件同机资源调度冲突 | 单机只部署一种插件，或分机部署 dcu 直通与 vdcu 切分（<internal-ticket>） |
| K8s 无 dcu-exporter 时需获取 DCU 实时参数 | K100-AI（K8s） | 无 k8s 环境取 DCU 实时监控 | 按 k8s-dcu 插件样例配置转成 docker 命令单独启 dcu-exporter 容器（`--device=/dev/kfd --device=/dev/mkfd --device=/dev/dri -v /opt/hyhal:ro` 等）（<internal-ticket>） |
| BW150 K8s 单机多卡 NCCL 初始化失败（单卡正常） | BW150（K8s + vllm） | k8s 分配后 /dev/dri 映射错位（card/renderD 对应关系不一致，如 card6→renderD133 而非 134） | 通过 `HIP_VISIBLE_DEVICES` 指定卡运行或直接 docker 指定 device 运行可正常；排查 k8s device-plugin 的 /dev/dri 映射（<internal-ticket>） |

---

## D. 库缺失 / 编译 / 框架版本

| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| `libamdsmi` 缺失 | 全系 | 自动下载版不兼容 | 装特定 amdsmi whl（非 DAS 自动下载版） |
| `libtorch_cuda.so` / `libcudart` / `librocm_smi64` 缺失 | 全系 | LD_LIBRARY_PATH 未含框架镜像自带路径 | 补 `LD_LIBRARY_PATH` 指向镜像内对应 lib |
| hip 编译报链接 CUDA 目标失败 | 全系 | CMakeLists 未链接 DTK | `find_package()` + `source /opt/dtk/cuda/env.sh` |
| nvcc `All arch params replaced by gfx906,gfx926,gfx928` | 全系 | CMakeLists 未加 DTK 链接 | CMakeLists 加 DTK 路径或 source cuda/env.sh |
| numpy 2.x 与 torch 不匹配 | K100-AI/BW | 镜像 numpy 2.x vs torch 2.5 | 降 `numpy==1.25.0` 或 `1.26.4` |
| transformers 版本不兼容（缺 padding_side / check_model_inputs 误判） | 全系 | 版本与模型要求不符 | 按模型定：4.44.2 / 4.57.1 / 5.3.0 等 |
| 7490 CPU 不支持 AVX512 | 全系 | 产品侧 deepspeed 按 AVX512 编译 | 暂不支持 adam offload 到内存；用改编译版本（咨询深度组） |
| 编译 hip 找不到 torch.h | 全系 | include 路径未加 | Makefile 加 `/usr/local/lib/python3.10/dist-packages/torch/include/...` |
| `bc` 缺失导致脚本失败 | 全系 | 基础工具未装 | `apt/yum install bc` |
| `PYTHONPATH` 污染导致导入错包 | 全系 | 环境变量串味 | `unset PYTHONPATH` 后重跑 |
| 紫光/新支点 OS 限制脚本执行 | 全系 | OS 安全策略 | `bash xxx.sh` + 改用用户目录 tmp |
| AISBench 压测请求速率突发（burstiness=0 未生效） | 全系（压测工具） | 老版本 AISBench 的 burstiness=0 参数未生效 | 更新 AISBench 到最新版本 |
| ORT 部署模型加载正常，推理特定 size 数据报 vmfault | 全系（ORT 框架） | LayerNorm 算子融合产生 nullptr 空指针 | 关闭算子融合绕过；或以此为排查思路 |
| vLLM Bench 压测吞吐远低于 EvalScope（同模型同参数差数倍） | 全系（vLLM Bench 工具） | vLLM Bench 的 `temperature` 参数不再默认 0（随机采样 vs 贪心解码，吞吐特性完全不同） | 手工添加 `temperature=0` 对齐 EvalScope 行为 |
| LLaMA-Factory zero3 训练 OOM（显存碎片无法回收，训练场景） | 全系（DTK26） | DTK26 不支持 `expandable_segments` 特性，显存碎片无法被充分使用 | CPU offload ADAM 优化器 + step 前和 FWD/BWD 中间插入 `empty_cache()` 动态回收碎片；根治等 DTK26.04.2 支持 expandable_segments |
| ms-swift 多模态微调 OOM（训练场景） | 全系（ms-swift 多模态微调） | 样本序列长度不一，attention buffer 随最长样本逐渐增大后进入平台期（显存按最大序列长度分配；机制分析，该 case 当时复现中） | 降低 `max_length`（截断超长样本）复测 |
| BW1000 上缺 bitsandbytes 预编译包（pip 装不上/不兼容） | BW1000（DTK） | 官方源无 BW1000/DTK 适配的 bitsandbytes wheel | 使用 DTK 原生编译的 bitsandbytes-0.50.2-cp310 wheel（内部已编译，可复用） |
| hipcc 报 `cannot find -lhsa-runtime64` | K100-AI | CMakeLists 未链接 hsa-runtime64 | CMakeLists 加 `link_directories(/opt/dtk-25.04/lib /opt/dtk-25.04/hsa/lib)` + link hsa-runtime64（<internal-ticket>） |
| hipcc 编译 CUDA 报 `cannot be used when making a PIE project` | K100-AI | PIE 编译约束 | 编译加 `-fPIE` 或 `-fPIC`（<internal-ticket>） |
| rocminfo 报 `GLIBC_2.29 not found`（Astra Linux） | K100-AI | libgalaxyhip.so.5 需 GLIBC_2.29，直接升级会搞崩系统 | 装 additional-base-lib 包，`ablrun rocminfo` 解决；勿直接升 GLIBC（<internal-ticket>） |
| import torch 报 libibverbs/libgobject/libffi 缺失 | K100-AI | 缺系统库且 libffi 版本符号冲突 | 装 libibverbs/libgtk2.0；`LD_PRELOAD=/usr/lib64/libffi.so.7`；或 symlink 修正 libffi.so.7（<internal-ticket>/8935） |
| OpenEuler rocminfo 报 `Open kfd failed at Load` | K100-AI | 缺 ncurses 依赖 | `yum install ncurses ncurses-compat-libs` 后重启（<internal-ticket>） |
| rocky9 import torch 报 libmkl_core.so permission denied | K100-AI | SELinux 未禁用 | 编辑 /etc/selinux/config 把 SELINUX=enforcing 改 disabled 后 reboot（<internal-ticket>） |
| centos 装驱动报 `cmake path required is /usr/bin` | K100-AI | 缺 cmake/gcc | 安装 cmake 及 gcc 满足驱动编译要求（gcc≥7.3）（<internal-ticket>/9318） |
| pip install -e . 报 `error: incomplete-download`（setuptools） | K100-AI | `--use-pep517` 默认构建临时隔离环境，不复用本地包而重新下载 | `pip install -e . --no-build-isolation` 或 `--no-use-pep517`（<internal-ticket>） |
| DTK26.04 编译 PyFR 报找不到 hipGetDevicePropertiesR0600 | BW1000/BW150/BW1100 | libgalaxyhip.so.5 缺该符号 | DTK26 HIP runtime 符号缺失；联系 DTK 团队或用匹配版本（<internal-ticket>） |
| nvImageCodec 编译报 `math.h file not found` | BW1000（BW150） | nvImageCodec 在 BW1000 编译缺头文件路径 | 按随驱动发布代码编译时补 math.h 相关 include；确认 `ln -s /opt/dtk-26.04 /opt/dtk` + source dtk/cuda/env.sh 后 make（<internal-ticket>） |

### D2. PaddlePaddle 框架（2026-09-07 K100-AI 实测）

| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| `import paddle` SIGILL (exit 132)，dmesg `trap invalid opcode ... in libphi_core.so` | K100-AI（Zen1 CPU，无 AVX-512） | dtk2604 早期构建（260410）libphi_core.so 含 AVX-512 指令，Zen1 CPU 不支持 | 用 260417 或更新构建（官方测试报告源 `http://<internal_paddle_report_host>:18000/customized/paddle/dtk-26.04/`（内网 IP 已脱敏））；dtk25042 wheel 无此问题（未启用 AVX-512） |
| PP-OCRv5 完整 pipeline（det+rec 同进程）KERNEL VMFault，fault kernel `ConcatTensorWithDifferentShape` | K100-AI (gfx936) | dtk25042 paddle wheel concat 算子在 gfx936 有 bug（det/rec 单独正常、CPU 正常、matmul 正常 → 确认是 DCU kernel 问题） | 升级 dtk2604.260417 wheel + DTK 26.04 pytorch 镜像做底座（`pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`，光源无 26.04 paddle 镜像）；model.properties 标注 BW1000，K100-AI 非官方适配卡型 |

> **PaddlePaddle 在 DCU 上的版本选择**：dtk25042 是"concat 有 bug"的版本；dtk2604.260410 有 AVX-512 问题（Zen1 CPU 不可用）；**dtk2604.260417 是首个全通版本**（import/matmul/PP-OCRv5 完整 pipeline 均正常）。选版本时先看 CPU 是否支持 AVX-512（`grep avx512 /proc/cpuinfo`），Zen1（C86 7380 等）不支持，必须用 260417+。
> **主机端跑 CPU 认知（关键）**：Paddle 是"主机 + 设备"框架——Python 进程和主机端运行时（`libphi_core.so` 等）跑在 **CPU**，计算 kernel 跑在 **DCU**。即使 100% 在 DCU 推理，CPU 也是宿主，所以 **wheel 的 CPU 指令集兼容性是硬前提**（dtk2604.260410 死在 CPU 层，根本没轮到 DCU）。上机先 `grep avx512 /proc/cpuinfo` 确认支撑。
> **诊断隔离法**：DCU 上框架崩溃时，先跑 ① CPU 完整 pipeline（确认模型/代码正确）② DCU 简单 matmul（确认 DCU 基础通路）③ det/rec 单独（定位哪个子模型）——三步隔离可快速区分"模型问题 / DCU 通路问题 / 特定 kernel 问题"。
> **镜像打包坑**：`docker export` + `gzip -9` 高负载可能打挂磁盘 95% 满的设备（实测打包时失联 15 分钟+）；且 `docker export` 不导出挂载卷。打包方法见 `dcu-docker` 的「DCU 镜像打包/导出与离线部署」章节，此处不重复。

### D3. MIGraphX
| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| migraphx-driver fp32→fp16 报 `static assert MIGRAPHX NGLOBAL>0` | K100-AI | migraphx4.5.0 老版本 bug | 用新版镜像 stablediffusion-migraphx-ubuntu20.04-dtk24.04.3-py310 解决，加载时间也缩短一半（<internal-ticket>） |
| migraphx onnx/mxr 推理结果异常（含不支持的算子） | K100-AI | 模型含 migraphx 不支持的算子 | netron 解析模型 → `onnx.utils.extract_model` 提子模型 → CPU(onnxruntime) 与 DCU(migraphx) 对比定位异常算子 → 联系框架修复（<internal-ticket>） |
| migraphx 保存 .mxr 后 load 推理 VMFault（特定 seq_len 如 256） | K100-AI | 特定 seq_len 下保存/加载编译模型崩溃 | 用 gdb/复现脚本（reproduce_crash_seq256.py）隔离；seq_len=256 触发，需排查 map_input_dims 与编译缓存（<internal-ticket>） |

---

## E. 已知硬件 RMA 提示（非技术可避，走换卡/报修）
以下现象经验证为硬件故障，**不要花时间调软件**：
- dmesg 报 `GCEA err detected at instance`（UE 错误）→ 卡硬件损坏，换卡
- ubb 未正常供电导致 OAM 整体不被识别 → 查供电/换模组
- PCIe 卡槽故障 / 物理机 lspci 找不到显卡 → 换卡槽/报修
- 锁频、电源 VR 固件老、过流丢卡（128cu 瞬时电流高）→ 产线改 hyqual_normal 或换卡
- 旧库存 BM 型号板卡 i2c 上拉电阻不符 → 厂商改硬件或返厂

---

## F. 驱动健康 dmesg 规则引擎（dcu_env_check 移植，drvdiag 逻辑）

> 以下规则来自 dcu_env_check 工具的 driver_health checker（移植自 drvdiag -s dmesg 逻辑）。
> 每条格式：**错误模式关键字 → 严重级别 → 修复建议**。这是一种**正向规则引擎**，
> 与本文「现象反查」互补。实际排障时先 grep 关键字（源码匹配用 `(?i)`，大小写不敏感），命中则按建议处理。

| 错误模式关键字 | 级别 | 修复建议 |
|---|---|---|
| `PCIe link lost` | FAIL | 1.检查PCIe链路的FOM值、卡槽是否插好、供电是否正常 2.是否有nbio ras 3.检查下行端口是否有错误报告 4.报告问题 |
| `GPU FW Trigger Critical Temperature Fault` | FAIL | HCU内部过温，触发踢卡。检查散热相关 |
| `The cp might be in an unrecoverable state due to an unsuccessful queues preemption` / `HIQ MQD's queue_doorbell_id0 is not 0` | WARN | 错误可以自动恢复。如运行程序每次都出该故障：可逐一排查卡故障、软硬环境、程序。排查后报告问题 |
| `DQM create queue type` | PASS | 不是错误，不用关注 |
| `waiting for vbios ready timeout` | FAIL | vbios初始化失败，可以尝试（掉电）重启。早期版本驱动可以尝试重新加载驱动。最后做坏卡确认 |
| `p2p access=[0-1]* real vram szie` / `p2p_access=[0-1]* p2p_addressable=0*` / `Can't accessible` / `Failed to map the VRAM for peer device access` | WARN | 该日志会导致p2p访问异常。确认设备的pcie地址空间是否超过44bit。ACS的状态 |
| `no-retry page fault` | FAIL | 空/野指针。该问题首先排查软件其次排查硬件 |
| `UMC instance .* channel 0 CE .* UE [1-9]{1,}[0-9]*` | WARN | UMC不可纠错误，异常后驱动自动修复该错误。数据太过异常时，请排查硬件是否损坏、散热、供电 |
| `GFX SubBlock .*, Instance[.*][.*], DED [1-9]{1,}[0-9]*` | WARN | GFX不可纠错误.检查散热、供电，必要时定频(高、低频率)后运行程序以及坏卡检查 |
| `HDP DED [1-9]{1,}[0-9]*` | WARN | HDP不可纠错误，确认是否设备损坏 |
| `wait for kiq fence error` | WARN | 7+8环境下存在硬件bug，目前debug中 |
| `read BIOS error` | WARN | 更新驱动包到最新。如果是7+8的拓扑则不要尝试手动加载驱动 |
| `EEPROM NOT FOUND` | WARN | 找不到eeprom。确认设备是否支持eeprom。确认卡是否异常。尝试掉电重启。即使没有eeprom大多数情况下不影响设备的使用 |
| `lspci` 显示 `rev ff`（BDF 配置空间全 0xff，卡已从总线掉落） | FAIL | 官方部署文档明确：reboot 可能触发掉卡，**必须硬重启（DC 供电）或掉电重启（AC）恢复**，warm reboot 无效；恢复后核对驱动/VBIOS 版本匹配（手动刷 VBIOS 见 dcu-driver §3.4 hyflash）。来源：官方 wizard（2026-10-14 摘录，未实测） |

**通用建议（drvdiag normalWay）**：使用控制单一变量方法定位问题引入的路径。
必要时进行坏卡确认，一般方法：1、交叉卡槽/主板/cpu 2、检查供电路径 3、问题是否固定某卡 4、跑hyburn/hyfieldiag检查。

> ⚠️ 本 F 节规则来自源码工具，是"正向索引"；与各报错现象反查条目互补。
> 命中规则但处理无效时，回到 A/B/C/D 节按现象反查继续排查。

---

## 关联
- 操作类手册见：`dcu-driver`（驱动安装）、`dcu-docker`（容器/库）、`dcu-rccl-test`（集合通信测试）、`dcu-dtk`（DTK 编译）。
- 本手册是**纯排查索引**，与上面操作手册互补，不重复维护操作步骤。
- `dcu-driver` / `dcu-docker` / `dcu-rccl-test` 正文中已加"遇到具体报错先查本 skill"的提示，**只指路、不抄 env 内容**（避免双源维护）。
- **XID/SXID 错误码正向速查**（错误码→含义/原因/处理，与本文「现象反查」互补）已分离到 `references/xid_sxid_errors.md`，单独存放便于针对性查询，不混入主文。
- **关联诊断规则引擎**（dcu_env_check 源码提炼，多检查项同时失败/警告→指向方向）已分离到 `references/diagnosis_rules.md`，供多条件关联排查参考。
- **本包维护细则（存量 enrich / references 分离规范、不动根计数与版本）**见 `references/pack-enrich-maintenance.md`。
- **算子库源码级 bug 导致的静默卡死 / kernel 越界崩溃**（如 lightop `moe_marlin_w4a16` 在 BW100 静默卡死、FlashAttention `paged_attention_kernel` 超长上下文越界），根因多为插件库 hardcode 的 CU/gridx 假设与真实卡不符，其**分析流程 + 修复手法**在 `dcu-op-optimization` 实战案例库（案例一/二），本手册不重复（遵循"只指路不抄"）。

## 如何维护 / 从 ontrack 工单增量更新本手册

> 本手册源自 Hygon ontrack 系统导出的工单 CSV。增量更新时严格按以下流程，避免把噪声搬进来。

### 增量更新工作流（强制）
1. **全量覆盖，不抽样**：两个 CSV 都要逐行读完（用户明确要求"每个字都要涉及到，不要只读前500行"）。用 `\|CSD-\d+\|` 作为主键切分工单，不要用 csv.reader（易把多行线索拆错）。提取脚本与覆盖校验方法见 `references/ontrack-csv-extraction.md`。
2. **去重**：两份文档并集，按 CSD 编号去重（交集通常仅 1 个左右）。
3. **二次筛选（用户明确要求，不能省）**：只保留"技术可规避、带根因+修复手段"的经验。剔除：
   - 纯进度跟进（"已解决可关闭""待反馈""持续观察"）
   - 纯链接/网盘/截图/密码/提取码/真实 IP
   - 纯"换镜像版本就解决、无根因"
   - 型号空且无报错无修复的空工单
   - 需换模型 / 硬件 RMA（UE 错误、ubb 未供电、PCIe 卡槽故障、锁频）→ 仅进 E 节提示"走换卡/报修"，不写软件规避
4. **不抄只指路**：新增的 env/修复条目**只存在于本手册（唯一样本源）**。在 `dcu-driver`/`dcu-docker`/`dcu-rccl-test` 等老 skill 里加一句"先来本 skill 按现象反查"的提示即可，**不要**把 env 内容再抄进老 skill（用户原话："不是抄一份，可以提示去哪里找有没有处理经验"）。
5. **脱敏自检（落地前必跑）**：grep 确认无 `harbor.sourcefind.cn` / `image.sourcefind.cn` / 真实 IP / `密码` / `提取码` / 用户名（<device-A>/<device-B>/<internal-user> 等）。内网域统一记 `<internal_registry>`。
6. **不进 skill**：性能数值、逐行 dmesg 日志、基线表、客户名/行业/项目名一律不写。

### 条目格式（新增时遵守）
`现象关键词 → 设备型号 → 根因 → 规避/修复手段`，每条必须带来源工单可反查的根因，不能是孤立命令。

### ⚠️ 通信/拓扑类条目必加「产品形态前置」（防回归）
涉及 IOMMU/ACS、PCIe P2P、卡间拓扑、NUMA 跨 socket 等**与卡形态强相关**的条目，**禁止写"全系/全机型"**——
不同形态影响差异极大：
- **PCIe 形态卡**（K100-AI、BW10、BW100、BW150 等）：走 PCIe P2P 拓扑，BIOS 未关 IOMMU/ACS 会隔离直连、带宽不达标，须关。
- **OAM 形态产品**（BW1000、BW1100）：基本不受 IOMMU/ACS 影响，无需为此改 BIOS/内核；若现象出现在 OAM 机型，先排查其他根因，勿盲目套 PCIe 卡的修复。
新增/修订此类条目时，设备列按形态限定（如"主要为 PCIe 卡（K100-AI/BW/BW10/BW100/BW150 等；OAM 产品 BW1000/BW1100 基本无影响）"），根因列点明形态前提。
（来源：用户 2026-07-23 实测纠正——原"双机 ACS 导致通信异常 | 全系"误述已修正；dcu-env/dcu-driver/dcu-rccl-test 正文同步加前提。）
