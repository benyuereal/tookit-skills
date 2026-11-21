---
name: dcu-sglang-source
description: SGLang DCU源码结构与海光适配点索引
version: 1.0.1
author: DCU Engineer
license: MIT
---

# SGLang 源码适配点索引（dcu-sglang-source）

> ## 🔗 开源上游锚点（sglang-das）
> 官方开源版 **sglang-das**（`github.com/HYGON-AI/sglang-das`，44⭐）主线 tag **v0.5.15.post1**。本地实测基于 0.5.12 海光 fork + DTK 26.04；
> 0.5.15 起 DCU 术语全面演进为 HCU。`v0.5.15+` 的 fork 路径/函数名可能与本地 0.5.12 快照不同，跨版本先比对。
> 刷新：`python3 scripts/hygon-upstream-sync.py --check`（快照 `references/hygon-upstream/upstream-snapshot.json`）。

本 skill 沉淀 **SGLang 在 DCU（海光）上的源码结构与海光适配点**，并给出算子/backend 开发的扩展入口。
与 [`dcu-vllm-source-*`](../dcu-vllm-source/) 平行，但 SGLang **不按大版本拆独立 skill**（海光适配以 fork 构建形式随版本推进，见 §四实测），故单一 `dcu-sglang-source` 覆盖。

> 实测来源：<device-A> 设备 DTK26.04 镜像 `dsv4-flash-k100ai-sglang0.5.12-20260728-jupyter`，`sglang 0.5.12+g9a3a3e5f9`（海光 fork 构建），`/usr/local/lib/python3.10/dist-packages/sglang`。下文 §二/§三 的所有路径与文件均来自该容器实测，非推测。
>
> ⚠ **版本演进对照（2026-08-28 <device-A> 实测，0.5.12 vs 0.5.15）**：海光 fork 在新版把 DCU 术语全面演进为 **HCU**。写代码时先确认目标版本术语，别把旧版写法硬套：
> | 项 | sglang 0.5.12（本 skill 主体） | sglang 0.5.15.post2（新版 fork） |
> |---|---|---|
> | 平台判断 | `is_dcu()` | `is_hcu()`（server_args 引入 `is_hcu`/`is_hcu_native_fp8_supported`） |
> | MLA backend | `srt/layers/attention/dcu_mla_backend.py` | `srt/layers/attention/hcu_mla_backend.py` |
> | server_args 常量 | `DCU_ATTENTION_BACKEND_CHOICES`（含 `dcu_mla`） | `HCU_ATTENTION_BACKEND_CHOICES`（含 `hcu_mla`），且 `is_hcu()` 时默认返回 `"hcu_mla"` |
> | 新增文件（0.5.15 独有） | — | `hip_flash_mla.py`、`deepseek_v4_backend_hip_radix.py`、`dsa/`、`nsa/`、`hybrid_linear_attn_backend.py`、`minimax_*`；`dsv4/` 新增 `fp4_indexer.py`/`fused_compress_triton.py`/`compress_hip.py` |
>
> 二者 backend 名、平台函数名不同，混用会找不到符号。grep 定位海光适配入口时：0.5.12 搜 `is_dcu`/`dcu_mla`，0.5.15 搜 `is_hcu`/`hcu_mla`。

## 一、SGLang 算子/backend 开发的三种扩展体系（通用）

SGLang 不像 vLLM 只有一个 `_custom_ops.py` C++ 注册体系，它有三套并存：

| 体系 | 编译方式 | 入口 | 适用 | DCU 适配现状 |
|---|---|---|---|---|
| **Triton kernel** | 运行时 Triton JIT | `python/sglang/srt/layers/attention/triton_ops/*`、`triton_backend.py` | 最易上手、跨平台 | ✅ 海光主力路径（`dcu_mla_backend` 即 Triton/flash_mla） |
| **`sgl-kernel`** | AOT 编译 CUDA/C++ 二进制（`.so`） | `sgl_kernel` 包（C++/CUDA + torch 绑定） | 极致性能、生产 | 海光版带 `sgl_kernel.flash_mla` 的 DCU 实现 |
| **`jit_kernel`** | JIT 运行时编译（C++ in `python/sglang/jit_kernel/csrc` + `tvm-ffi` 绑定） | `python/sglang/jit_kernel/`（`load_jit` 工具） | 免 AOT、运行时编译 | 海光版存在该顶层目录（实测） |

> 开发新算子优先 **Triton**（开发快、易验证）；要极致性能再下沉到 `sgl-kernel`(AOT) 或 `jit_kernel`(JIT)。`dcu-kernel-dev` 的 HIP/TileLang 路径可直接复用为 SGLang kernel 的 DCU 后端。

## 二、attention backend 注册机制（SGLang 最核心的扩展点）

SGLang 的 attention 高度插件化。新增 backend 模式（官方 `attention_backend.md` + 实测 `triton_backend.py`）：

1. 继承 `sglang.srt.layers.attention.base_attn_backend.AttentionBackend`；
2. 用 `@register_attention_backend` 注册（或 `direct_register_custom_op` 注册自定义 op）；
3. 通过 `--attention-backend <name>` 选择，`--prefill-attention-backend` / `--decode-attention-backend` 可 **prefill/decode 分相位选择**；
4. 支持 per-phase 后端混搭（如 prefill=FA4, decode=TRTLLM MLA）。

### ★ DCU 海光实测适配点（<device-A> 容器，§四 来源）
- **`srt/layers/attention/dcu_mla_backend.py`**：版权 `Copyright 2026 Hygon Information Technology Co., Ltd.` —— **海光官方写的 MLA attention backend**。
  - 用 `from sglang.srt.utils import is_dcu; _is_dcu = is_dcu()` 做平台分发；
  - `from sgl_kernel.flash_mla import dcu_create_flashmla_kv_indices` + `flash_mla_with_kvcache`（FP8 路径 `flash_mla_with_kvcache_fp8`）；
  - 受 `SGLANG_USE_FUSED_MLA_CAT` / `SGLANG_USE_FUSED_RMSNORM_ROPE` 环境变量开关控制。
- **`srt/layers/attention/dsv4/`** 子目录：海光 DSV4 模型完整适配（`compressor.py` / `indexer.py` / `flash_mla_triton.py` / `tilelang_kernel.py` / `sparse_prefill_utils.py` 等），含 **TileLang kernel**（`tilelang_kernel.py`）与 Triton 双实现。
- **`srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`**（227 行）：ROCm 专属 fused RMSNorm+Rope。
- **`srt/server_args.py`** 已原生支持 DCU 开关：`DCU_ATTENTION_BACKEND_CHOICES = {"dcu_mla", ...}`、`DCU_GENERIC_ATTENTION_BACKEND_CHOICES`、`DCU_NSA_PREFILL_BACKEND_CHOICES` —— 即 `--attention-backend dcu_mla` 是海光版一等公民。

## 三、kernel 开发 / profiling 纪律（与 vLLM 互补）

- **平台分发**：SGLang 海光用 `is_dcu()`（而非 vLLM 的 `is_rocm()`/`is_cuda()`）；新写 DCU 分支时认准 `is_dcu()`，不要套 vLLM 的 `is_rocm` 写法（两套命名不同）。
- **ROCm 条件编译范例**（上游 AMD PR #7661 AWQ dequant、#4178 Flex Attention）：统一用 `if is_cuda() or is_hip(): use_cuda_kernel else: use_triton/other` 模式。
- **Triton 后端 DCU 适配**：参照 `dcu-kernel-dev` 的 TileLang/HIP 已知问题（warp_size=64、target=hip）；SGLang 内 `dsv4/tilelang_kernel.py` 已是实战样板。
- **profiling 纪律（LMSYS Agent-Assisted SGLang Development 方法论）**，与 `dcu-op-optimization` 闭环一致：
  - 先产出 **kernel 表 / pipeline 表 / overlap-fuse 表 / NCU 报告**，用 profile 解释差距，**不要急于改代码**；
  - 只 patch **证据支持**的路径（hybrid attention / Mamba-GDN / radix cache / CUDA graph / MoE-EP / quant kernel）；
  - **hard-stop 条件**写进 skill：若日志显示 silent 切换到其他 backend、禁用 CUDA graph、走 diffusers wrapper 等，该 trace **不能作为目标路径的证据**。

## 三·补、KV page-size 与 vendor flash-attn 硬约束（2026-08-20 <device-A> 真机实测，sglang 0.5.12 + DTK 26.04 + DAS flash-attn 2.8.3+das.opt1.dtk2604）

> 背景：评估"KV 页对齐命中 vendor paged-attn"类优化时实测沉淀。适用 gfx936（BW1000 类）；其他 gfx 以 strings 实测为准。
> **版本边界**：以下结论均基于标题版本组合（sglang 0.5.12 海光 fork + DTK 26.04 + DAS flash-attn 2.8.3+das.opt1）。.so 文件名、默认 page_size、mamba 参数冲突等随版本/构建可能变化——换环境先 `strings <fa .so> | grep -i "block size"` + 读 `_handle_page_size()` 复核，别直接套用。

- **SGLang 在 DCU 上默认 `page_size=1`**（`server_args.py` 的 `_handle_page_size()`）。生产脚本必须**显式 `--page-size 64`**，否则服务能启动（权重/KV 分配/cuda graph 全成功）但**第一个请求的 prefill 即崩**：`RuntimeError: Paged KV cache block size must be 64`（`flashattention_backend.py` forward_extend → `vllm_flash_attn_varlen_func` → C++ `vllm_mha_varlen_fwd`）。
- **报错文案有误导**："must be 64" 实际约束是 **page 为 64 的倍数**——独立进程实测 prefill：page=64/128/256 通过，1/16/32 失败。二进制内另有 `Paged KV cache block size must be divisible by 64 at gfx936 platform`、`...divisible by 16 at gfx928 platform`、`unified attn block size must be divisible by 64 at gfx936 & gfx938 platform` 等字符串佐证（不同路径约束不同）。
- **decode 路径（C++ `paged_attention`，max_seqlen_q<5 快速路径）无对齐要求**：page=1/16/32/64/128/256 单独跑全部正常。且 bs=1 微基准下 page=1 反而最快（32K 比 page=64 快 77%）——但 page=1 因 prefill 崩溃不可作为 serving 配置，"decode 更快"无工程意义。
- **vendor 包结构坑**：`flash_attn/flash_attn_interface.py` 里是 `import flash_attn_2_cuda as flash_attn_cuda`，C++ 扩展实际是 `flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so`（不是 `flash_attn_cuda` 模块）。约束/报错字符串在 .so 里，Python 源码 grep 不到，须 `strings <so> | grep -i "block size"`。
- **DCU KV layout 转置**：`SGLANG_KV_LAYOUT_DCU_FA=true`（默认）→ k=(pages, h_k, page, d)、v=(pages, h_v, d, page)。写 standalone 复现/微基准时必须按此 layout 构造张量，否则结论不可信（layout 不匹配的复现会得出"kernel 是坏的"的错误结论）。
- **参数冲突**：`--mamba-scheduler-strategy extra_buffer` 与 `--disable-radix-cache` 硬冲突（ValueError），混合注意力模型（qwen3_5 类）做关前缀缓存的 A/B 时须改 `no_buffer`。

## 三·补2、量化与融合路径可用性清单（2026-08-21 <device-A> 实测，sglang 0.5.12 海光 fork + DTK 26.04，BW150 gfx936）

> 背景：Qwen3.8-27B（qwen3_5 混合架构）输出吞吐优化时逐条验证的量化/融合路径可用性。换版本/构建先复核，别直接套用。

- **FP8 权重路线死路（两条路径都不可用）**：
  - block-wise 路径 `hipblaslt_w8a8_block_fp8_linear`（128x128 权重 scale + per-token group 输入 scale）在 M=1 decode 形状（如 up_proj m=17408, n=1, k=5120）直接 **VMFault**：`[BLASLT] No suitable hipblasLt algo found` → 非法访存。scale 布局验证正确，是 hipblaslt 对该 decode 形状无可用算法。
  - per-tensor 动态路径（`w8a8_fp8`）：fork bug——`sgl_per_tensor_quant_fp8` 只在 `if _is_cuda or _is_musa` 分支 import，DCU else 分支调用即 `NameError`；native fallback 也被 `not _is_dcu` 门控。
  - 独立验证：`deepgemm.fp8_gemm` 传 per-tensor scale 能跑但结果全错（rel err 1.0~68）——它期望 block-wise scale。
- **INT8 路线死路**：`sgl_kernel` 的 `int8_scaled_mm` / `sgl_per_token_group_quant_int8` / `sgl_per_token_group_quant_8bit` 符号在 `dir()` 列出但**未链接**（调用即 AttributeError）。DCU 构建里 INT8 GEMM 路径不可用，除非重编译 sgl_kernel。
- **lightop 现成融合路径是半成品（`SGLANG_USE_FUSED_RMS_ROTARY=1` 启动即崩）**：`gammarms_rotary_embedding_fuse_with_kv_store`（融合 RMSNorm+RoPE+kv_store）两个坑：① `output_dtype=self.kv_cache_dtype`（fp8_e5m2）把 q 也转 fp8，fa3 prefill 断言 `UnSupport q.dtype:fp8`；② 返回 2 元组而模型代码按 3 元组 `q, k, v = ...` 解包。结论：不能直接用，需自写 kernel。
- **自写融合 kernel 接入范式（已验证可落地）**：env 开关（`get_bool_env_var`）+ import 失败自动 fallback 上游路径 + **bit-identical 门禁**（与现有 kernel 输出逐元素比对，含非连续 stride 布局）+ 生产脚本默认开启。bit-identical 关键：复刻 torch 两步运算的**中间 bf16 舍入**（如 `sigmoid` 后先 `.to(bf16).to(f32)` 再乘，否则 ~75% 元素差 1 ULP）。
- **elementwise 融合 ROI 判据**：decode 全程 CUDA graph 时，小 kernel 的 CPU launch 开销已被吃掉，融合只省 GPU 时间（少一个 kernel + 一次中间张量 HBM 往返）；若 GEMM+COMM 占比 ~70%，elementwise 融合端到端增益≈0（噪声内）。**融合收益主要在投机解码开启时**（draft+verify 多份前向放大小 kernel 开销，c=1 可测出 +4% 量级）。先 profile 看 elementwise 占比再决定投入。
  - ⚠ 上述占比（GEMM 39.9%/COMM 28.8%）与增益量级（纯 decode ≈0、带 EAGLE c=1 +4%）均为 <device-A>/BW150/0.5.12 fork/Qwen3.8-27B 单点实测，**只适合当前环境**：换硬件（HBM 带宽/卡间互联不同）、换模型（层数/混合比不同）、换版本（kernel 实现不同）时占比与增益都会变，结论方向可参考、数字不可套用，落地前必须在本环境重新 profile + A/B。

## 三·补3、gfx928 q≥5 paged multi-token no-write vendor bug（2026-08-24 Qwen3.8-27B 三产品线研究，平台事实）

> 适用 gfx928（K100AI 类）；其他 gfx 以 sentinel-write 实测为准。

- **现象**：vendor 的 paged multi-token attention consumer 在 **q≥5** 时是 **no-op**（不写输出）——历史「超快」的 native 多 token 路径实为无效（漏写/重复），导致 quality 回归但表面 TTFT 异常快。
- **验证方法（sentinel-write）**：在输出 buffer 预填 sentinel 值，跑 native 路径后检查 `changed_count`——q≥5 时 `changed_count=0` 证明 kernel 根本没写（no-op）。这是判定「vendor kernel 是否真写」的可靠手段，比看 TTFT 快慢可靠得多。
- **对策**：禁 vendor q≥5 no-write 路径，改用 proven corrected path（q≤4 native decode 仅在已验证路径允许）。所有 q≥5 prefill/verify 走 corrected consumer。
- **通用教训**：「TTFT 异常快」可能是 kernel 漏写（no-op）而非优化——遇到「快得可疑」先做 sentinel-write 验证 kernel 是否真写，再下「优化成功」结论。
- **上游状态**：截至 2026-08-24 公开检索未找到完全对应的 upstream issue，最强证据是项目自身 full-model A/B + isolated reference + sentinel-write 实验，不应误写成「已被上游官方确认」。

## 三·补4、Skip-Softmax 环境变量在 DCU 上无效（2026-09-03 <device-A>/<device-B> 实测，sglang 0.5.12 + DTK 26.04）

> 背景：评估 NVIDIA Skip-Softmax 长上下文 prefill 优化（BLASST，arXiv:2512.12087，TRT-LLM/FlashInfer kernel 级优化，`SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR=587.18` 类推荐值）能否在 DCU 上生效。

- **环境变量存在但被静默忽略**：`SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR` / `SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR` 在 `srt/environ.py` 有定义（继承上游，默认 None），但全代码库**只有 3 个 backend 读它**：`trtllm_mha_backend.py`（→ `flashinfer.prefill.trtllm_batch_context_with_kv_cache`）、`nsa_backend.py`（→ `flashinfer.prefill.trtllm_ragged_attention_deepseek`）、`trtllm_mla_backend.py`（→ flashinfer，且该 backend 开头 assert 仅支持 Blackwell SM100/SM12x）。
- **3 个消费方全依赖 flashinfer 的 TRT-LLM kernel**（NVIDIA Hopper `fmha_v2` / Blackwell `trtllm-gen`），DCU 容器**没装 flashinfer**（`is_flashinfer_available()=False`，`import flashinfer` 直接 ModuleNotFoundError），DCU 上也没有对应 kernel。
- **DCU 默认 attention backend 是 `aiter`**（`server_args.py` `_get_default_attn_backend` 的 `is_hip()` 分支），aiter 包**没有任何 skip-softmax 实现**（grep `skip_softmax|softmax_threshold|threshold_scale` 0 命中）。
- **结论**：DCU 上设 `SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR`（任意值）完全无效——默认 aiter 路径不读该变量（静默忽略）；强行 `--attention-backend trtllm_mha/nsa` 会因缺 flashinfer 报错。上游文章/博客里的校准值（587.18 等）只适用于 NVIDIA Hopper/Blackwell。
- **可复用验证配方**（评估任意上游 SGLang 优化环境变量在 DCU 是否生效）：① `grep -rn "<ENV_VAR>" $SGDIR/srt/` 找消费它的 backend → ② 看消费 backend 是否依赖 flashinfer/NVIDIA 专属 kernel → ③ 看 DCU 默认 backend（aiter）是否读它 → ④ `python3 -c "from sglang.srt.utils import is_flashinfer_available; print(is_flashinfer_available())"` 确认 flashinfer 可用性。②是且③否 → 该 env var 在 DCU 无效。
- **DCU 长上下文 TTFT 替代方向**：chunked prefill（`--chunked-prefill-size`）、aiter/lightop attention 调优；确需稀疏 attention 得在 aiter 里自实现 skip-softmax 逻辑（参考 BLASST 论文），属算子开发工作（见 `dcu-kernel-dev`），不是改环境变量。

## 四、实测探查方法（DCU 容器）

> 验证/扩展 SGLang DCU 适配时，在 <device-A>/<device-B> 等设备的**指定镜像容器**内操作（镜像名含 `dtk26.04` / `sglang`）。

```bash
# 1. 起容器（必挂 hyhal，DCU 可见）
docker run -dit --name sgl_<ver> --device=/dev/kfd --device=/dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  <registry>/<sglang-img>:0.5.12-ubuntu22.04-dtk26.04-...-jupyter \
  bash -c "sleep infinity"

# 2. 定位源码
SGDIR=$(python3 -c "import sglang,os;print(os.path.dirname(sglang.__file__))")
echo $SGDIR   # 实测: /usr/local/lib/python3.10/dist-packages/sglang

# 3. 查 ROCm/DCU 适配入口
grep -rlE "is_dcu|is_hip|is_rocm" $SGDIR/srt | head
ls $SGDIR/srt/layers/attention/        # 看 dcu_mla_backend.py / dsv4/ 等
python3 -c "import sglang;print(sglang.__version__)"   # 看是否 +fork 后缀

# 4. 起服务验证 backend（DCU）
python3 -m sglang.launch_server --model <model> --attention-backend dcu_mla --tp 8
```

- gfx 用物理机 `rocminfo` 查（不写死）；`HIP_VISIBLE_DEVICES=0,1,2,3` 指定部分卡。
- 改动后重编译：SGLang 的 `sgl-kernel`/`jit_kernel` 在容器内 `pip install -e .` 或按镜像内构建脚本；条件同 `dcu-vllm-op-replace` §五 重编译纪律。

## 离线参考代码（DTK 镜像内，未随本包分发）

海光 SGLang 0.5.12(+g9a3a3e5f9) 海光 fork 的 DCU 专属适配源码位于 DTK26.04 镜像（`dsv4-flash-<ver>` 系列）容器内 `/usr/local/lib/python3.10/dist-packages/sglang`，**未随本包分发**（上游代码合规红线见 `docs/STANDARDS.md`）。需要查看原文时自行 `docker cp <容器>:/usr/local/lib/python3.10/dist-packages/sglang <目标目录>` 抽取，保持源码原样。

> 本节只记录源码位置与结构，不内联长代码、不存性能基线/逐行日志。

### 目录与文件清单（镜像内布局，docker cp 抽取后一致）
```
sglang（dist-packages 下）:
├── attention/
│   ├── dcu_mla_backend.py                    # 海光官方 MLA backend（Copyright 2026 Hygon），is_dcu() 分发 + sgl_kernel.flash_mla + FP8
│   └── dsv4/                                  # DSV4 模型完整适配子目录
│       ├── __init__.py
│       ├── compressor.py / compressor_v2.py / int8_compressor.py   # KV cache 压缩（含 int8）
│       ├── quant_k_cache.py / dequant_k_cache.py / int8_k_cache.py  # K cache 量化
│       ├── indexer.py / index_buf_accessor.py / metadata.py / metadata_kernel.py  # 索引与元数据
│       ├── sparse_prefill_utils.py            # 稀疏 prefill 工具
│       ├── flash_mla_triton.py                # FlashMLA Triton 实现
│       ├── tilelang_kernel.py                 # ★ TileLang kernel 实战（与 dcu-kernel-dev TileLang 路径呼应）
│       ├── flashmla_logic_triton/             # FlashMLA 逻辑（dense_decode / sparse_attn Triton kernel + BW1000B 配置）
│       └── mqa_logic_triton/                  # MQA 逻辑（Triton kernel + BW1000B 配置）
└── deepseek_common/
    └── attention_forward_methods/
        └── forward_mla_fused_rope_rocm.py    # ROCm 专属 fused RMSNorm+Rope（227 行）
```

### 用途
- 写新的 DCU attention backend：以 `dcu_mla_backend.py` 为模板（认准 `is_dcu()` 分发 + `sgl_kernel.flash_mla` 调用 + `SGLANG_USE_*` 环境变量开关）。
- 做 DSV4 / DeepSeek-V4 类模型适配：参考 `dsv4/` 的 compressor/indexer/flash_mla_triton 组合。
- TileLang kernel 开发：参考 `dsv4/tilelang_kernel.py`（海光已在生产路径使用 TileLang，验证 dcu-kernel-dev 的 TileLang 路线可行）。
- ROCm fused 算子：参考 `forward_mla_fused_rope_rocm.py`。

### 注意
- 这些是**海光 fork 源码**，上游 SGLang main 分支路径/函数名可能不同；跨版本请先比对。
- 镜像内源码 py_compile 校验通过，但**未经运行验证**（需真实 DCU 设备起服务验证，参照 §四）。

## DSpark 移植参考

**DSpark = Confidence-Driven Speculative Decoding**（置信度驱动投机解码），上游 SGLang 已集成（PR #30261，2026-07-06），支持 dense + sparse 模型（Qwen3 / DeepSeek-V4）。完整移植参考见 [`DSPARK_PORTING.md`](./DSPARK_PORTING.md)，上游源码快照见 [`scripts/dspark_upstream/`](./scripts/dspark_upstream/)（sglang main 快照，23 文件，py_compile 通过），算法级验证见 [`scripts/dspark_sandbox_verify.py`](./scripts/dspark_sandbox_verify.py)（纯逻辑跑通）。

要点：
- **海光 fork 0.5.12 不支持 DSpark**（grep `dspark` 全量 0 命中）；已落地的是 DeepSeek-V4 的注意力/压缩部分（`dsv4/`），投机解码只到 eagle/mtp/ngram。
- **移植阻断点**：海光 fork 缺 `draft_worker_common.py` / `kernels/ops/speculative/dspark/`（4 Triton kernel）/ `ragged_verify.py` / `kv_canary` / v2 worker 体系 —— **必须先 rebase 上游 main 对应 commit**，不是加 `is_dcu()` 桩就能跑。
- **is_dcu() 适配点**：rebase 后改 `dspark_worker_v2.py` 两处 `is_cuda()` → `is_cuda() or is_hip()`。
- **版本错配警告**：快照来自 main（post #30261），非 0.5.12，严禁直接拷进海光 0.5.12 指望能跑。

## DFlash2 / 原生 MTP 投机解码适配点（2026-08-20/21 <device-B> 实测，Qwen3.8-27B mamba 混合）

- **原生 MTP 走 EAGLE 路径，不是 FROZEN_KV_MTP**：`--speculative-algorithm EAGLE --speculative-draft-model-path <target ckpt> --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`。后三个参数**必须显式给**（server_args 的 EAGLE/FROZEN_KV_MTP 块不会自动设，否则 worker `range(None)` 崩）。FROZEN_KV_MTP 常因 draft 模型缺 `build_frozen_kv_mtp_context` hook → kv_context=None → graph capture 崩。
- **DFlash2**：`--speculative-algorithm DFLASH --speculative-draft-model-path <DFlash2 ckpt> --speculative-num-draft-tokens 4`。draft 共享 target 的 req_to_token_pool 但有**独立 KV pool**，commit 时物化 draft KV（dflash_worker.py `_append_target_hidden_to_draft_kv`）。本 build 对 DFLASH 强制 `disable_overlap_schedule=True`（spec v1 无 overlap）。
- **draft 注意力 fa2/fa3 选择（DCU 特有约束）**：
  - fa3 **paged** 路径在 gfx936 要求 "Paged KV cache block size divisible by 64"，head-major 布局（`SGLANG_KV_LAYOUT_DCU_FA=true` 默认，k `[pages,heads,page,dim]`）不满足 → 不可用。
  - fa3 **packed** 路径正确且长上下文快 6-12x/层，但 `pack_paged_kv_to_varlen` 有 `seq_lens_cpu.tolist()` CPU 同步 → **与 draft CUDA graph 互斥**（graph 值 ~22ms/轮，fa3 只省 ~1ms）→ 净负收益。
  - **fa2 回退（`vllm_flash_attn_varlen_func`）忽略 sliding window**（耗时随上下文线性增长，读整个上下文）——非因果+滑窗的 draft 注意力走 fa2 回退是 DCU 上的最优组合（graph 安全、全 GPU tensor）。
- **窗口 clamp / compact draft cache（现成 flag，零代码）**：`--speculative-draft-window-size N`（= draft 训练 sliding_window）开启 compact cache（dflash_worker.py `use_compact_draft_cache`），draft 只 attend/物化最近 N token 并 page 对齐。长上下文 2.2x（还砍 commit 成本）。**提"要自己实现 X"前先 grep server_args 有没有现成 flag。**
- **draft CUDA graph 烧死标量**：fa2 回退的 `max_seqlen_k`（每 batch Python int）会被烧死在 capture → replay 只读 5 个 KV token → accept 恒 0。改常量 `self.max_context_len`（flashattention_backend.py 三处）。详见 `dcu-cudagraph-baked-scalars`。
- **并发 KV 泄漏**：window=2048 compact cache 在并发下请求结束后可能有未归还 token（`pool memory leak detected! ... evictable=N`），严格检查 `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE`（默认 True）会 raise 崩 server。临时绕过设 False；根治查 compact req→token 表释放路径。
- **selector 成本**：block-diffusion 的 selector 瓶颈是 **TP all-gather 通信**（vocab 分片下全局 top-k 必需），不是 torch.topk，换 triton kernel 无收益。
- **prefill CUDA Graph 必须独立 correctness promotion（不要全开）**：2026-08-19 SGLang 上游 issue 报告 DFlash2 + prefill CUDA Graph 的 BCG capture assert / first-request IndexError。decode/verify graph 可激进，但 **prefill graph 必须单独走 correctness 晋级**（exact semantic + repeated SHA + 首请求无 IndexError），未晋级前保持 prefill eager 或 gated。这是「speculative/graph/kernel 组合必须逐 geometry 晋级，不能假设在别的 GPU/TP 能跑就一定安全」的延伸。
- **Tree EAGLE（`eagle-topk>1`）与 fa3 backend 互斥（启动即崩）**：tree 路径经 `hybrid_linear_attn_backend` 进 `flashattention_backend.forward_extend` 时 unpacked-tuple 崩——tree 路径和 fa3 不能同用。可跑的 tree 配置（triton+page1）端到端 **-18~-30%**（负收益）。要 tree 投机就别配 fa3，反之亦然。
- **DFLASH 算法被接受但需外部 draft ckpt**：`--speculative-algorithm DFLASH` 本 build 接受（overlap scheduler 自动禁用），需外部 draft checkpoint（如 `incoai/Qwen3.8-27B-DFlash2`，~3.85 GB）。

## HiCache（分层 KV 缓存）DCU 适配 + 三个上游 Bug（2026-09-01 8 卡 TP=8 + 2026-09-03 4 卡 TP=4 实测，Qwen3.8-27B 混合 Mamba）

> 适用：SGLang HiCache（`--enable-hierarchical-cache`）在 DCU 上跑**混合 Mamba + 全注意力**模型（qwen3_5 类）。纯注意力模型走 `hi_radix_cache.py`，混合模型走 `hi_mamba_radix_cache.py`——**后者 L2 host 回读路径有上游 bug**（纯注意力版无此问题）。DCU 上 HiCache 混合模型要跑通需同时解决 **Bug A（kernel 后端野指针）+ Bug B（direct 后端分页布局）+ Bug C（load_back 引用计数）**，缺一即崩。

### 启动参数（混合模型 + HiCache）
- `--enable-hierarchical-cache --hicache-ratio <N> --hicache-io-backend direct --hicache-mem-layout layer_first --hicache-write-policy write_through`
- `--hicache-ratio` = host KV 池相对 GPU KV 池的倍数。
- **L3 磁盘 file 后端**（在 L1/L2 基础上加）：`--hicache-storage-backend file --hicache-storage-prefetch-policy timeout` + 环境变量 `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=<l3磁盘目录>`（**不是 `--hicache-storage-path`，该参数在 sglang 0.5.12 不存在**）+ `--enable-cache-report`（否则 `cached_tokens_details` 恒 null）。详见下方「L3 磁盘 file 后端 + write policy 选型 + 命中观测」。
- 混合模型 Mamba host 池 = device Mamba 池 × ratio，**默认 ratio 下 Mamba host 池巨大**（曾见 20.8GB/rank）。内存受限 pod 必须用 `--max-mamba-cache-size <slot>` 显式压 Mamba device 池（host 池随之缩小）。并发上限 ≈ `max_mamba_cache_size / mamba_ratio`（MTP EAGLE + extra_buffer 时 mamba_ratio=5）。

### ★ 三个上游 Bug（A/B/C，缺一 HiCache 混合模型跑不通）

> 一键补丁脚本：[`scripts/hicache_dcu_patch.py`](./scripts/hicache_dcu_patch.py)（幂等，`--check` 只查不改，自动备份 + py_compile 校验）。Bug A 是配置级（改 `--hicache-io-backend`），Bug B/C 是代码补丁（脚本自动打）。

**Bug A：kernel 后端 — JIT 传输 kernel 写野指针（配置级修复）**
- `--hicache-io-backend kernel`（默认）走 JIT 传输 kernel（`transfer_hicache_all_layer` / `transfer_kv_all_layer`），这些自定义 kernel 在 DCU 硬件上越界写/写野指针，损坏后续 kernel 读取的内存，表现为 GDN/decode 等**无关 kernel 的 VMFault**（scheduler exit -11），极难定位。首个 ≥256 tok 的 prefill 即崩，完全不可用。
- **修复**：改用 `--hicache-io-backend direct`（配置级，走另一条已注册的 torch op `transfer_kv_direct`，是真正不同的代码路径）。

**Bug B：direct 后端 — 分页 KV 布局与 host 扁平布局不匹配（代码补丁）**
- 仅改 direct 后，fresh prefill 能跑，但一旦触发 GPU 驱逐→写回 host 就崩：`RuntimeError: The size of tensor a (2) must match the size of tensor b (64) at non-singleton dimension 2`（`evict → write_backup → backup_from_device_all_layer → transfer_kv_direct`）。
- 根因：DCU 上 `SGLANG_KV_LAYOUT_DCU_FA=true`（默认），GPU 侧 KV 是**分页布局** K=(page_num, head_num, page_size, head_dim)、V=(page_num, head_num, v_head_dim, page_size)；host 侧 layer_first 是**扁平布局** (size, head_num, head_dim)。编译好的 `transfer_kv_direct` C++ op 假设 GPU 侧是扁平 token 索引布局，遇到分页布局就在 dim2 上 2 vs 64 广播冲突。
- 注意"fresh 能跑"的真相：token 没填满 GPU KV 池时从未触发驱逐/写回 host，所以 fresh 测试测不出此 bug。
- **修复**：`memory_pool_host.py` 纯 PyTorch 分页 D2H/H2D 回退补丁（`_dcu_paged_device` 判定 + `_dcu_d2h_paged`/`_dcu_h2d_paged` gather/scatter，direct 后端 H2D/D2H 两处调用点加分支，只在冷路径触发、热路径零开销）。

**Bug C：`hi_mamba_radix_cache.py` `load_back` host 引用计数缺失（代码补丁，上游缺陷）**
- **触发条件**：前缀被从 GPU 驱逐（write_back 策略下驱逐时才写 host）后再次请求 → 走 L2 host 回读（`load_back`）。**工作集必须 > GPU 池但 ≤ GPU+host 总容量**才触发；工作集 > 总容量时数据被彻底丢弃（全 MISS），不触发 load_back，所以小负载/全 MISS 测试测不出来。
- **崩溃 1**：`TypeError: object of type 'NoneType' has no len()`（`load_back` 里 `n_len = len(n.host_value)`）。根因：`load_back` 收集 evicted 节点进 `nodes_to_load` 却**从不调 `protect_host()`**；而 `inc_lock_ref` 显式跳过 evicted 节点（`if node.evicted: node = node.parent; continue`），`host_ref_counter` 恒 0。回读期间 `evict()` 级联删 `host_ref_counter==0` 的节点 → `host_value=None` → `len(None)` 崩。
- **崩溃 2**（修完崩溃 1 后暴露）：`RuntimeError: Host reference counter is already zero.`（`release_host()`）。根因：成功路径里 `release_host()` 被调**两次**（1→0→-1）。
- **修复**（`load_back` 内 4 处补丁，见 `scripts/hicache_dcu_patch.py` 的 R1-R4）：① 收集 `nodes_to_load` 后对每个节点 `protect_host()`；② 三条返回路径（skip / load 失败 / 成功）各只 `release_host()` 一次——把成功路径里重复的那次移进 `if full_device_indices is None` 分支内。改完 `py_compile` 校验。
- **⚠ 版本锚点**：R1-R4 锚点基于 **SGLang 0.5.12+das.opt1.dtk2604** 的 `load_back` 原始代码。其他 SGLang 版本若 `load_back` 实现有差异，锚点 assert 会失败（**不会静默打错**），需先核对 `hi_mamba_radix_cache.py` 的 `load_back` 源码再调整锚点。
- **诊断要点**：8 个 TP rank 同时命中同一异常 → SIGQUIT → 进程被 Killed。**这是进程级 Python 崩溃，不是 OOM**——区分手段：cgroup `memory.events` 的 oom/oom_kill 计数 + 容器主机名是否变（OOM 重建会换主机名后缀，进程崩溃不会）。

### 验证方法（可复用：A/B + 精度）
- **A/B 公平**：baseline 与 HiCache 用**完全相同**的 `--max-total-tokens` / `--max-mamba-cache-size` / `--mem-fraction-static`，只差 4 个 HiCache 参数。
- **触发 L2 的负载设计**：并发工作集（concws）——N 个会话 × 唯一长前缀，工作集落在 (GPU 池, GPU+host 总容量) 区间。Phase1 全 fresh，Phase2 全重发。Phase2 看 `cached_tokens_details`：`host>0` 即 L2 命中（只有 HiCache 有）；baseline 同负载 Phase2 应大量 MISS（无 host 层，驱逐=丢弃）。
- **L1/L2 区分**：请求加 `return_cached_tokens_details: true`，响应 `sglext.cached_tokens_details = {device, host}`。
- **正确性**：前缀末尾内嵌唯一事实标记 `FACT-<seed>-OMEGA` 要求原样输出，fresh 与 host 命中都正确召回 → host 读回 KV 无损坏。
- **精度不退化**：跑 GSM8K（`temperature=0`、`enable_thinking=false`），baseline 与 HiCache 跑**同样 N 题**，accuracy 应一致（连错题都相同）。HiCache 只改缓存路径不改权重/采样，理论上零精度影响，实测确认。

### 内存受限 pod 的坑
- 混合模型权重占大头，pod 内存上限（cgroup 可能显示虚高值，以实际 OOM 重建点为准）下 KV/Mamba 池只能开很小。**加载期内存峰值（模型 shard 读入 host）会瞬时超过稳态**——稳态安全不代表加载期安全，加载期尖峰才触发 OOM 重建。
- 稳态内存 baseline 通常**高于** HiCache（HiCache 把部分 KV 下沉 host 后 device 侧运行时缓冲更小），别据此误判"HiCache 更费内存"。

### L3 磁盘 file 后端 + write policy 选型 + 命中观测（2026-09-03 4 卡 TP=4 实测，<<internal-ip>> BW150）

**L3 磁盘 file 后端配置**（在 L1/L2 基础上加）：
```bash
--hicache-storage-backend file --hicache-storage-prefetch-policy timeout
# L3 磁盘目录用环境变量（不是 --hicache-storage-path，该参数在 sglang 0.5.12 不存在）
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/public/hicache_l3
mkdir -p /public/hicache_l3
# 观测命中必须带
--enable-cache-report
```
- file 后端走 HiCacheFile（POSIX 文件 I/O，host↔disk，不是 GDS；海光 GDS/XDS 走 LMCache+hyfile 另一条路，见 `dcu-lmcache` §5）。
- 磁盘文件命名：KV 为 `<64位hash>_<model>_<tp_rank>_<tp_size>.bin`（每 TP rank 各一份，各 512KB），Mamba state 为 `<hash>.mamba_...bin`。
- L3 跨重启持久化：磁盘缓存在服务重启后仍有效，重启后 fresh 请求直接从磁盘召回（storage>0）——这是 L3 最有力的证据。

**write policy 选型**（`--hicache-write-policy`，可选 write_back / write_through / write_through_selective，默认 write_through）：
- **write_through**（POC/生产推荐）：prefill 时 KV 同时写 L1+L2，L2 满即逐出 L3。数据最快落盘、崩溃安全（L3 已有数据）、跨重启持久化可靠、开销可预测（D2H 在 prefill 时固定发生）。POC 演示三层命中（尤其"重启后 L3 命中"杀手锏）必须用它。
- **write_back**：KV 先只写 L1，驱逐才写 L2，L2 满才写 L3。prefill 延迟更低（无 D2H），但数据要两次驱逐才到 L3，低负载下可能永远到不了 L3，崩溃即丢。仅当"只关心 L1/L2 命中 + 要零 prefill 回退 + 不需要 L3 持久化"时选。
- 本文所有测试均用 write_through（默认值）。

**命中观测（cached_tokens_details）**：
- 前提：`--enable-cache-report`（不带则 details 恒 null）。
- 原生 `/generate`：`meta_info.cached_tokens_details = {device, host, storage, storage_backend}`，直接返回无需额外参数。
- OpenAI `/v1/chat/completions`：请求体加 `"return_cached_tokens_details": true`，结果在 `sglext.cached_tokens_details`（结构同上），`usage.prompt_tokens_details.cached_tokens` 给总命中数。
- 判读（三值互斥，一个 token 只从一层命中）：`device>0`=L1（GPU，最快）/ `host>0`=L2（host，需搬回 GPU）/ `storage>0`=L3（磁盘，最慢）；全 0 或 null=全 miss（null 表示完全没命中，不是零值）。
- 辅助：L3 磁盘文件数/体积增长 = L3 在写入；Prometheus `/metrics` 无 hicache 专属指标（file 后端 `get_stats()` 返回 None，开了也拿不到 L3 统计）；serve.log 无逐请求命中日志。
- 触发 L2 的前提：工作集 > GPU KV 池容量（先查 `max_total_num_tokens`，灌入量要 > 它才触发驱逐）。Mamba 混合模型命中层会"错位"（驱逐时 Mamba 状态写 L3 → 再请求 storage>0；回载时 host>0），是回载机制正常行为，核心信号看事实召回 + 日志 0 错误，别死磕"必须命中 host 层"。

### 干净环境 rebuild 抓到的 4 个脚本 bug（2026-09-03，旧容器环境污染全掩盖了）

> 教训：**验证适配脚本必须用干净环境（强制重建容器）跑一遍**，别只在旧容器上测——旧容器的环境污染（bashrc 塞了 DTK 变量、cgroup 配置不同、之前手动操作过）会掩盖脚本自身的 bug。

| # | bug | 现象 | 修复 |
|---|---|---|---|
| 1 | 参数解析 | `--rebuild --step 1` 所有步骤都不跑 | `for a in "$@"` + 内部 shift → `while [ $# -gt 0 ]` 标准循环 |
| 2 | 验证依赖 DCU 环境 | 干净容器验证报 `No HIP GPUs`（import sglang 触发 lightop，/dev/kfd 打不开） | 补丁验证改 **ast 静态解析**（不 import sglang，不触发 lightop，不依赖 DCU 环境） |
| 3 | DCU 设备 EPERM | 新容器 `No accelerator available`（`-v /dev/kfd` 绑定挂载字符设备，cgroup 设备白名单没加） | 改 `--device /dev/kfd --device /dev/mkfd --device /dev/dri`（docker 自动加白名单） |
| 4 | pkill 自杀 | 脚本中途退出（`pkill -f sglang.launch_server` 匹配到执行它的 bash -c 自己，命令行含该字符串） | pattern 加方括号 `[s]glang.launch_server`（正则 `[s]glang` 匹配 "sglang"，但命令行里的 `[s]glang` 不匹配） |

**一键适配脚本**：[`scripts/hicache_adapt_dcu.sh`](./scripts/hicache_adapt_dcu.sh)（5 步幂等：预检→建容器→打补丁+ast 验证→起服务→验服务；`--rebuild` 强制重建容器防环境污染、`--step N` 单步、`--dry-run` 干跑；内嵌 Bug B/C 补丁 heredoc + serve 脚本模板；启动命令带 `< /dev/null` 防 SSH 会话被占住）。

### Bug B 跨设备复现 + 升级 flash-attn 修不了（2026-09-02/03 实测）

- **Bug B 是 sgl-kernel 0.3.21 的确定性 kernel bug，与硬件无关**：BW150（<device-A>，8 卡）和 HCU（ssh.zzai.<internal-domain>，2 卡）环境一致（sgl-kernel 0.3.21 + KV_LAYOUT 默认 true 分页 + page-size 64 + direct 后端）且真触发驱逐时都崩，崩溃路径完全一致（`write_backup → backup_from_device_all_layer → transfer_kv_direct (sgl_kernel/kvcacheio.py:242) → RuntimeError: The size of tensor a (2) must match the size of tensor b (64)`）。"另一台设备没有 Bug B"的真相是之前没真触发驱逐（KV 池没灌满）或已打补丁。
- **升级 flash-attn 修不了 Bug B**：崩溃 op `transfer_kv_direct` 在 sgl-kernel（0.3.21）里，不在 flash-attn 里。升级 flash-attn 到 2607280958 后不打补丁照样崩，**Bug B 必须靠代码补丁修复，不可省略**。
- **判断某环境是否有 Bug B**：不能看"跑没崩"，要看"是否真触发过驱逐 + sgl-kernel 版本"。

## 四·补5、PD 分离（disaggregation）部署模板与关键 DCU 环境变量

> 来源：某 BW1100 项目 PD 分离部署方案（SGLang 0.5.12 海光 fork + DTK 26.04 + DeepEP），**脱敏方法论点**——IP/模型路径/镜像/wheel/客户标识已去除。**完整可运行的脱敏启动模板见 [`scripts/sglang_pd_deploy_examples/pd_deploy_template.sh`](./scripts/sglang_pd_deploy_examples/pd_deploy_template.sh)**（P/D/Router + DeepEP 配置 + 单机/多模态，占位符替换即可用）。
> ⚠ 版本锚点：SGLang **0.5.12 海光 fork** + DTK 26.04。其他版本参数名/默认值可能不同，上线前以容器内 `sglang serve --help` / `python3 -m sglang_router.launch_router --help` 复核。

### 5.1 PD 分离三组件与部署要点

SGLang PD 分离 = **Prefill 实例 + Decode 实例 + Router** 三件套，用 `--disaggregation-mode prefill / decode` 区分 P/D，靠 Router 聚合。

- **Prefill（P）**：`--disaggregation-mode prefill` + `--disaggregation-ib-device <IB_DEVICES>`；内部常用 **CP8+EP8**（`--attn-cp-size 8 --ep-size 8`）+ `--enable-nsa-prefill-context-parallel --nsa-prefill-cp-mode round-robin-split`；`--moe-a2a-backend deepep --deepep-mode normal`；`--deepep-config <ep_config.json>`。
- **Decode（D）**：`--disaggregation-mode decode`；内部常用 **DP16+EP16**（`--enable-dp-attention --enable-dp-lm-head`）+ `--moe-a2a-backend deepep --deepep-mode low_latency`；**EAGLE/MTP 投机在这侧**（`--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`，后三个必须显式给，见 §DFlash2/原生 MTP 的坑）。
- **Router**：`python -m sglang_router.launch_router --pd-disaggregation --prefill <P> --decode <D> --policy <POLICY> --port <PORT>`。

### 5.2 ★ 配比 → 部署参数落地图

| 配比 | 场景 | 部署要点 |
|---|---|---|
| **1P1D** | 长输入（如 128k）、输出相对短、不要求低 TTFT | 1 P 副本 + 1 D 副本；P 用 CP8EP8、D 用 DP16EP16；router `cache_aware` |
| **2P1D** | 输入更短、要求更高吞吐/单卡并发 | 在 1P1D 基础上**再增 1 个 P 副本**；★ 每个 P 节点必须各加**唯一的** `--disaggregation-bootstrap-port <不同端口>`；★ router 策略**必须改 `round_robin`**，且每个 `--prefill` 后跟对应 bootstrap 端口 |

> 配比选型的通用方法论（1P1D vs 2P1D 怎么选、按 SLA/输入长度定）见 `dcu-llm-benchmark` §3.9。此处落**具体到 SGLang 部署参数**的差异。

### 5.3 关键 DCU 环境变量分类清单（脱敏）

完整清单见外置模板 `pd_deploy_template.sh`；以下是**类别与代表性开关**（含义/源码落点对照 §一~§三·补）：

| 类别 | 代表性变量 | 作用/备注 |
|---|---|---|
| 网络/HCA | `NCCL_SOCKET_IFNAME`、`GLOO_SOCKET_IFNAME`、`UCX_NET_DEVICES`、`NCCL_IB_HCA`、`NCCL_MIN/MAX_NCHANNELS` | 管理网口 + IB HCA；NCCL 通道数固定 |
| ROCShmem | `ROCSHMEM_IB_GID_INDEX=0`、`ROCSHMEM_TOPO_FILE_FORCE=<topo.config>`、`ROCSHMEM_HEAP_SIZE`、`ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX`、`ROCSHMEM_ALLOWED_IBV_DEVICES`、`ROCSHMEM_MAX_NUM_CONTEXTS` | DeepEP/RDMA 关键；`TOPO_FILE_FORCE` 指向卡网映射文件 |
| HIP/HSA | `GPU_MAX_HW_QUEUES`、`HIP_GRAPH_ACCUMULATE_DISPATCH`、`HIP_H2D/D2H_DIRECT_COPY_THRESHOLD`、`HIP_KERNEL_BATCH_CEILING`、`HIP_KERNEL_EVENT_SYSTENFENCE`、`HSA_KERNARG_POOL_SIZE`、`ROC_AQL_QUEUE_SIZE`、`GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000` | decode/大并发调优；`GPU_MAX_HW_QUEUES` P 用 2、D 用 3 |
| MC/allreduce | `MC_ENABLE_DEST_DEVICE_AFFINITY`、`MC_TE_FILTERS`、`USE_DCU_CUSTOM_ALLREDUCE`、`ALLREDUCE_STREAM_WITH_COMPUTE` | 通信/自定义 allreduce |
| SGLang 融合/量化 | `SGLANG_ROCM_USE_AITER_MOE`、`SGLANG_USE_DEEPGEMM_MOE`、`SGLANG_USE_FP8_W8A8_MOE`、`SGLANG_USE_LIGHTOP`、`SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT`、`SGLANG_USE_FUSED_MLA_CAT`、`SGLANG_USE_FUSED_DPSKV4_*`、`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`、`SGLANG_ENABLE_SPEC_V2`、`SGLANG_OPT_USE_FUSED_HASH_TOPK`、`SGLANG_NSA_FUSE_TOPK`、`SGLANG_KVALLOC_KERNEL` | DCU 适配核心（开关是否生效以源码落点为准，见 §二/§三·补）；P/D 侧取值可能不同（如 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` P 常用 256 / D 常用 128） |
| PD 超时 | `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`、`SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | disaggregation 建链/等待超时，常见 1200~1800 |
| 其他 | `SGLANG_APPLY_CONFIG_BACKUP=none`、`SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0`、`SGLANG_SET_CPU_AFFINITY=1`、`SGLANG_USE_MODELSCOPE=1`、`TVM_FFI_DISABLE_TORCH_C_DLPACK=1`、`SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` | 前缀缓存在 PD 分离常禁用 |

### 5.4 ★ 踩坑

1. **2P1D 的 Router 策略若仍用 `cache_aware` 会路由错乱**——多 P 副本必须改 `round_robin`，且每个 `--prefill` 后要跟该 P 节点的 `--disaggregation-bootstrap-port`。
2. **2P1D 各 P 节点的 bootstrap 端口必须唯一**，否则建链冲突。
3. **P/D 环境变量取值常不同**（如 `GPU_MAX_HW_QUEUES`、`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`、`SGLANG_OPT_FLASHMLA_SPARSE_PREFILL`），别把一套环境照抄到 P/D 两侧。
4. **PD 分离下 `--disable-radix-cache` 常配合 `SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0`**（避免前缀缓存干扰），多模态/精度场景尤其。
5. **PD 分离的 `--disaggregation-ib-device` 必须与 `200G NDR HCA` 实际网卡名一致**（对应 `mlx5_x:1` 端口），错配会退化。
6. **PD 分离下 KV cache 传输目前只能走 IB**（RDMA），不走 TCP/Socket——组网规划时 P/D 节点间必须预留 IB 链路，否则 KV 传输成为瓶颈。

## 四·补6、GLM-5.x 版本代际、NSA→DSA/DCU→HCU 改名与官方 PD 骨架（2026-10-14 公开 fork 源码核实）

> 源头：HYGON-AI/sglang-das **main @ e295faff**（0.5.18 官方公开线）+ **inference-cookbook-das**（官方模型部署文档库）。
> ⚠ 本 skill 主体基于 0.5.12 海光 fork（见顶部"版本演进对照表"）。**下面给出"同一机制在不同代的准确名字"**，避免旧文档（大量还是 NSA 名）与新版（DSA 名）混用。

### 6.1 版本代际特征（0.5.12 → 0.5.15 → 0.5.18）

| 代际 | 术语 | GLM 稀疏注意力后端 | 命名 | 备注 |
|---|---|---|---|---|
| 0.5.12 海光 fork（本 skill 主体） | DCU / `is_dcu()` | `nsa` / `srt/layers/attention/nsa` | **`SGLANG_NSA_*`** | GLM-5.2 官方文档即此命名；GLM-5.3-Flash 老 fork 亦在此代 |
| 0.5.15.post2 新版 fork | **HCU** / `is_hcu()` | `hcu_mla` + 新增 `dsa/`、`nsa/` | `SGLANG_DSA_*`（旧 `NSA_*` 走 alias） | 见顶部"版本演进对照表" |
| 0.5.18 官方 main | HCU | **`glm5_next` 包 + `dsa_backend.py`** | `SGLANG_DSA_HCU_*` / `SGLANG_DSA_KPOOL_*` | GLM-5-Next 支持已并入；`is_glm5_next_hcu()` |

> **改名机制**：官方用 `EnvBoolWithAlias(FALSE, deprecated_name="SGLANG_NSA_*")` 给旧名保留向后兼容（`python/sglang/srt/environ.py`）。所以 **0.5.18 里填 `SGLANG_NSA_*` 仍生效**，语义上游统一为 `DSA`——文档里新旧名混用不报错，但读代码要按 `DSA`。

### 6.2 NSA→DSA / DCU→HCU 改名对照表（0.5.12 / 旧 fork 名 → 0.5.18 官方名）

| 旧名 | 0.5.18 官方同名 | 备注 |
|---|---|---|
| `SGLANG_NSA_FUSE_TOPK` | `SGLANG_DSA_FUSE_TOPK` | decode |
| `SGLANG_NSA_KPOOL_AITER_TOPK` / `SGLANG_NSA_KPOOL_LIGHTOP_TOPK` | `SGLANG_DSA_KPOOL_AITER_TOPK` / `SGLANG_DSA_KPOOL_LIGHTOP_TOPK` | 默认 False |
| `SGLANG_NSA_DCU_USE_LIGHTOP_DECODE_GATHER` | `SGLANG_DSA_HCU_USE_LIGHTOP_DECODE_GATHER` | 默认 True |
| `SGLANG_NSA_DCU_MQA_LOGITS_WORKSPACE_GB` | `SGLANG_DSA_HCU_MQA_LOGITS_WORKSPACE_GB` | 默认 2.0 |
| `SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB` | **未改名（保留）** | `SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION` 为新加 |
| `SGLANG_NSA_HCU_REUSE_SORTED_TOPK` / `SGLANG_NSA_HCU_INT8_INDEX_K_CACHE` | `SGLANG_DSA_HCU_*` | alias 兼容 |
| `--enable-nsa-cache-layer-split` | `--enable-dsa-cache-layer-split` | 仅 DSA 模型 + 需 PD |
| `--nsa-prefill-cp-mode round-robin-split` | `--enable-prefill-cp --cp-strategy interleave`（字段 `dsa_prefill_cp_mode`） | 官方 GLM-5.3 PD 正式写法 |
| `USE_DCU_CUSTOM_ALLREDUCE` | `USE_HCU_CUSTOM_ALLREDUCE`（官方 cookbook 名） | 0.5.12 fork 仍可用 DCU 名 |

### 6.3 官方 GLM 稀疏注意力 kernel 控制块全集（0.5.18 `environ.py` "HCU GLM5-Next kernel controls"）

`SGLANG_USE_LIGHTOP_PREFILL_DEQUANT`(True)、`SGLANG_USE_FUSED_METADATA_COPY`(True)、`SGLANG_DSA_HCU_USE_LIGHTOP_DECODE_GATHER`(True)、`SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA`(True)、`SGLANG_DSA_KPOOL_AITER_TOPK`(False)、`SGLANG_DSA_HCU_MQA_LOGITS_WORKSPACE_GB`(2.0)、`SGLANG_DSA_HCU_USE_TRITON_PAGED_MQA`(False)、`SGLANG_DSA_HCU_USE_INT8_MQA_LOGITS`(True)、`SGLANG_DSA_KPOOL_LIGHTOP_TOPK`(False)、`SGLANG_MOE_ROUTER_USE_CONFIG_DTYPE`(False)、`SGLANG_GLM5_NEXT_FUSE_QKVBFG`(False)、`SGLANG_DSA_CP_FUSE_SYMM_MEM`(False)。
另有 `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`(2048)、`SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION`(0.2)、`SGLANG_DSA_TOPK_BROADCAST`、`SGLANG_DISABLE_DSA_INDEXER_FUSION`、`SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK`（alias `SGLANG_USE_LIGHTOP_MASK_TOPK`；gfx936-only INT8 索引 K 缓存另见 `SGLANG_DSA_HCU_INT8_INDEX_K_CACHE`）。调 DSA/NSA decode 与 topk 时按需打开。

### 6.4 KDA/Mamba（线性注意力）在 DCU 上的投机解码与 dtype 约束

- **KDA/SSM 投机解码状态回放**：draft 步 SSM 状态写入 ring，**commit 时只把被接受前缀精确折叠（exact fold）回已提交状态**。HCU 专用 kernel：`kernels/ops/attention/fla/hcu/kda_replayssm_spec_decode.py`；泛化 flag：`--enable-linear-replayssm` / `--enable-linear-replayssm-spec`（旧 `--enable-kda-replayssm-spec`）。
- **`--mamba-ssm-dtype`**：混合模型（linear attention + gated SSM）在 W4A8 量化场景下 **P/D 两侧须一致**（`arg_groups/attention_hook.py` 有校验），W4A8 场景官方/GLM 项目均锁 `float32`；不一致会导致精度退化。
- 相关 kernel：`fused_recurrent_linear_replayssm.py`、`kda_replayssm_spec_decode.py`、`gdn_replayssm_spec_decode.py`、`kda_decode_mtp.py`（MTP 与 replayssm 共用）。

### 6.5 官方 PD 分离骨架（跨模型通用，非 GLM 专属）

> 出处：`README_HCU.md` PD 段 + `inference-cookbook-das` `docs/model-deployment/sglang/glm-5.3.md`。**给结构不给定值**——具体 mem-fraction/并发/EAGLE 步数按模型与卡型查对应官方文档。

- **P**：`--deepep-mode normal` + `--enable-prefill-cp --cp-strategy interleave`（`--attn-cp-size` 取 P 的 CP 数）+ `--enable-dsa-cache-layer-split`（PD 缓存分层）。
- **D**：`--deepep-mode low_latency` + `--enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1`；EAGLE/MTP 在这侧（参数须显式给，见 §四·补5）。
- **Router**：`sglang_router.launch_router --pd-disaggregation --policy round_robin`（多 P 副本必须 round_robin + 唯一 bootstrap 端口）。
- **DeepEP low_latency 官方基线**：`ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288`、`ROCSHMEM_HEAP_SIZE`≈3G；`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 官方示例 128。`DEEPEP_ENABLE_LL_LAYERED_OPT` 与 `DEEPEP_ENABLE_LL_DISPATCH_OPT` 是**两个并存**的低延迟优化开关（消费方在 deep_ep 包内）。
- **Mooncake PD**：`MC_TOPO_FILE_FORCE=<topo>`、`MC_ALLOWED_IBV_DEVICES`、`MC_IB_GID_INDEX`（RoCE 需设 GID）；`SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`/`WAITING_TIMEOUT`=1200。
- **flash_mla 稀疏 decode 三件套**：`FLASH_MLA_SPARSE_PAGE64_SPECIALIZE=1`、`FLASH_MLA_SPARSE_DECODE_NUM_SM_PARTS`、`FLASH_MLA_SPARSE_DECODE_DISABLE_SPLITKV=0`。

### 四·补7、GEMM launch-config 查表注入（best_config）适配点（2026-10-14 K100-AI 实测，SGLang 0.5.12 + Qwen3.8-27B w8a8）

SGLang 系（非 vLLM）在 DCU 上做 INT8 GEMM / linear-attention 的**查表传 best_config**注入点索引。

- **目标文件**：`srt/layers/quantization/w8a8_int8.py`（stock 查表 + `quant_ops.triton_scaled_mm`）+ `srt/layers/attention/fla/wy_fast.py`（GDN `recompute_w_u`）+ aiter `ops/triton/configs/`（`chunk_fwd_o` / `chunk_gated_delta_rule_fwd_h` 走 `AITER_TRITON_CONFIGS_PATH` JSON 查表）。
- **机制**：lightop/stock 的 decode/prefill GEMM 用**固定 num_stages/num_warps**（小 M decode 无软件流水线、大 M prefill 用 `num_stages=0`）。注入一张按 `(M,K,N)` 查表的 best_config（BK/BN/BK/GROUP/SPLIT_K/num_stages/num_warps），命中才用、未列回退 stock（fail-closed）。
- **结构性安全**：不改算子数学、不加显存副本、无 int8 布局/失步副作用 → 精度 0pp（与 int8-lm_head 这类改数值路径/布局是本质区别）。
- **env 门控纪律**：`SGLANG_INT8_TUNE`（decode M≤128）/ `SGLANG_INT8_PREFILL_TUNE`（prefill M≥129）/ `SGLANG_GDN_TUNE`（GDN chunked）各独立、fail-closed（env 未设 = 完全 stock），可单独 A/B。
- **补丁重放**：`patch -p1 -N --forward < ${PATCH_DIR}/*.patch`（幂等，重跑 skip），run.sh 按 glob 字母序 0003→0006→0007→0008（0008 上下文依赖 0006 已应用）。⚠ `patch ... || true` 会**吞失败**——必须用启动日志 `[q38 int8-tune] ACTIVE` 行证明补丁确实生效，否则静默 no-op。

## 五、组合引用

- **sglang-router（独立包，DCU 适配）**：`sglang_router` 是 SGLang 的**请求路由/负载均衡器**（Rust 核心 `sglang_router_rs.abi3.so` + Python 封装），源里 `0.3.2+das.dtk2604.torch2100`（release/dtk2604）。它是**纯 CPU 路由**（.so 无 DTK/hip 直接依赖，ldd 实测），DCU 适配主要是构建对齐（torch 标记），不涉及 GPU kernel。提供 `Router`（`router.py`）+ 多种 policy（random/round_robin/cache_aware/power_of_two）+ 控制面（JWT/Redis/Postgres/Oracle 配置）。多实例 SGLang 部署做 PD 分离/负载均衡时用，与 `dcu-mooncake-source`（KV 传输）配合。
- 算子内核怎么写（HIP/TileLang）→ `dcu-kernel-dev`。
- vLLM 自定义算子开发（对照 SGLang 的差异）→ `dcu-vllm-custom-op`。
- 优化闭环/性能度量 → `dcu-op-optimization`、`dcu-perf-profiling`。
- **单请求 SGLang profiling + Perfetto 读 trace（prefill/decode 拆 trace、roofline 预判、读图三步法、batch=1 带宽受限归因）**：方法见 `dcu-perf-profiling` 对应节（源自 SGLang 官方博客，NVIDIA L4+Qwen3.5-0.8B；**DCU 上内核名/HBM 带宽需重测，不抄 L4 结论**）。海光 fork 同样支持 `/start_profile` 端点，DCU 真机剖析可直接复用该流程。
- vLLM 源码适配 → `dcu-vllm-source`（上游复用版）。

## 使用须知

- 经验/方法仅供参考，不保证修复。先核对设备型号(gfx)+驱动/DTK版本+SGLang版本+报错原文。
- 海光 SGLang 为 **fork 构建**（版本号带 `+<commit>`），上游 tag 与海光 fork 路径可能不同，改前先 `git`/版本确认。
- 不存性能数据/基线；数字用 `dcu-op-optimization` 模板留痕。
- 高危操作谨慎，优先测试机验证。
- 平台判断认准 `is_dcu()`（非 vLLM 的 `is_rocm`）；prefill/decode 可分相位选 backend。
