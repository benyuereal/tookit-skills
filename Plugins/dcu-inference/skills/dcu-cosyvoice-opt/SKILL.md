---
name: dcu-cosyvoice-opt
description: "CosyVoice2(0.5B TTS)在DCU上的部署与推理优化"
version: 1.0.0
author: DCU Engineer
license: MIT
---

# CosyVoice2 TTS 部署与优化（dcu-cosyvoice-opt）

> 定位：CosyVoice / CosyVoice2（阿里开源 TTS，0.5B）在 DCU 上的**部署踩坑 + 推理优化**沉淀。架构深度解读与源码片段见 `references/source_patches.md`（溯源 file_id=<internal-file-id>）。
>
> 三大优化（基于 BW1000，作者<internal-author>/部署总结）：Opt1 CFM 推理步数 10→5、Opt2 NUMA 绑定、Opt3 vLLM 显存/批处理深度优化。基线 RTF=0.2871/FTL=1.5628 → 优化后 RTF=0.2100/FTL=1.109324（RTF↓约26%、FTL↓约30%）；叠加部署总结的 fp16.half + numactl 后 OFFLINE RTF 0.5577→0.3830、STREAMING RTF 0.7029→0.4497（均↓约30%+）。

---

## 一、部署环境

- 源码：`https://github.com/FunAudioLLM/CosyVoice`
- 测试镜像（已验证可用）：`<internal_registry>/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk25.04.2-1226-das1.7-py3.10-20251226`（别的 092 镜像也可；0110 的 vllm 有问题）。
- 容器启动（必挂 `-v /opt/hyhal:/opt/hyhal:ro`，否则 DCU 不可见）：见 `dcu-docker` / `dcu-env`。示例镜像 `vllm:0.9.2-...-dtk26.04-0130-py3.10-20260202`。
- 依赖：原生 requirements.txt 含 NV 相关库，DCU 上需替换/升级（pydantic、onnxruntime 海光轮）。关键：`pip install ./onnxruntime-1.19.2+das.opt1.dtk2604.torch251-cp310-cp310-manylinux_2_28_x86_64.whl`、`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`。
- cosyvoice3：基于 cosyvoice2 环境，额外 `pip install Matcha-TTS x_transformers`。

---

## 二、DCU 部署踩坑（必看）

1. **pydantic / pydantic-core 版本不兼容**（vLLM 与 pydantic 冲突，`TypeError: 'str' object cannot be converted to 'PyType'`）→
   `pip uninstall -y pydantic pydantic-core && pip install "pydantic==2.10.6" "pydantic-core==2.27.2"`
2. **Tokenizer 加载失败**（`TypeError: expected str... not NoneType`，子目录缺 tokenizer.json/vocab.json）→ `modelscope download --model aiwantaozi/CosyVoice2-0.5B CosyVoice-BlankEN/tokenizer.json --local_dir ./` 重下缺失文件。
3. **libonnxruntime_providers_rocm.so 加载失败**（`librocm_smi64.so.1: cannot open shared object file`）→ ROCm lib 目录加 LD_LIBRARY_PATH 并补软链：
   ```bash
   export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:$LD_LIBRARY_PATH
   cd /opt/dtk-26.04/.hyhal/rocm_smi/lib
   ln -sf librocm_smi64.so librocm_smi64.so.1
   ```
4. **VLLM_USE_V1=1 不支持 `--enable-prompt-embeds`**（`NotImplementedError`）→ `VLLM_USE_V1=0 uvicorn cosyvoice_server:app ...` 退回 V0 引擎。
5. **端口被占用**（`address already in use`）→ 换端口，客户端同步换。
6. **load_vllm=True 报错（ModelRegistry 未注册）** → 在 cosyvoice_server.py 顶部加：
   ```python
   from vllm import ModelRegistry
   from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
   ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
   ```
7. **客户 benchmark 脚本启动报错**（`activated_builtin_plugins`）→ 编辑 `vllm/platforms/__init__.py` 约 229 行加 `activated_builtin_plugins = ['rocm']`，并 `pip install --upgrade pydantic pydantic-core`。

---

## 三、三处源码级优化

### Opt1：CFM 推理步数优化（降计算量）
- 位置：`cosyvoice/flow/flow.py` 的 `self.decoder(...)` 调用，把 `n_timesteps` 从 **10 改为 5**。
- 两类（MaskedDiffWithXvec 非流式 / CausalMaskedDiffWithXvec 流式 / CausalMaskedDiffWithDiT DiT）均改。
- 效果（单独 Opt1）：RTF=0.225048，FTL=1.187064。
- 完整类实现见 `references/source_patches.md`（flow.py 段）。

### Opt2：NUMA 绑定 CPU 与内存
- 多 NUMA 节点下跨节点访问有损耗，`numactl` 绑核心节点与内存：
  ```bash
  VLLM_USE_V1=0 numactl --cpunodebind=0 --membind=0 uvicorn cosyvoice_server:app --host 0.0.0.0 --port 8000 --workers 1
  ```
- 效果（单独 Opt2）：RTF=0.266048，FTL=1.512107（注：本项单独测反而略升，需与 Opt1/Opt3 组合才收益；NUMA 收益依赖实际拓扑，先用 `dcu-topology` 查）。

### Opt3：vLLM 推理调度与显存深度优化
- 位置：`cosyvoice/cli/model.py` 的 `EngineArgs(...)`，关键参数：
  ```python
  engine_args = EngineArgs(
      model=model_dir, skip_tokenizer_init=True, enable_prompt_embeds=True,
      gpu_memory_utilization=0.80,      # 实测 0.9 会 OOM
      max_num_batched_tokens=4096, max_num_seqs=32,
      enable_chunked_prefill=True, max_seq_len_to_capture=4096,
      disable_log_stats=True, trust_remote_code=True)
  ```
- 另外两处 fp16 修正（CosyVoiceModel.init / CosyVoice2Model.init）：若 `fp16=True` 则 `self.llm.half(); self.flow.half()`。
- 效果（与 Opt1/Opt2 组合 + fp16.half + numactl，部署总结实测）：OFFLINE RTF 0.5577→0.3830（↓31.3%）、STREAMING RTF 0.7029→0.4497（↓36.0%）、首包延迟同步↓约 30%+。

### 组合结论
- RTF/FTL 指标定义：**RTF**（实时率）= 处理耗时/音频时长，越小越快；**FTL**（首包延迟）= 请求到首段语音输出时间，越小越低。
- 调 GPU 利用率（`gpu_memory_utilization` 0.2→0.35）实测 RTF/FTL 提升不稳定、无特别大影响 → 不必盲目调高。
- 原生代码两处修正（部署总结/C 文档）：`load_vllm` 高版本 vLLM 的 `paged_alignment/device_type` 参数不兼容需移除；流式 RTF 统计逻辑错误（实际算成非流式）需重定义计时节点。

---

## 四、RTF/FTL 实测结论（定性 + 溯源）

> 原始 RTF/FTL 性能汇总表来自 内部文档《Cosyvoice2-0.5B vllm加速RTF和FTL性能测试》(file_id=<internal-file-id>)。3090 为客户环境自测，其余内部环境。按 skill 内容纪律，**具体 RTF/FTL 数字不入本 skill**（如需引用转 `dcu-perf-cases` 或溯源上述 内部文档）；此处只保留关键结论。

**关键结论**：BW1000 开 vllm 加速后 RTF 从 ~1.2 降到 ~0.36（长文本），**不开 vllm 的 BW1000 比开 vllm 慢 3-4 倍**——vLLM 加速对 CosyVoice2 是必选项。BW1000(vllm) 与 4090(vllm) 仍有约 2x 差距（长文本 RTF 0.277 vs 0.120），属硬件代差。

---

## 五、交叉引用

- 通用推理优化（torch.compile/NHWC/TF32/MIOpen/rocBLAS）→ `dcu-inference-optimization`。
- vLLM 部署/参数 → `dcu-vllm-deploy`；NUMA 拓扑查询 → `dcu-topology`；DTK/镜像 → `dcu-dtk`、`dcu-docker`。
- 源码片段/架构解读完整版 → `references/source_patches.md`（溯源 file_id=<internal-file-id>）。

## 使用须知
- 经验/方法仅供参考，不保证修复。先核对设备型号(gfx)+驱动/DTK 版本+报错原文。
- 性能数字以源文档/实测为准；本页数据表为作者公开测试，仅供方法佐证。
- 高危操作谨慎，优先测试机验证。
