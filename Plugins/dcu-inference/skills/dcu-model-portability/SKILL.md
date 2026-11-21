---
name: dcu-model-portability
description: "评估任意PyTorch模型能否即插即用部署到海光DCU并产出部署手册。"
version: 1.0.0
author: DCU Engineer
license: MIT
---

# DCU 模型适配可行性评估与部署手册（dcu-model-portability）

本 skill 解决一类高频任务：**把某个开源 PyTorch 模型适配 / 部署到海光 DCU，并产出一份
《海光 DCU 模型部署手册》**。核心是先做「可行性分流」再动手，避免对即插即用模型过度工程、
或对需 kernel 移植的模型估计不足。

> 与 `dcu-inference-infra` 的关系：本 skill 聚焦「新模型从零评估 + 产出部署手册」的方法学；
> 若确认需要 kernel 移植，转 `dcu-vllm-op-replace` / `dcu-pytorch-source` / `dcu-op-optimization`。
> 二者分工：`dcu-inference-infra` 是推理域总入口路由器，本 skill 是其中「新模型评估」的具体方法承载。

## 一、可行性分流判定树（第一步，必做）

1. **是否含自定义 CUDA 扩展（.cu / 自定义 aten 算子）？**
   - 查仓库：`*.cu` 文件、`setup.py` 里 `CUDAExtension` / `cpp_extension.load()`、
     依赖 `xformers` / 自定义 `triton` kernel / `flash-attn` / `deepspeed-ops` 等需编译的扩展。
   - **无 .cu 扩展、仅用标准 torch 算子** → 走「即插即用」路径（下方第二节），无需移植 kernel。
   - **有 .cu 扩展** → 走 kernel 移植 / 重编译路径（转 `dcu-vllm-op-replace` / `dcu-pytorch-source`）。
2. **非深度学习依赖是否纯 Python/CPU 库？**（rdkit / ase / lmdb / sklearn / scipy / tqdm 等）
   - 与硬件后端无关，`pip install` 即可；唯一常见坑：**rdkit 在 cp310 上装 `rdkit-pypi` 而非 `rdkit`**。
3. **权重 / 数据是否需出网下载？**（HF / ModelScope）
   - 内网环境需提前从可出网机下载后拷贝，或配镜像源。

## 二、即插即用路径（无 .cu 扩展的模型）

DCU 的 HIP 把 `torch.cuda.*` 全部映射到 DCU（`torch.cuda.is_available()`→True，
`device='cuda'`→实际落 DCU）。只用标准算子的模型**不需要**移植 CUDA kernel。适配动作收敛为：

1. 装**海光私有源带 `+das.dtk2604` 的 torch wheel**（绝不能装上游原生 torch，无 DCU 后端）；
2. 装齐纯 CPU 依赖；容器必挂 `-v /opt/hyhal:/opt/hyhal:ro`、运行前 `source /opt/dtk/env.sh`；
3. 模型代码里 `device='cuda'` 原样保留，无需改代码。

最小校验见 `references/model-portability-assessment.md` 第二节。

## 三、产出物：DCU 模型部署手册（标准结构）

此类任务最终交付结构统一的部署手册，模板见 `references/dcu-model-deploy-manual-template.md`，
直接照填即可（含概述 / 环境规格 / 可行性分析 / 部署步骤 / 验证 / 排障表 / 性能建议 / 附录）。
与《海光DCU模型部署手册——Qwen3-VL-Embedding-2B》《——Uni-Mol》等保持同构。

## 四、实例：Uni-Mol（deepmodeling）—— 即插即用类（**2026-08-06 实测通过**）

> 以下为 <device-B> 设备（<<internal-ip>>，8×BW100/gfx936，DTK 26.04，容器 <container_vllm015_rebuild>，
> torch 2.9.0）真机实测确认，非推测。

- 架构：SE(3) 等变 Transformer（分子表征 / 性质预测 / 构象生成）。
- 计算：全部建立在标准 PyTorch 算子，**不含 .cu 扩展** → 无 kernel 移植工作量（实测确认）。
- **包名是 `unimol_tools`**（PyPI 稳定版，实测装到 0.1.6），**不是** `unimol` / `unimol-models`；
  推理 API：`from unimol_tools import UniMolRepr, MolTrain, MolPredict`（表征 / 训练 / 预测）。
- 非深度学习依赖：rdkit、pandas(<2)、numpy(<2)、scikit-learn、joblib、numba、tqdm、huggingface_hub。
- 权重：走 HuggingFace `dptech/Uni-Mol-Models`，运行时自动下载（与设备无关）。
- 适配动作：装对 DCU torch + 装齐 CPU 依赖 + 挂 hyhal + source DTK。
- **实测关键踩坑（必须照做，否则跑不起来）**：
  1. **卸载 `cupy-cuda12x` + 锁 `numpy==1.26.4`**：<device-B> 容器原本残留 CUDA 版 `cupy-cuda12x`，
     它与 numpy 2.x 冲突并连带破坏 torch 的 numpy 桥（`Failed to initialize NumPy: _ARRAY_API not found`）。
     DCU 用不到该 CUDA 包，必须 `pip uninstall -y cupy-cuda12x` 并锁 numpy<2.0。
  2. **权重下载关 Xet**：HF 新版默认 Xet 分块下载对镜像站 401。必须
     `export HF_HUB_DISABLE_XET=1` + `HF_ENDPOINT=https://hf-mirror.com`（国内镜像），实测可下。
  3. **性质预测输入用 CSV 文件**（SMILES 列 + TARGET 列），不要直接传 `{smiles,target}` dict
     （unimol_tools 0.1.6 的 DataHub 对 dict 直接传 smiles 解析有 bug，会取到 None 报
     `'NoneType' object is not iterable`）。
  4. **回归指标名用 `mae` / `mse`**，不是 `rmse`（0.1.6 报 `Unknown metric: rmse`）。
- **实测结果**：① 设备可见 `device='cuda'`→BW100；② `UniMolRepr` 推理 3 分子各得 512 维 CLS 表征
  （cls_repr (3,512)，单卡 5.27s）；③ `MolTrain` 单卡 DCU 训练+评估产出 mae/mse/pearsonr/r2。
- 完整手册（含真实 stdout）：原存本机 `/root/Uni-Mol_DCU部署手册.md`（非仓库内）；仓库内方法细节见 `dcu-model-deploy-verify/references/unimol-dcu-deployment.md`；
  方法细节见 `references/model-portability-assessment.md` 第三节。

## 五、通用纪律（复用 DCU 系列）

- **部署手册必须真机实测通过才算数（用户 2026-08 硬性要求）**：未在上目标 DCU 设备实跑跑通前，
  产物只能叫「方案草稿」，不准命名为「部署手册」。实测要覆盖设备可见性校验 + 至少一类推理/训练真实跑通。
- **手册必须含可复现代码，不能只贴结果（用户 2026-08 硬性要求）**：每个步骤给出可直接复制执行的
  完整命令；验证章节给出最小可跑脚本全文（照抄即跑）+ 真机真实 stdout 节选；并强制内嵌
  「一键复现脚本」章节，该脚本须先在目标设备完整跑通（结尾 ALL DONE / 全段 PASS）再定稿。
  严禁声称"脚本可用"而未实跑——写脚本先 `bash -n`，上机报错（依赖回弹、API 签名变化等）
  必须修掉再交付，不要带已知崩点交付。
- 测试日志 / 性能基线不写入 skill；手册如引实测数据须与「观察」分离、原样保留异常。
- 容器必挂 `/opt/hyhal:/opt/hyhal:ro`；gfx 用物理机 `rocminfo`；指定卡用 `HIP_VISIBLE_DEVICES`。
- 撰写时若无法联网核对仓库最新 `requirements.txt` / API，须在手册「待复核」节列出 curl 复核命令，
  不可把旧认知写死为确定内容。
- 版本绑定矩阵（私有源 `https://pypi.sourcefind.cn/release/dtk<ver>/`，注意不是 `/simple/`）见 `dcu-env-packages`。
