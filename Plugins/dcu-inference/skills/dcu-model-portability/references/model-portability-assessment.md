# 模型 DCU 适配可行性评估（方法 + 实例）

> 配套 `dcu-model-portability` SKILL.md。给出判定方法，并用 Uni-Mol 作为「即插即用」类实例。

## 一、判定方法（通用）

适配一个开源 PyTorch 模型前，按以下顺序判定工作量：

1. **查 .cu / 自定义 CUDA 扩展**
   - 在仓库里搜：`*.cu`、`setup.py` 中 `CUDAExtension` / `load()`、`cpp_extension`、
     依赖 `xformers`、`triton` 自定义 kernel、`flash-attn`、`deepspeed-ops` 等需编译的扩展。
   - 无 .cu 扩展 → **即插即用**（标准 torch 算子经 HIP 直接落 DCU）。
   - 有 .cu 扩展 → 需 kernel 移植/重编译，走 `dcu-vllm-op-replace` / `dcu-pytorch-source` /
     `dcu-op-optimization`。

2. **列非深度学习依赖**
   - 纯 Python/CPU 库（rdkit、ase、lmdb、scikit-learn、scipy、tqdm 等）与硬件后端无关，
     `pip install` 即可。唯一常见坑：**rdkit 在 cp310 上装 `rdkit-pypi` 而非 `rdkit`**。

3. **权重 / 数据来源**
   - HF / ModelScope 自动下载 → 内网需提前备离线包或镜像源。

4. **代码里 device 处理**
   - 即插即用模型通常写 `device='cuda'`，DCU 上 HIP 自动映射，**原样保留即可**，无需改代码。

## 二、验证「即插即用」的最小校验
```python
import torch
print(torch.__version__)                 # 期望含 +das.dtk2604
print(torch.cuda.is_available())         # DCU 上应为 True
print(torch.cuda.get_device_name(0))     # 期望显示 DCU / BW100
x = torch.randn(4, 4, device="cuda")
print(x.device)                          # 期望 cuda:0（即落到 DCU）
```

## 三、实例：Uni-Mol（deepmodeling）—— 即插即用类（**2026-08-06 <device-B> 真机实测通过**）

- 架构：SE(3) 等变 Transformer（分子表征 / 性质预测 / 构象生成）。
- 计算：全部建立在标准 PyTorch 算子（matmul / einsum / scatter / gather / layernorm），
  **不含自定义 CUDA(.cu) 扩展** → 无 kernel 移植工作量（实测确认）。
- **包名是 `unimol_tools`**（PyPI 稳定版，实测 0.1.6），**不是** `unimol` / `unimol-models`。
  推理 API：`from unimol_tools import UniMolRepr, MolTrain, MolPredict`。
  官方安装 `pip install unimol_tools`（PyPI）；权重经 HuggingFace `dptech/Uni-Mol-Models` 自动下载。
- 非深度学习依赖（实测）：rdkit、pandas(<2)、numpy(<2)、scikit-learn、joblib、numba、tqdm、huggingface_hub。
- 适配动作：装对 DCU torch + 装齐 CPU 依赖 + 挂 hyhal + source DTK；代码里 `device='cuda'` 原样保留。

### 3.1 实测环境基线（<device-B> 设备）
- 设备 <<internal-ip>>，8×BW100(gfx936)，驱动 rock 6.3.30-V1.4.1a（DTK 26.04 配套）。
- 容器 `<container_vllm015_rebuild>`（torch 2.9.0，DTK 26.04）。若从零起容器，已核实 <device-B> 存在
  `<internal_registry>/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10`。
- 实测验证覆盖：设备可见性（device='cuda'→BW100）、UniMolRepr 推理（cls_repr (3,512)，单卡 5.27s）、
  MolTrain 单卡训练+评估（产出 mae/mse/pearsonr/r2）。

### 3.2 实测踩坑（关键，必须照做）
1. **卸载 `cupy-cuda12x` + 锁 `numpy==1.26.4`**：<device-B> 容器原本残留 CUDA 版 `cupy-cuda12x`（DCU 用不到），
   它与 numpy 2.x 冲突并连带破坏 torch 的 numpy 桥（报错 `Failed to initialize NumPy: _ARRAY_API not found`，
   torch 直接 import 失败）。必须 `pip uninstall -y cupy-cuda12x` 后锁 `numpy==1.26.4`、`pandas==1.5.3`。
   > 注：unimol_tools 0.1.6 的 setup.py 声明要 numpy>=2.0，但那是上游声明；在 DCU 容器里 numpy 2.x
   > 会被 cupy 冲突拖垮，**实测 numpy 1.26.4 才是可用态**。
2. **权重下载关 Xet**：HF 新版默认 Xet 分块下载对镜像站返回 401
   （`CAS Client Error: HTTP status 401 ... xethub.hf.co`）。必须
   `export HF_HUB_DISABLE_XET=1` + `HF_ENDPOINT=https://hf-mirror.com`（国内镜像），实测可下。
3. **性质预测输入用 CSV 文件**（SMILES 列 + TARGET 列），不要直接传 `{smiles,target}` dict。
   unimol_tools 0.1.6 的 `DataHub` 对 dict 直接传 smiles 解析有 bug，会取到 None，
   报 `'NoneType' object is not iterable`。CSV 格式：`SMILES,TARGET` 两列。
4. **回归指标名用 `mae` / `mse`**，不是 `rmse`（0.1.6 报 `Unknown metric: rmse`）。

### 3.3 实测安装命令（<device-B> 容器，已跑通）
```bash
source /opt/dtk/env.sh
pip uninstall -y cupy-cuda12x
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "numpy==1.26.4" "pandas==1.5.3" "numba==0.60.0"
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  rdkit joblib pyyaml addict scikit-learn tqdm huggingface_hub
pip install unimol_tools --upgrade          # 实测到 0.1.6
# 运行前：export HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com HIP_VISIBLE_DEVICES=0
```

### 3.4 最小验证脚本要点
```python
import torch
from unimol_tools import UniMolRepr
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))   # True, BW100
clf = UniMolRepr(data_type="molecule", remove_hs=False)
rep = clf.get_repr(["CCO","CC(=O)O"], return_atomic_reprs=True)
print(__import__("numpy").array(rep["cls_repr"]).shape)            # (2, 512)
```
- 完整手册（含真实 stdout + 可复现命令/脚本 + 一键复现脚本全文）：/root/Uni-Mol_DCU部署手册.md
  （本机）。一键脚本已在 <device-B> 真机从头完整跑通（输出 `ALL DONE: Uni-Mol on DCU reproduction OK`），
  涵盖装依赖→设备校验→UniMolRepr 推理→MolTrain+MolPredict 训练预测，四段全 PASS。

## 四、反例（需 kernel 移植，不在即插即用范畴）
凡含 `*.cu` 或需编译 aten 算子的模型（部分检测/分割/ MoE 推理框架），不在此快速路径，
须走 `dcu-vllm-op-replace`（自定义算子替换）或 `dcu-pytorch-source`（低版本 torch 源码编译）。
