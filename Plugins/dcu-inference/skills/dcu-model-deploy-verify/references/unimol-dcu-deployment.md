# Uni-Mol 在 DCU（DTK 26.04）适配实测

> 知识库（condensed）：Uni-Mol 在海光 DCU 上部署的实测经验，供后续同类「纯 PyTorch 科学计算模型上 DCU」任务复用。
> 实测环境：<device-B> 设备 <<internal-ip>> / 容器 <container_vllm015_rebuild> / DTK 26.04 / torch 2.9.0 / 8×BW100(gfx936)。
> 完整手册：/root/Uni-Mol_DCU部署手册.md。本文件只沉淀方法学与踩坑，不存性能基线/日志。

## 适配结论
- Uni-Mol 全部建立在标准 PyTorch 算子，**无自定义 CUDA(.cu) 扩展** → DCU 上无需 kernel 移植；HIP 把 `torch.cuda.*` 映射到 DCU，直接跑。
- 这是「纯标准 PyTorch 算子模型」上 DCU 的典型范式：适配工作量 = 装对 DCU torch + 装齐 CPU 依赖 + 挂载 hyhal，**不碰算子**。
- 核心包名是 **`unimol_tools`**（PyPI 稳定版，实测 0.1.6），不是 `unimol` / `unimol-models`。
- 预训练权重走 HuggingFace `dptech/Uni-Mol-Models`，运行时自动下载。

## 安装要点（实测）
- 卸 `cupy-cuda12x`（CUDA 包，与 numpy 2.x 冲突并破坏 torch numpy 桥）→ 锁 `numpy==1.26.4` + `pandas==1.5.3` + `numba==0.60.0`（详见 `dcu-env-packages` 排障第 7 条）。
- `pip install -i 清华源 rdkit joblib pyyaml addict scikit-learn tqdm huggingface_hub` + `pip install unimol_tools --upgrade`。
- 权重下载必须：`export HF_HUB_DISABLE_XET=1`（关 Xet，否则镜像站 401 `cas-server.xethub.hf.co`）+ `HF_ENDPOINT=https://hf-mirror.com`。

## 推理/训练 API（实测可用）
- 表征：`from unimol_tools import UniMolRepr; clf=UniMolRepr(data_type='molecule'); clf.get_repr(smiles_list, return_atomic_reprs=True)` → `cls_repr` (N,512)；`atomic_reprs` 变长 list，每元素 (n_atoms,512)，**不能**直接 `np.array()`（inhomogeneous shape）。
- 性质预测：`from unimol_tools import MolTrain; clf=MolTrain(task='regression', metrics='mae'); clf.fit(data='<csv路径>')`。
  - **输入用 CSV 文件（列名 `SMILES` + `TARGET`）**，不要直接传 `{smiles,target}` dict——0.1.6 的 `DataHub` 对 dict 直接传 smiles 解析会取到 None，报 `'NoneType' object is not iterable`。
  - 回归指标名用 `mae` / `mse`，**不是** `rmse`（0.1.6 报 `Unknown metric: rmse`）。

## 验证结果（实测，非推测）
- 设备可见性：`cuda_avail=True`，`device name: BW100`，张量落 `cuda:0` ✓
- UniMolRepr 推理：权重下载 53.8s，构象生成 100%，单卡 DCU 推理 5.27s，cls_repr (3,512) ✓
- MolTrain 训练：单卡 DCU 训练+评估，产出 mae/mse/pearsonr/r2 ✓

## 排障速查
| 现象 | 处理 |
|---|---|
| `Failed to initialize NumPy: _ARRAY_API not found` | 卸 `cupy-cuda12x` + 锁 `numpy==1.26.4` |
| 权重下载 `401 ... xethub.hf.co` | `HF_HUB_DISABLE_XET=1` + `HF_ENDPOINT=https://hf-mirror.com` |
| `MolTrain.fit` `NoneType not iterable` | 改用 CSV 文件输入，勿传 dict |
| `Unknown metric: rmse` | 改 `metrics='mae'` |
| `cuda_avail=False` / DEVCOUNT=0 | 容器加 `-v /opt/hyhal:/opt/hyhal:ro` 重启 |

## 可复用范式（给其他纯 PyTorch 科学模型）
1. 先确认模型是否含 `.cu` 扩展（无则免移植，直接 torch.cuda 路径）。
2. 装 DCU torch（`+das.dtkXXXX`）→ 装 CPU 依赖 → 查是否有 CUDA 专用包（cupy 等）需卸载。
3. 权重下载类（HF/ModelScope）一律准备镜像源 + 关 Xet。
4. 上设备跑「设备可见性 → 最小推理 → 训练/评估」三步，把真实 stdout 回填文档。
