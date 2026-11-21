# DCU 模型部署手册模板（通用结构）

> 用途：产出「海光 DCU 模型部署手册——<模型名>」时按本结构填写，保持与既有手册同构。
> 适配判定方法见 `references/model-portability-assessment.md`。

---

## 1. 概述
- 模型是什么（任务类型、架构要点，1–2 段）。
- 与 DCU 适配强相关的特征：是否含自定义 CUDA(.cu) 扩展、是否仅用标准 torch 算子、
  非深度学习依赖有哪些（rdkit/ase/lmdb/sklearn/scipy 等纯 CPU 库）。
- 结论：在 DCU 上属「即插即用」还是「需 kernel 移植」，及主要风险点。

## 2. 环境规格与目标基线
| 项 | 规格 / 取值 |
|---|---|
| 硬件设备 | <IP / 卡型 / 卡数，如 <device-B> 设备 <<internal-ip>>，8×BW100(gfx936)> |
| DTK 版本 | <如 26.04，硬要求> |
| 驱动 | <rock ≥ x.y.z-Va，DTK 配套，hy-smi -a 确认> |
| 基础镜像 | <<internal_registry>/dcu/admin/base/...:dtk<ver>-py3.10> |
| Python | <3.10，cp310，与私有源 wheel cp 标签一致> |
| torch | <如 2.9.0+das.opt1.dtk2604.*，海光私有源，非上游原生> |
| 容器参考 | <现有可复用容器名，若有> |

> 版本绑定矩阵见 `dcu-env-packages` skill（sourcefind 私有 PyPI，dtk<ver> 子目录）。

## 3. 可行性分析与适配点
- 3.1 为什么能直接跑（HIP 映射 torch.cuda.* → DCU；无 .cu 扩展则免 kernel 移植）。
- 3.2 适配动作清单（必须做：用对 +das.dtk2604 torch / 装齐 CPU 依赖 / 挂 hyhal / source DTK）。
- 3.3 风险点（rdkit 安装方式、显存、权重下载等）。

## 4. 部署步骤
- 4.1 容器与镜像：`docker pull` 基础镜像；`docker run` **必挂 `-v /opt/hyhal:/opt/hyhal:ro`**
  + `--device=/dev/kfd --device=/dev/dri`；指定卡用 `HIP_VISIBLE_DEVICES=0`。
- 4.2 容器内依赖安装：`source /opt/dtk/env.sh`；核对 torch 版本串含 `+das.dtk2604`；
  CPU 依赖走公网源（rdkit 用 `rdkit-pypi`）；DCU 适配包（如需重装）走私有源
  `https://pypi.sourcefind.cn/release/dtk<ver>/`（注意不是 `/simple/`）。
- 4.3 源码与权重获取（可选）：`git clone` + `pip install -r requirements.txt`；
  **执行前用 curl 核对仓库最新 requirements.txt 与 API**，避免按旧认知写死。

## 5. 验证（**必须真机实跑，禁止只贴结果**）
> ⚠️ 用户硬性要求（2026-08 明确）：**没有在目标 DCU 真机实测跑通，不准叫「部署手册」，只能算方案草稿**。
> 且手册里**不能只有输出结果**，必须附上别人照着能复现的完整命令/脚本源码。
- 5.1 设备可见性校验（必过）：`torch.cuda.is_available()`→True；`torch.cuda.get_device_name(0)` 显示 DCU；
  张量 `.to('cuda')` 后 `device` 为 `cuda:0`。**这一条要真跑、真打印，不是写预期值。**
- 5.2 推理/训练示例：给出**最小可跑脚本全文**（照抄即跑），并附**本次真机实跑的真实 stdout 节选**。
  脚本须与真机跑过的文件一致（建议落盘后读回，避免凭记忆重写引入差异）。
- 5.3 **一键复现脚本（强制章节）**：手册必须内嵌一个 `unimol_oneclick.sh` 式的一键脚本，
  覆盖「装依赖 → 设备校验 → 推理 →（训练/预测）」全流程。**该脚本必须在目标设备完整跑通、
  结尾打印 ALL DONE / 全段 PASS 后，才可把手册定稿**；严禁声称"脚本可用"而未实跑。
  写脚本先 `bash -n` 语法检查；上机跑若报错（依赖回弹、API 签名变化）必须修掉再交付，不要带已知崩点交付。

## 6. 已知问题与排障（表）
| 现象 | 根因 | 处理 |
|---|---|---|
| cuda available False / DEVCOUNT=0 | 未挂 /opt/hyhal | 加 `-v /opt/hyhal:/opt/hyhal:ro` 重启 |
| illegal memory access | 未 source DTK | 每条命令前 `source /opt/dtk/env.sh` |
| ImportError libc10_hip.so | 装成上游原生 torch | 重装 `torch <ver>+das.dtk2604.*` |
| rdkit 编译失败 | cp310 应装 rdkit-pypi | `pip install rdkit-pypi` |
| 权重下载超时 | 内网不可达 | 可出网机下载后拷入 / 配镜像源 |
| 构象/大模型 OOM | batch 过大 | 减小 batch；多卡拆分；`hy-smi -a` 监控 |
| 多卡 NCCL 异常 | 镜像预置 NCCL_NCHANNELS_PER_PEER=2 | 进容器先 `unset NCCL_NCHANNELS_PER_PEER` |

## 7. 性能与精度建议
- 指定卡 `HIP_VISIBLE_DEVICES=0`；多卡高吞吐用 `gpurun` 绑核。
- 精度验证：取若干样本在 DCU 与 CPU 各跑一次，对比 max_diff（浮点容差 1e-5 级）。
- 显存监控：`hy-smi -a`。

## 8. 附录
- 8.1 待复核命令清单（撰写时若无法联网，列出核对仓库依赖/API、私有源可达性、rocminfo 的 curl 命令）。
- 8.2 参考链接（模型仓库、私有 PyPI、DTK 下载、相关 DCU skill）。

> 纪律：测试日志/性能基线不写入 skill；手册里如引用实测数据须与「观察」分离、原样保留异常。
