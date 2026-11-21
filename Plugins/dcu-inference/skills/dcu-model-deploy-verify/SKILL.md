---
name: dcu-model-deploy-verify
description: "DCU模型部署：先真机实测跑通再写文档，含PyTorch范式与Uni-Mol实测"
version: 1.0.0
author: DCU Engineer
license: MIT
---
# DCU 模型部署验证纪律与实战范式（dcu-model-deploy-verify）

> 本 skill 沉淀两类知识：① 用户强约束的「部署文档必须先实测再写」纪律；
> ② 把纯标准 PyTorch 算子模型（无 .cu 扩展）部署到 DCU 的通用范式，
> 并以 Uni-Mol on DTK 26.04 为实测范例（详见 `references/unimol-dcu-deployment.md`）。

## 0. 铁律（用户明确，2026-08-06）
**「没有跑通算什么部署文档」** —— 禁止把方案/计划草稿当部署手册交付。
标准流程：
1. 上设备（<device-B> 等，按 `dcu-ssh` 规范 SSH 进 DCU 服务器，在 DTK 容器里）实测：装包 → 跑最小推理 → 跑训练/评估。
2. **抓取真实 stdout** 作为证据（不裁剪、不自行下结论，异常原样保留）。
3. 把真实输出回填手册，定稿标注「实测通过」；未实测的文档一律标「待复核」，不得写「已验证通过」。
> 违反案例：本 skill 首版曾先产出一份「方案草稿」（含错误的包名 `unimol`），被用户驳回。
> 正确做法见 `references/unimol-dcu-deployment.md`——先上 <device-B> 真机跑通，再定稿。

## 1. 适配可行性快速判断
- 先看目标模型**是否含自定义 CUDA(.cu) 扩展**：
  - **无 .cu**（如 Uni-Mol，纯标准 PyTorch 算子）→ DCU 上无需 kernel 移植，HIP 把 `torch.cuda.*` 映射到 DCU，直接 `device='cuda'` 跑。
  - **有 .cu**（如 FlashAttention 类）→ 需 HIP 化/换 DCU 适配 wheel（见 `dcu-vllm-op-replace` / `dcu-aiter-source`）。
- 非深度学习依赖（rdkit/ase/sklearn/scipy）多为纯 Python/CPU 库，与硬件后端无关，正常 pip 装。

## 2. 通用部署流程（纯 PyTorch 模型上 DCU）
1. **进 DTK 容器**：`docker run ... --device=/dev/kfd --device=/dev/dri -v /opt/hyhal:/opt/hyhal:ro`（hyhal 不挂则 DEVCOUNT=0）；运行前 `source /opt/dtk/env.sh`。
2. **装 DCU torch**：必须海光私有源带 `+das.dtkXXXX` 的 wheel（`https://pypi.sourcefind.cn/release/dtk<版本>/`），**不能**用上游原生 torch（无 DCU 后端）。
3. **查 CUDA 专用残留包**：容器若带 `cupy-cuda12x` 等 CUDA 版包，会与 numpy 2.x 冲突并破坏 torch numpy 桥 → 卸掉 + 锁 `numpy<2.0`（见 `dcu-env-packages` 排障第 7 条）。
4. **权重/数据下载**：HF/ModelScope 类一律准备镜像源 + **关 Xet**（`HF_HUB_DISABLE_XET=1`，否则镜像站 401）。
5. **三步验证**：设备可见性（`torch.cuda.is_available()` + 张量落 `cuda:0`）→ 最小推理 → 训练/评估；把真实 stdout 回填。
6. **装声明 torch 依赖的第三方包一律 `--no-deps`**（见 §2b 坑）。

## 2b. 坑：装依赖 torch 的包必须 `--no-deps` 保护 DCU 定制 torch（2026-10-10 FunASR v1.4.16 / K100_AI 实测）

**现象**：`pip install <pkg>`（pkg 的 install_requires 里声明了 torch/torchaudio/torchvision）时，pip 依赖解析会**把容器里的 DCU 定制 torch（如 `2.5.1+das.opt1.dtk2604`）卸掉、换装上游 CUDA 版**——之后 `torch.cuda.is_available()` 变 False，整个环境报废且不易察觉（包都能 import，跑起来才错）。

**规则**：
1. DCU 容器里装任何依赖树可能拉 torch/torchaudio/torchvision 的包，一律 `pip install --no-deps <pkg>`。
2. 装完必查三件套：`pip list | grep -iE '^(torch|torchaudio|torchvision) '`，版本号**必须仍带 `+das...dtkXXXX` 后缀**；任何一个变成纯数字（如 `2.5.1`）就是被换成了 CUDA 版 → 立即恢复 DCU wheel 或回滚容器，禁止继续往下测。
3. `--no-deps` 缺的纯 Python/CPU 依赖单独 `pip install <dep>` 补（如 rapidfuzz），注意这些补装包自身也不要带 torch 解析链。
4. **判读口诀**：DCU 定制 torch 版本号恒带 `+das`；纯上游版本号 = CUDA 版。

**真实案例**：FunASR 1.4.16 on K100_AI（底座 `pytorch:2.5.1-dtk26.04-py3.10`）：`pip install --no-deps funasr==1.4.16` + 单独补 `rapidfuzz` → torch 三件套不变，SenseVoiceSmall 在 cuda:0 推理（4.0s/条、RTF 0.72、峰值显存 ~0.9GB、识别文本正确），离线镜像 load 回来复测 PASSED。

## 3. 设备连接与实测（dcu-ssh 规范要点）
- 凭证在 `~/.config/dcu-toolkit/credentials.json`（chmod 600），设备清单在 `<agent-home>/skills/dcu/data/devices.json`（模板）。
- <device-B> 设备 = <<internal-ip>>，8×BW100(gfx936)，现有可用容器常带 torch 2.9.0+das.dtk2604（如 `<container_vllm015_rebuild>`）。
- 实测命令走 SSH（`sshpass -e`，禁 `-p`）；长任务用宿主侧 `nohup ... > log 2>&1 &` + 轮询，别让 SSH 直连阻塞。
- 传脚本进容器优先 `tar -C /tmp -cf - f | docker exec -i <c> tar -xf - -C /tmp/`（docker cp 在本机 docker 偶有 `/proc/self/fd` 解析 bug，tar 管道更稳）。

## 4. 与其他 DCU skill 的关系
- 路由入口：`dcu-inference-infra`（本 skill 是其「部署纪律 + 纯 PyTorch 模型范式」的补充）。
- 包矩阵/冲突：`dcu-env-packages`；torch 适配/精度交叉验证：`dcu-pytorch-source`；连接规范：`dcu-ssh`；DTK 激活：`dcu-dtk`。

## 5. 知识库
- `references/unimol-dcu-deployment.md`：Uni-Mol on DTK 26.04 实测（包名 `unimol_tools`、cupy 冲突、HF_XET、CSV 输入、mae/mse 指标、真实 stdout、排障表）。
- **PaddlePaddle 框架**（非 PyTorch）的部署案例 PP-OCRv5 on K100-AI（DTK 26.04）：踩的是「框架 wheel 的 DTK 版本 + CPU 指令集匹配」这个非 PyTorch 特有坑（`dtk2604.260410` SIGILL / `dtk25042` concat VMFault / `dtk2604.260417` 全通）。完整案例见 `dcu-ontrack-experience` D2 节（故障排查）+ 内部文档《PP-OCRv5_海光DCU离线镜像部署手册》；镜像打包/离线部署见 `dcu-docker`「DCU 镜像打包/导出与离线部署」节。
