---
name: dcu-env-packages
description: DCU(海光)适配版软件包清单与安装排查指南
version: 1.1.0
author: DCU Engineer
license: MIT
---

# DCU 适配软件包清单与安装排查（dcu-env-packages）

> 用途：下次要看环境、搭环境、或排查"缺哪个 DCU 适配包 / 装哪个版本"时，按本 skill 走。
> 数据来源：① <device-B> 设备 `<container_vllm015>` 容器 `pip list` 实测；② sourcefind 私有 PyPI 索引 `https://pypi.sourcefind.cn/release/dtk/` 在线枚举（2026-08-03）。

## 1. 索引路径（最容易踩错的一步）

sourcefind 的私有 PyPI **不是** `https://pypi.sourcefind.cn/simple/<pkg>/`（那个路径返回 652B 的 DAS Hub 前端空壳页，curl 拿不到包）。

正确路径（pip 安装用）：
```
最新/汇总索引:  https://pypi.sourcefind.cn/release/dtk/
指定 DTK 版本:  https://pypi.sourcefind.cn/release/dtk25041/
                 https://pypi.sourcefind.cn/release/dtk2604/
```

⚠ **curl 机器化枚举用 `+simple/` 子路径（2026-08-28 <device-A> 实测）**：`/release/dtk/<包名>/` 直接 curl 返回的是 devpi 的 **web UI 页**（JS 渲染，grep 不到 wheel 链接）；真正的 PEP 503 纯链接页是 **`/release/dtk/+simple/<包名>/`**（devpi 的 simple 接口，直接列出 `torch-2.10.0+das.opt1.dtk2604...whl#sha256=...` 链接）。`pip install -i` 不受影响（pip 自动处理 simple 协议），但脚本里 curl 枚举版本/校验 sha256 时必须用 `+simple/` 路径：
```bash
# 机器化枚举某包全部 das 轮（+simple/ 纯链接页）
curl -sL 'https://pypi.sourcefind.cn/release/dtk/+simple/torch/' | grep -oE 'torch-2\.[0-9.]+\+das[^"<]*'
```

安装用法（来自官方示例）：
```bash
# 装最新（按当前 DTK 解析，注意：/release/dtk/ 是跨 DTK 汇总，pip 会按 wheel 文件名里的 DTK 标记筛）
pip install torch -i https://pypi.sourcefind.cn/release/dtk/

# 装指定 DTK 版本（推荐：显式锁 DTK，避免拉到不匹配的 wheel）
pip install torch -i https://pypi.sourcefind.cn/release/dtk25041/
```

⚠ 不要用 `/simple/` 路径，也不要用根域名裸页（那是 Hub 网页，非索引）。

## 2. 版本号约定（看标签就能判断适配关系）

海光发布轮的版本字符串格式固定：
```
<包基础版本>+das.<optN>.<dtk标记>.<torch绑定?>.<时间戳/commit>
```
- `das` = 海光 DAS 适配标识（区别于上游原生轮）
- `dtk<数字>` = 目标 DTK 版本，如 `dtk25041`、`dtk2604`、`dtk25042`
- `torch<版本>` = 绑定的 torch 基础版本（`torch251`=2.5.1、`torch271`=2.7.1、`torch280`=2.8.0、`torch290`=2.9.0、`torch2100`=2.10.0）；**部分 torch 自身轮不带此标记**（torch 是基座，不绑自己）
- `opt<N>` = 优化/构建轮次

例：`vllm-0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba` → vLLM 0.15.1，DTK 26.04，绑 torch 2.9.0。

## 3. 容器内实测适配包（<device-B> 设备 <container_vllm015>，DTK 26.04）

该容器当前装的全部 `+das` 适配轮（其余为上游原生依赖，不列）：

| 包 | 容器内版本 | 角色 |
|----|------------|------|
| torch | 2.9.0+das.opt1.dtk2604.20260331.g4e3c1e7 | PyTorch 基座（DCU HIP 后端） |
| torchvision | 0.24.0+das.opt1.dtk2604.torch290.2604091152.g66069c | 视觉库（绑 torch2.9） |
| triton | 3.3.0+das.opt2.dtk2604.torch290.20260331.g31542e | Triton 编译器（DCU 适配） |
| flash_attn | 2.8.3+das.opt1.dtk2604.torch290.202604031932.ga3ac7b | FlashAttention（DCU 适配） |
| flash_mla | 1.0.0+das.opt1.dtk2604.torch290.202604031618.g908bbb | FlashMLA（DCU 适配，MHA 解码加速） |
| lightop | 0.6.0+das.dtk2604.torch290.2604091144.ge8aebd | LightOp 自研融合算子库 |
| lmslim | 0.3.1+das.opt4.dtk2604.torch290.2604091256.gafc115 | 推理/压缩工具链 |
| vllm | 0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba | vLLM 推理框架（DCU 深度改造） |

> 该容器是 **torch 2.9.0 + DTK 26.04** 这一组合的可参考基线；换 torch 大版本时，其他包也要跟着换对应 `torch<版本>` 标记的轮（见 §4 矩阵）。

## 4. sourcefind 上各包在 dtk2604 的 torch 绑定矩阵（在线枚举）

> 查自 `https://pypi.sourcefind.cn/release/dtk/<包名>/`，取含 `dtk2604` 的构建，列出其基础版本与可绑定的 torch 版本。换 torch 大版本必须对齐这里的绑定，否则 import 报错。

| 包 | dtk2604 提供的基础版本 | 可绑定的 torch 标记 |
|----|------------------------|----------------------|
| torch | 2.5.1 / 2.7.1 / 2.8.0 / 2.9.0 / 2.10.0 | （自身，无 torch 绑定标记） |
| torchvision | 0.20.1 / 0.22.0 / 0.23.0 / 0.24.0 / 0.25.0 | torch251 / torch271 / torch280 / torch290 / torch2100 |
| triton | 3.1.0 / 3.3.0 / 3.4.0 / 3.5.1 | torch251 / torch271 / torch280 / torch290 |
| flash_attn | 2.6.1 / 2.8.3 | torch2100 / torch251 / torch271 / torch280 / torch290 |
| flash_mla | 1.0.0 / 1.2.0 | torch2100 / torch251 / torch271 / torch290 |
| lightop | 0.6.0 | torch2100 / torch251 / torch271 / torch290 |
| lmslim | 0.3.1 | torch2100 / torch251 / torch271 / torch290 |
| vllm | 0.9.2 / 0.11.0 / 0.15.1 / 0.18.1 | torch2100 / torch251 / torch271 / torch290 |
| aiter | 0.1.2 / 0.1.3 | torch2100 / torch290 |
| deepgemm | 2.1.0 | torch2100 / torch251 / torch271 / torch290 |
| sglang | 0.5.7 / 0.5.10rc0 | torch251 / torch290（py3-none-any 纯 Python 轮） |
| xformers | 0.0.33 | torch251 / torch271（py3-none-any） |

> 注：以上为 2026-08-03 在线枚举快照，**轮子会持续更新**，装之前用 `pip index versions <包> -i <索引>` 或浏览器/该 URL 复核最新。不要硬编码版本号进脚本。

> ⚠ **PaddlePaddle 不在 `/release/dtk/` 索引里（2026-09-07 实测）**：`/release/dtk2604/` 和 `/release/dtk25042/` 都没有 `paddlepaddle-dcu`。Paddle 的 DCU 适配轮散落在 **DAS Hub 的 nightly/customer 仓库**（`das_nightly/dtk2604-rc1`、`customer/dtk2604-aarch64` 等），要用 `dcu-das-hub-api` 的 JSON API 跨仓库查（`/api/pypi/package/<repo>/paddlepaddle-dcu?size=-1`），不能走 `/release/dtk/` pip 索引。版本选择坑（dtk25042 concat bug / dtk2604.260410 AVX-512 / dtk2604.260417 首个全通）见 `dcu-ontrack-experience` D2 节。

## 4.5 sourcefind 全量适配包清单（2026-09-01 DAS Hub API 枚举，53 包）

> 用 `dcu-das-hub-api` 的 `/api/pypi/packages/?repo=<repo>&size=-1` 枚举 `release/dtk2604`（39 包）+ `customer/dtk2604-rc4` 独有 14 包 = **53 包**。比 §4 的 12 包矩阵全（§4 是 2026-08-03 快照，只列了 torch 绑定矩阵关注的包）。**rc4 独有的 14 包 release 主索引没有**，装包要用 rc4 仓库。

**基座框架**：torch（2.5.1/2.7.1/2.8.0/2.9.0/2.10.0）+ torchvision + torchaudio、triton（3.1~3.6）、tensorflow、onnxruntime、jax 全家桶（jax/jaxlib/jax-rocm60-pjrt/jax-rocm60-plugin）、cupy（仅 rc4）

**推理框架**：vllm（0.9.2/0.11.0/0.15.1/0.18.1）+ **vllm-hcu**（0.21.0，插件 patch 层，见 `dcu-vllm-hcu-source`）、sglang 全家桶（sglang/sglang-kernel/sgl-kernel/**sglang-router**）、**mooncake-transfer-engine**（+ -rpc 仅 rc4，见 `dcu-mooncake-source`）

**算子/kernel**：flash-attn、flash-mla、deepgemm、aiter、lightop、lmslim、**deep-ep**（+ deep-ep-shca 仅 rc4，见 `dcu-deepep-source`）、causal-conv1d（仅 rc4）、fast-hadamard-transform-cuda（仅 rc4）、mhc-fused-kernel（仅 rc4）、hipdnn、hip-dsa/hip-dsa-kernels（仅 rc4，见 `dcu-hipdsa-source`）、tilelang（见 `dcu-tilelang-source`）、bolt-ops、fastsafetensors（见 `dcu-fastsafetensors-source`）

**训练框架**：apex、deepspeed、megatron-core、**dcu-megatron**、transformer-engine（+ transformer-engine-hygon 仅 rc4）、dgl、mmcv/mmcv-full（仅 rc4）、tvm、xformers、fastpt、byte-flux、spconv-rocm（仅 rc4）、torch-scatter（仅 rc4）、mori（仅 rc4）、qwen-asr（仅 rc4）——见 `dcu-training-frameworks-source`

**rc4 独有 14 包**（release/dtk2604 主索引没有，装包用 `customer/dtk2604-rc4` 仓库）：cupy、deep-ep-shca、fast-hadamard-transform-cuda、hip-dsa、hip-dsa-kernels、mhc-fused-kernel、mmcv、mmcv-full、mooncake-transfer-engine-rpc、mori、qwen-asr、spconv-rocm、torch-scatter、transformer-engine-hygon

> 源码结构索引（2026-09-01 新增 10 个 source skill）：`dcu-vllm-hcu-source` / `dcu-triton-source` / `dcu-ort-source` / `dcu-deepep-source` / `dcu-mooncake-source` / `dcu-fastsafetensors-source` / `dcu-tilelang-source` / `dcu-mhc-kernels-source` / `dcu-hipdsa-source` / `dcu-training-frameworks-source`。

### 4.6 z-file Cat4「AI生态包(DAS)」安装包目录（2026-10-14 实测，80 包）

> §4.5 是 pypi 索引口径（53 包，全 pip 可装）；Cat4 是 z-file 下载站安装包目录，**含非 pip 分发的组件与文档**，两者互补。

- **新组件**：`aitemplate`（注意≠aiter 算子库，见 `dcu-aiter-source`）、`XCompute`（DAS1.8.3/1.8.4 新增）、`XSystems`（同新线）；其余 fastmoe/fastpt/msda/sginfer/sglang-router/mooncake/boltops/flash-kda 等照旧。
- **DAS 技术文档 PDF 8 份**（Cat4「技术文档/」）：PyTorch2.7.0 兼容列表、LightOP、MIGraphX 5.2.0、LayerCheck、JAX/Paddle/TF 兼容列表等——版本配套疑问先查这里。
- DAS ReleaseNote 1.1→1.7 也在同目录；包路径结构 `<包>/<版本>/`。
## 5. 环境搭建 / 排障清单

**搭基础环境（DTK 26.04 + torch 2.9.0 组合，对齐 §3 容器基线）**
```bash
pip install torch torchvision triton flash_attn flash_mla lightop lmslim vllm \
  -i https://pypi.sourcefind.cn/release/dtk2604/
# 指定 DTK 子目录最稳；其他 torch 大版本把 2604 换成对应标记（如 dtk25041）
```

**排障按下列顺序排查：**

1. **索引路径错**（最常见）：用 `/release/dtk/` 而不是 `/simple/`。错用 `/simple/<pkg>/` 会得到 652B 的 Hub 网页，pip 报 "not a supported wheel" 或 "no matching distribution"。
2. **DTK / torch 版本错配**：装 `<包>+das...dtk2604.torch290` 的轮，但环境 torch 是 2.5.1 → `ImportError` / ABI 不匹配。必须先定 torch 版本，再选带对应 `torch<标记>` 的轮（见 §4 矩阵）。
3. **Python 版本错配**：轮子文件名带 `cp310/cp311/cp312`（如 `torch-2.9.0-...cp310-cp310`）。容器是 cp310（Python 3.10）；换 Python 版本要选对应 cp 标签的轮。
4. **装到了上游原生轮**：版本里**没有 `+das`** 的就是未适配 wheel（如纯 `torch==2.9.0`），在 DCU 上跑不起来。确认 `pip show <包>` 的 Version 字段含 `+das`。
5. **内网不可达**：sourcefind 是内网/私有源，出网受限的主机可能拉不动（见 `dcu-docker` 镜像逆向章节的"网络可达性预检"）。构建机/容器需能访问 `pypi.sourcefind.cn`。
6. **torch 自身升级破坏了配套**：triton/flash_attn/flash_mla/lightop/lmslim/vllm 都绑 torch 标记，torch 升大版本要整体重装一整套，不能只升 torch。

## 6. 与其他 skill 的关系
- `dcu-dtk` / `dcu-dtk-layout`：DTK 版本与 `/opt/dtk` 布局基础（版本号里的 `dtkXXXX` 对应什么）。
- `dcu-pytorch-source`：低版本 torch 源码编译适配 + LibTorch 提取；高版本走本 skill 的预编译 wheel。
- `dcu-vllm-source/0.18/0.26`：按 vLLM 大版本定位源码适配点；装包时版本须与本表 dtk2604 矩阵对齐。
- `dcu-docker`：镜像逆向重建章节——其中"网络可达性预检"与本 skill 的私有源可达性直接相关。

## 验证步骤（本 skill 自检）
- [x] 容器 `pip list` 实测 8 个 +das 适配轮（<device-B> 设备 <container_vllm015>）
- [x] sourcefind 索引正确路径 `/release/dtk/` 已验证可枚举（torch 23 个 das 轮）
- [x] dtk2604 torch 绑定矩阵已在线枚举（12 个包）
- [x] 标注"索引快照时间 2026-08-03，勿硬编码版本号"
- [x] 未写入性能基线/日志（遵守 skill 内容纪律）
