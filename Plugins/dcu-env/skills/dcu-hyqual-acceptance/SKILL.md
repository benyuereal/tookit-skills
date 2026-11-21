---
name: dcu-hyqual-acceptance
description: 海光DCU HyQual 官方压力验收工具：5 卡型镜像版本矩阵 + du-bootstrap/DuDiag 环境部署自检 + OOB 带外管理下载位（标前/验收/装机场景）
version: 1.0.0
author: DCU Engineer
license: MIT
metadata:
  toolkit:
    tags: [dcu, hygon, hyqual, acceptance, stress-test, du-bootstrap, dudiag, oob, 验收]
    related_skills: [dcu-env, dcu-rccl-test, dcu-ontrack-experience, dcu-driver, dcu-hy-smi]
  source:
    provenance: "sourcefind 下载站 Cat5/13 + 文档中心 hyqual 树 2026-10-14 匿名实测（未真机复测）"
    reference: "z-file CategoryID=5 基础压力工具/环境部署检测工具, CategoryID=13 带外管理, 文档中心树 5ce53abd"
---

# DCU HyQual 官方压力验收与环境自检

## Overview

HyQual 是海光官方压力验收/稳定性测试工具包（z-file Cat5「基础压力工具」）。适用场景：标前验收、装机后基线压测、长时间稳定性拷机。

**分工边界**（避免第 4 个 hyqual 落点）：通信专项带宽测试（RCCL allreduce）→ `dcu-rccl-test`；环境信息检测（CPU/NUMA/DTK/driver 采集）→ `dcu-env`；按报错关键词反查踩坑 → `dcu-ontrack-experience`；监控巡检 → `dcu-hy-smi`。本 skill 只管官方验收工具链的下载位、版本配套与入口。

## HyQual 镜像版本矩阵（2026-10-14 z-file 实测）

路径：`download.sourcefind.cn:65024` → CategoryID=5 → `基础压力工具/<机型>/`

| 机型目录 | 测试包 | 配套用户手册（同目录 PDF） |
|---|---|---|
| BW1000系列 | docker_hyqual_v3.0.10.tar.gz | HyQual-用户手册-BW1000-v3.0.10.pdf |
| BW1100 | docker_hyqual_v3.2.3.tar.gz | HyQual-用户手册_v3.2.3.pdf |
| K100 | hyqual-v2.2.6.tar.bz2 | HyQual-用户手册-v2.2.6.pdf |
| K100-AI | hyqual-v2.2.7.tar.gz | HyQual-用户手册-v2.2.7.pdf |
| Z100L | hyqual-v2.2.6.tar.bz2 | HyQual-用户手册-v2.2.6.pdf |

- **格式差异**：BW 系用 docker 镜像包（docker_hyqual_*.tar.gz），K/Z 系用裸 tar 包（hyqual-*.tar.bz2/.tar.gz）。
- **配套版本**（报告实测口径）：BW1000 v3.0.10 配 dtk2604、驱动 ≥rock-6.3.16；BW1100 v3.2.3 配驱动 ≥rock-6.3.30；K100/Z100L 共用 v2.2.6、K100-AI v2.2.7（K100/Z100L 的 DTK/驱动配套以各包内用户手册 PDF 为准）。
- `previous_release/`：dtk23.04/、dtk23.10.1/ 旧环境子目录 + 历史版本 hyqual-1.3.8 / 2.0.2_dtk22.04 / 2.0.3 / hyqual-K100_AI-v2.2.2 / docker_hyqual_v3.0.10_base + 历史手册 v1.6.0/v2.0.0/v2.2.6。

## 环境部署检测工具（同 Cat5「环境部署检测工具」）

- `du-bootstrap.tgz` + 《DTK 26.04 du-bootstrap使用手册.pdf》——装机自检入口。
- 《DTK 26.04 DuDiag使用手册.pdf》——诊断工具。
- 《DTK 26.04 DTK Container Toolkit 使用手册.pdf》 + `DTK Container Toolkit/` 目录：**24 个 OS** 的容器工具包（Alinux3.0/3.2304、Anolis8.4/8.6、BCLinux8.2、CentOS7.6/8.5、Kylinv10SP3/Kylinv11、NFS3.2/4.0/NFSCNS4、Opencloud8.8、RedHat7.9/8.7、Rocky8.6/9.2、SugonOS8.9、TencentOS3.3/4.4、UOS1021e、Ubuntu20.04.1/22.04、openEuler22.03）；`previous_releases/DTK-25.04.3/`。

## 带外管理 OOB（Cat13）

`V1.0.0-DCU-OOB-Manager.tar.gz`（hydcu-oob-manager，OpenBMC 带外管理，单文件下载位）。带外查询/重启/固件场景用；文档中心暂无独立手册树。

## 官方文档中心 hyqual 手册树

版本树 `5ce53abd-1b65-11f1-a4a0-0242ac150003`（12 叶）：hyqual-BW1000系列 / hyqual-BW1100系列 / hyqual-k100-ai 三系列 × 修改记录/概述/HyQual安装与配置/使用指南。取全文 `GET https://developer.sourcefind.cn/api-doc/document/DetailDocument?UUID=<叶uuid>`（匿名）。**K100/Z100L 机型文档只在 z-file 包内 PDF**，文档中心无对应叶。23 树总索引见 dcu-toolkit-skills 包 `references/sourcefind/doc-center-index.md`。

## 使用须知

- 全部下载位/版本号为 2026-10-14 官方下载站实测；**未真机复测**——验收跑分结论只认端到端指标，镜像配套疑义先读同目录用户手册 PDF。
- 列目录实时校验：`curl -s "https://download.sourcefind.cn:65024/api-static/file/ListFile?CategoryID=5&Path=基础压力工具"`（Path 相对分类根、URL 编码、不带前导斜杠）。