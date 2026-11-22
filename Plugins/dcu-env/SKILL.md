---
name: dcu-env
description: "调优前的 DCU 服务器环境探测（DCU env detect）：CPU、NUMA、DTK、驱动。何时用：接手一台机器、准备做性能优化前先摸清环境底数。"
version: 1.6.0
author: DCU Toolkit Contributors
license: MIT
---

# DCU 基础环境 — 域级技能包

## 工具介绍

海光 DCU 基础环境技能集：SSH、环境检测、DTK/驱动、容器、hy-smi、NUMA、RCCL、RoCE 与 K8s 部署。本目录为 Kits Platform Skill Tool `dcu-env` 的发布单元，包含 24 个子 skill：

| skill | 说明 |
|---|---|
| dcu-adaptation-overview | DCU 适配总览——版本兼容矩阵（DTK×torch×框架×算子包）+ 软件能力矩阵（哪些能跑/验证状态/限制）+ 真机验证结果 + 升级 … |
| dcu-cpu-affinity | 海光DCU CPU绑核优化（NUMA亲和性调度） |
| dcu-csv-passthrough | 海光DCU机密计算CSV+PCIE设备直通虚拟机部署与踩坑 |
| dcu-das-hub-api | 查 DCU 源(pypi.sourcefind.cn)包版本/下 wheel 的 JSON API 用法与踩坑 |
| dcu-docker | Operate DCU environments inside Docker. |
| dcu-driver | 海光DCU驱动安装、版本兼容、加载验证与排障 |
| dcu-dtk | DCU DTK开发工具包下载、安装与激活；仅物理机需装，容器自带 |
| dcu-dtk-layout | DCU DTK目录结构与自带工具脚本速查 |
| dcu-env | Detect DCU server env (CPU/NUMA/DTK/driver) pre-tuning. |
| dcu-env-packages | DCU(海光)适配版软件包清单与安装排查指南 |
| dcu-hy-smi | hy-smi查询监控：温度/功耗/显存/拓扑/进程，DCU巡检必备 |
| dcu-hyqual-acceptance | 海光DCU HyQual 官方压力验收工具：5 卡型镜像版本矩阵 + du-bootstrap/DuDiag 环境部署自检 + OOB 带外… |
| dcu-image-hub | 查光合开发者社区(sourcefind.cn) DCU 镜像仓库：Web API 列镜像/列 tag、Harbor 匿名 token 直查、… |
| dcu-image-reverse-rebuild | 反推 DCU 镜像为可重建的 Dockerfile（内网源换公网源，另一台设备只凭 Dockerfile 即可重建）。 |
| dcu-image-to-native-env | 把 DCU 镜像复刻为物理机原生 conda 环境（非 Dockerfile 重建）。 |
| dcu-kubernetes | DCU在K8s上的组件部署与Device-Plugin三模式资源调度 |
| dcu-modelzoo | 查光合开发者社区 ModelZoo（GitLab group modelzoo，626 个 DCU 模型适配仓库）：GitLab API v… |
| dcu-opensource-platform | DCU/HCU 官方开源平台与运维组件：集群监控（hcu-dcgm + hcu-exporter/Prometheus）、Kubernete… |
| dcu-rccl-source | DTK RCCL源码级分析：PCIe混合链路优化 |
| dcu-rccl-test | RCCL 多卡互联带宽测试：单/多节点 AllReduce 性能测试与排障 |
| dcu-roce-network | 海光 DCU 集群 RoCEv2 网络环境搭建：IP/MTU 规划、静态 ARP 全量注入（关键）、QoS/PFC/ECN 配置、RDMA … |
| dcu-software-adaptation-guide | DCU 软件适配通用参考——拿到任意新软件包，按 5 大适配形态决策 + 7 步 SOP + DTK 核心差异 + 适配点定位技巧 + 验证… |
| dcu-ssh | SSH to DCU servers for remote command execution. |
| dcu-topology | 海光DCU多卡多节点拓扑优化 |

## 安装与使用

- **单 skill 使用**：Agent 按上表选定 skill 后，读取 `skills/<skill>/SKILL.md` 并按其流程执行（子 skill 自包含，含 scripts/ 与 references/）。
- **统一路由**：跨域任务优先读仓库根 `SKILL.md`（包级路由表）或 `dcu-core/skills/dcu-skill-router`（全量索引）。
- **安装**：见仓库根 `README.md` / `install.sh`（支持 Claude Code / OpenCode / Codex 等多框架，项目级或全局）。
- **版本**：与包根 SKILL.md 一致（1.6.0）；升级以包内版本为准。
