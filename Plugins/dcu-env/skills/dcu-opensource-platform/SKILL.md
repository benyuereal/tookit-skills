---
name: dcu-opensource-platform
description: "DCU/HCU 官方开源平台与运维组件：集群监控（hcu-dcgm + hcu-exporter/Prometheus）、Kubernetes 组件（device-plugin v3 / DRA driver / label-node / volcano / scheduler）、容器对接（hcu-container-toolkit）。补本地 env 域两大缺口：集群监控 + K8s 开源组件版本。"
version: 1.1.0
author: DCU Engineer
license: MIT
---

# DCU/HCU 官方开源平台与运维组件（dcu-opensource-platform）

> 定位：补本地 env 域两个缺口——**集群监控**（之前只有单机 `hy-smi` CLI，无 Prometheus 栈）与 **K8s 开源组件**（`dcu-kubernetes` 记的是内网 v2.4.0 部署手册，这里是官方 GitHub 开源版 v3.0.0 + DRA）。
> 单机巡检/硬件体检仍用 `dcu-hy-smi` / `hy-smi-tool`；本 skill 讲「集群级监控 + K8s 资源调度组件」。

> ⚠ 平台命名：官方把平台叫 **HCU**，仓库名/组件名带 `hcu-`；本地一律按 **DCU** 行文，仅保留仓库/组件原名（如 `hcu-dcgm`、`k8s-hcu-device-plugin`）。

## 更新机制（保证最新）

版本锚点来自 HYGON-AI GitHub 快照。刷新：

```bash
cd <agent-home>/skills/dcu
python3 scripts/hygon-upstream-sync.py --check   # 快检版本（device-plugin/DRA/dcgm 等 tag）
python3 scripts/hygon-upstream-sync.py --full    # 全量（含 README 缓存到 <agent-home>/cache/hygon-upstream/）
```

触发：部署/排查监控或 K8s 组件前先 `--check`；`--full` 建议每周跑。索引：`references/hygon-upstream/repos-index.md` §D。

## 一、集群监控（hcu-dcgm + hcu-exporter）

**架构**：`hcu-exporter`(Go, Prometheus) → 经 `hcu-dcgm`(Go, DCGM CGO 绑定) → `libhydmi.so`/`librocm_smi64.so`(HCU 驱动/DTK) → HCU/vHCU 硬件；K8s 下额外经 Kubelet Pod Resources Socket + Pod Informer 关联 Pod。

### hcu-dcgm（DCGM 管理/监控组件）
- Go 库 `github.com/HYGON-AI/hcu-dcgm/v3`；HTTP 服务默认 `:16081` + `dcgmi` CLI。
- 前置：Go 1.22+、CGO、已装 DCU 驱动；运行依赖 `/opt/hyhal/lib` 下的 `libhydmi.so` + `librocm_smi64.so`（MIG 还要 `libhydmi_mig.so`）。无硬件环境可复制 `.so` 到本地并设 `LD_LIBRARY_PATH` 编译，但无法真实监控。
- 构建/运行：
  ```bash
  go build -o hcu-dcgm ./pkg/service   # HTTP 服务，默认 0.0.0.0:16081
  go build -o dcgmi ./pkg/cmd          # 命令行工具
  ./hcu-dcgm -port 16081               # 或 -listen 127.0.0.1；HCU_DCGM_LISTEN env
  curl -G http://localhost:16081/NumMonitorDevices
  ./dcgmi discovery
  ```
- 部署：Docker 镜像 `hcu-dcgm:v2.0.0`（Dockerfile 不含驱动库，容器需挂 `/opt/hyhal` + 设备节点）；K8s 清单 `deployment/hcu-dcgm-k8s.yaml`。

### hcu-exporter（Prometheus 导出器）
- `--port 16080`，暴露 `/metrics`；Docker/K8s 部署见 README（特权 + `--device=/dev/kfd /dev/mkfd /dev/dri` + 挂 `/opt/hyhal`、`/etc/vdev`、`/etc/hostname`）。
- 配置：`--metrics-level=low|medium`（多卡/短 scrape 建议 low/medium）；`--enable-metrics` 精确指定；`--metrics-define`/`--label-define` 重命名；`--connect-k8s=false` 关 K8s 关联；`--hylink-detail`；IP/CIDR 白名单。
- 健康：>60s 未更新采集时间戳自动退出（便于 DaemonSet 重启）；HCU 驱动建议 ≥6.2.26，Go 1.25+。
- 指标：温度/功耗/利用率/显存/PCIe-DF-Hylink 带宽/ECC 错误；细粒度 CU 瞬时利用率/采样窗口/Wave/SE 利用率；vHCU + `hcunum` 动态映射（`/etc/vdev/dynamic`）。

## 二、Kubernetes 组件（官方开源版）

版本线：内网 `dcu-kubernetes` 手册为 v2.4.0；官方开源线已到 **device-plugin v3.0.0 / label-node v3.0.0 / DRA driver v1.0.0 / volcano v1.15.2**。

### k8s-hcu-device-plugin（v3.0.0，四模式）
| 模式 | 注册策略 | 典型资源 | 预切分 | 需 Scheduler |
|---|---|---|---|---|
| 物理整卡 | `hcu`/`mixed` | `hygon.com/hcu` | 否 | 否 |
| 预切分 vHCU | `vhcu`/`mixed` | `hygon.com/hcu-share-4c-16g` | 是（`hy-smi virtual`） | 否 |
| 动态切分 vHCU | `hami` | `hygon.com/hcunum` + `hcucores` + `hcumem` | 否 | **是**（k8s-hcu-scheduler） |
| MIG | `mig` | `hygon.com/hcu-mig-*` | 是（`hy-smi mig`） | 否 |
- 新增能力：NUMA 拓扑感知上报；HAMi 模式拓扑写入 `kube-system/hcu-topology-info` ConfigMap；健康检查走 DCGM。
- 部署：`deployment/static/k8s-hcu-plugin.yaml`(mixed) / `-mig.yaml` / `-hami.yaml`（含 RBAC）+ Helm。
- ⚠ HAMi 动态切分需 `k8s-hcu-scheduler`（device-plugin README 引用该仓库，org 列表未见，疑未公开）。

### k8s-hcu-dra-driver（v1.0.0，DRA 新模式）
- 机制：把物理 HCU 注册为 `ResourceSlice`（驱动名 `dra.hygon.com`），`Prepare` 返回 `CDIDeviceIDs`，容器运行时按 CDI spec 注入 `/dev/kfd`、`/dev/dri/*`、hyhal。
- 两种分配：整卡直通（Claim 不带 `exactly.capacity`）；按容量切 vHCU（Claim 含 `capacity.requests.cores/memory`，Prepare 动态建 vHCU，**单物理卡最多 4 份**）。
- 前置：**K8s 1.34+**；Feature Gate `DynamicResourceAllocation` + **`DRAConsumableCapacity`（apiserver/scheduler/kubelet 三门控）**；DRA CRD（DeviceClass/ResourceClaim/ResourceSlice）就绪；生产必须启用容器运行时 CDI；节点 `nodeSelector: hcu: "on"`；节点**不能已有预切分 vHCU**（`VDeviceCount != 0` 卡被跳过）。
- 默认镜像 `harbor.sourcefind.cn:5443/hcu/admin/base/hcu-dra-driver:v1.0.0`；与 device-plugin **可并存**（新应用用 DRA，存量 Device Plugin 逐步迁移）。
- 健壮性：驱动重启恢复已 Prepare Claim；定期清理无 Pod 的残留 vHCU。

### k8s-hcu-label-node（v3.0.0）
- DaemonSet 自动检测 HCU（`lspci` 识别 vendor Display/Co-processor + `/opt/hyhal/bin/hy-smi` 5s 超时验证驱动），给含 HCU 节点打标。
- 标签前缀 `hygon.com/hcu`：`hygon.com/hcu=true`（默认）＋可选 `-name`/`-cu-count`/`-vram`/`-driver-version`（经 `-name` 等 flag 开启）。
- 非 HCU 节点静默（不 crash-loop）；label 被删/改自动恢复。

### volcano（v1.15.2 下游适配）
- 批调度/队列/抢占/弹性，承接 HAMi 动态切分的多租户排队语义。

### z-file 官方下载位（Cat5 Kubernetes 插件，2026-10-14 实测）

- **v3.0.0 全家族**（download.sourcefind.cn:65024 Cat5「Kubernetes 插件」）：device-plugin v3.0.0、dra-driver v1.0.0、exporter、label-node v3.0.0、scheduler + admission-webhook、dcgm、**dcu-operator v1.0.0（console + helm）**、prometheus；另有 7 份部署指南 PDF 与 `previous_release/`（v1.5.1→v2.4.x 历史版本）。
- **仅 z-file 分发已坐实**（GitHub org/repo search 0 结果）：`k8s-hcu-scheduler`、admission-webhook、`dcu-operator` 三件不在 HYGON-AI org——§二 所述 scheduler「疑未公开」升级为确认，装 v3.0.0 全家族走 z-file。
- **dcu-operator 官方文档**：文档中心版本树 `7541efc4-54b1-11f1-8265-0242ac150003`（10 章 98 叶：产品简介/安装部署/Operator使用/DeviceConfig配置与变更/组件使用/控制台界面与使用/升级与日常运维/卸载与重装/故障排查/附录），`GET /api-doc/document/TreeDocument?VersionUUID=<树uuid>` 列目录、`DetailDocument?UUID=<叶uuid>` 取全文（匿名）。

## 三、容器对接（hcu-container-toolkit）
- 让容器运行时（docker/containerd）正确挂载 HCU 设备节点与 hyhal 的 Runtime 对接工具；简化容器内使用 HCU。配合 `dcu-docker` 使用（容器需 `--device=/dev/kfd --device=/dev/dri` + 挂 `/opt/hyhal:ro`）。

## 四、使用须知
- 先 `references/hygon-upstream/repos-index.md` §D 看版本/状态；README 全文在 `<agent-home>/cache/hygon-upstream/`。
- `hcu-dcgm`/`hcu-exporter` 与 `hy-smi` 互补：hy-smi 单机巡检，本套用于 Prometheus 集群监控 + Grafana。
- 组件版本对软件栈敏感（Go/CGO/驱动版本），升级前 `--check` 复核 tag 与 README 前置条件。

## 五、HYGON-AI 公开 org 资源地图与境内取包（2026-10-14）

> HYGON-AI GitHub org 共 45 个公开仓库（2026-10-14 复测；官网 codehub 只列 34 ⊂ 45，未列的含 pytorch-das / tensorflow-das / aiter-das / UltraEP-das / TransformerEngine-das / opencv 系 / faiss-das / spconv-das / FastPT / one-click-quant / cluster-manager-das / inference-cookbook-das / quality-gate 等——对照以 GitHub org 页为准，见 `references/hygon-upstream/repos-index.md`）。本 skill 主体讲其中"集群监控/K8s/容器对接"子集；下面是全 org 地图，取包或查配套组件时不再逐个搜。
> 刷新：org 仓库列表一条命令可现取（https://github.com/orgs/HYGON-AI/repositories）。下述清单仅作导航，确切以仓库页为准。

### 5.1 全 org 仓库一览（30 个，按用途分组）

| 仓库 | 定位 | 默认分支 |
|---|---|---|
| sglang-das | SGLang DCU/HCU 适配 fork（主仓库） | main(0.5.18 线)；release/20260825_v0.5.18；v0.5.12_dev / v0.5.15.post1_dev（旧代）；GLM-5-Next 支持已并入 |
| inference-cookbook-das | 官方模型部署文档库：docs/model-deployment 下 sglang 与 vllm 各模型文档，含 env+CLI+参考镜像+启动命令 | main |
| mooncake-das | Mooncake Transfer Engine（KV 跨节点 RDMA）DCU fork | main |
| rocSHMEM-das | DCU 版 SHMEM（low_latency 通信底层），另有 deepep-backend 分支 | develop |
| UltraEP-das | UCX 系 EP 通信库（与 DeepEP low_latency 并列的 EP 形态） | main |
| aiter-das | AITER 算子库（SGLANG_ROCM_USE_AITER_* 等） | main |
| triton-das / vllm-plugin-das / pytorch-das / jax-das / tensorflow-das / Megatron-LM-das / TransformerEngine-das / apex-das / flash-train / TileKernels-das / spconv-das / faiss-das / opencv-das / opencv_contrib-das / opencv-python-das / DiffSynth-Studio-das | 各上游库的 DCU/HCU 适配 | 各自默认分支 |
| hcu-dcgm / cluster-manager-das / volcano | 集群监控 / 运维调度（§一、§二 已覆盖） | main |
| one-click-quant / FastPT / TurboPhysAI | 量化 / 训练推理提速 / 物理 AI | main |
| train-cookbook-das | 训练配方（Megatron / mAP-cv / RL 矩阵） | main |
| LMCache-das | LMCache（KV 卸载）DCU 适配 | main |

### 5.2 境内取包技巧（GitHub API 常被限流时用）

- **API 限流**（403 / rate limit）时，GitHub 的 **codeload / raw / git-http（info/refs、smart HTTP）不走 API、基本不限流**——用它们绕过 API 限流。
- **下分支 tarball**：https://codeload.github.com/HYGON-AI/<repo>/tar.gz/refs/heads/<branch>（直接可下，无需 token；branch 名与默认分支无关）。
- **列分支 / refs（拿任意非 main 分支）**：GET https://github.com/HYGON-AI/<repo>/info/refs?service=git-upload-pack（Accept: application/x-git-upload-pack-advertisement）。返回是 **ASCII 十六进制 pkt-line 而非二进制长度**；解析：len=parseInt(hex4,16)（**长度含自身 4 字符前缀**），advance 按 len 前进（不是 4+len）；每行用 "^([0-9a-f]{40}) (.+)$" 取 ref 名。一次拿到全部分支/PR ref/tag。
- 若某环境 git over-schannel 被代理挡、而 node https 可直连：用 node:https 的 get 下载上述 URL（勿依赖全局 fetch）。

