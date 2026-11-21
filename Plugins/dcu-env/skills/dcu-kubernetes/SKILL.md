---
name: dcu-kubernetes
description: "DCU在K8s上的组件部署与Device-Plugin三模式资源调度"
version: 1.2.0
author: DCU Engineer
license: MIT
---

# 海光 DCU Kubernetes 部署与资源调度（dcu-kubernetes）

适用场景：在 **Kubernetes 集群上部署海光 DCU 组件**（让集群能分配、管理、监控 DCU / vDCU 资源），并**用 K8s 原生资源字段申请 DCU 算力**。本文是「通用 K8s 集群部署速查 + DCU 专属组件部署 + Device-Plugin 三模式资源调度」的完整手册。

> **来源**：内部文档《Kubernetes 在线及离线部署手册》（file_id `<internal-file-id>`，纯自包含已全文抓取；含 35 个未解析的 `[image]()` 截图占位——**已全部剔除，不搬运空占位**，截图内容用文字说明替代）。
> ⚠ **DCU 适配身份**：DCU K8s 组件（DCU-Label-Node / DCU-Exporter / DCU-Device-Plugin）版本 `v2.4.0`，镜像源 `<internal_registry>/dcu/admin/base/`；业务镜像（如 pytorch）源 `image.sourcefind.cn:5000/dcu/admin/base/`。DCU 资源通过 **扩展资源字段** `hygon.com/*` 申请。
> ⚠ **不存性能数字**：本文是部署方法，无基准；实测性能请用 dcu-inference-optimization 的方法学评估。
> ⚠ **离线包**：文末百度盘链接（pan.baidu.com/s/1pC1jRWrGtURS3MmgPGsa3g?pwd=rntt）保留为文字 URL，不内联二进制；提取码可能失效，以内部源为准。

## 0. 一句话机制

DCU 上 K8s = 通用 K8s 集群（kubeadm 1.23.0 + calico） + 三个 DCU 组件（打标/监控/设备插件）；业务 Pod 通过 `resources.limits` 里的 `hygon.com/*` 字段申请整卡 / vDCU / MIG 资源，由 DCU-Device-Plugin 注册、调度器分配。

## 1. 通用 K8s 集群部署（速查）

> 这部分是标准 K8s，非 DCU 专属，仅列关键命令速查；详细原理见官方文档。

**1.1 基础 OS 配置（所有节点）**：设 hostname、改 `/etc/hosts`、禁用 firewalld/SELinux/swap、`sysctl` 开 bridge-nf-call + ip_forward、可选 ntpdate 同步时间。

**1.2 yum 源**：在线加 kubernetes aliyun 源（`mirrors.aliyun.com/kubernetes`）；离线解压 `offline-packages.zip` 建 `local-repo`（baseurl `file:///opt/offline-packages`）。

**1.3 Docker**：`yum install -y docker-engine`；`daemon.json` 配 registry-mirrors（docker.1ms.run 等）+ `native.cgroupdriver=systemd`；`systemctl start/enable docker`。

**1.4 安装控制组件**：`yum install -y kubelet-1.23.0 kubeadm-1.23.0 kubectl-1.23.0`，`systemctl enable/restart kubelet`。

**1.5 初始化 Master**（在线/离线载入镜像后用）：
```bash
kubeadm init \
  --apiserver-advertise-address=<k8s_master_ip> \
  --image-repository registry.aliyuncs.com/google_containers \
  --kubernetes-version v1.23.0 \
  --service-cidr=10.96.0.0/12 \
  --pod-network-cidr=10.244.0.0/16 \
  --ignore-preflight-errors=all
# 配置 kubeconfig
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
export KUBECONFIG=/etc/kubernetes/admin.conf
```
> 注：init 输出的前两段（含 join 命令的 token / discovery-token-ca-cert-hash）中，前两段在 Master 配置、最后一段（kubeadm join ...）在 Node 执行。

**1.6 网络插件 calico**：下 `calico.yaml`，把 `CALICO_IPV4POOL_CIDR` 改为 `10.244.0.0/16`（与 pod-cidr 一致），`kubectl apply -f calico.yaml`。

**1.7 加入 Node**：`kubeadm join <k8s_master_ip>:6443 --token <token> --discovery-token-ca-cert-hash sha256:<hash>`。

**1.8 可选**：`kubeadm reset` 重置节点；`kubectl taint nodes --all node-role.kubernetes.io/master-` 让 Master 也可作工作节点。

## 2. DCU-Label-Node（节点自动打标）

以 DaemonSet 部署，运行时检查节点是否有 DCU，自动创建节点标签描述 DCU 信息。

```bash
# 在线/离线载入镜像 <internal_registry>/dcu/admin/base/dcu-label-node:v2.4.0
kubectl apply -f k8s-dcu-label-node.yaml
# 验证 pod Running
kubectl get pod -n kube-system -owide | grep dcu-label-node
# 验证标签：kubectl describe node <node> 应出现 DCU 相关标签
```

## 3. DCU-Exporter（监控）

实时监控 DCU 温度、显存占用、利用率等指标。

```bash
# 镜像 <internal_registry>/dcu/admin/base/dcu-exporter:v2.4.0
kubectl create -f dcu-exporter-v2.yaml
kubectl get pod -n kube-system -owide | grep dcu-exporter
# 验证指标采集
curl localhost:16080/metrics
```
> 可选：配 Prometheus（`prom/prometheus:v2.0.0`）+ Grafana（`grafana/grafana`，默认 admin/admin），导入 `dcu-cluster-monitor.json` / `dcu-exporter-dashboard.json` / `dcu-exporter-k8s-dashboard.json` 三个仪表盘看板。

## 4. DCU-Device-Plugin（核心：设备注册与资源调度）

device-plugin 负责 DCU 设备注册、健康检查、调度，含 **标准 / MIG / vDCU 动态切分** 三种模式。镜像 `<internal_registry>/dcu/admin/base/dcu-device-plugin:v2.4.0`。

### 4.1 标准模式（整卡 + 预切分 vDCU）

- 资源字段：`hygon.com/dcu`（物理整卡）、`hygon.com/dcu-share-{cu}c-{mem}g`（预切分 vDCU，如 `hygon.com/dcu-share-40c-30g` = 40 计算单元 30G 显存）
- 部署：`kubectl apply -f k8s-dcu-plugin.yaml`；验证 `kubectl describe node <node>` 的 Capacity 字段出现 `hygon.com/dcu: N` 或 `hygon.com/dcu-share-5c-4g: 1`

**整卡应用实例**（pytorch-dcu.yaml）：
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: dcu-pytorch-demo
spec:
  containers:
    - name: dcu-pytorch-demo
      image: image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk25.04.2-py3.10-minimal
      command: [ "/bin/bash", "-c", "--" ]
      args: [ "sleep infinity & wait" ]
      resources:
        limits:
          hygon.com/dcu: 1   # 申请 1 张 DCU 整卡
```

**预切分 vDCU 应用实例**：先在物理机切分，再申请对应规格
```bash
# 物理机预切分（每个 container 最多 1 张 vDCU）
hy-smi virtual -create-vdevices 1 -d 0 -vdevice-compute-units 20 -vdevice-memory-size 8000
hy-smi virtual -d 0 -show-vdevice-info   # 查看切分
```
```yaml
# pytorch-dcu-share.yaml：limits 用 hygon.com/dcu-share-20c-8g
      resources:
        limits:
          hygon.com/dcu-share-20c-8g: 1   # 名称按实际预切分规格
```

### 4.2 MIG 模式

- 资源字段：`hygon.com/dcu-mig-{mem}g-{gb}`（如 `hygon.com/dcu-mig-2g-15gb`）
- 节点打标签 `dcu-mode=mig`

```bash
# 物理机开 MIG + 建 GI/CI
hy-smi -mig 1
hy-smi mig -cgi 3 -C -i 0
hy-smi mig -L                  # 查看 MIG 实例
# 节点打标
kubectl label nodes <node> dcu-mode=mig
# 部署
kubectl apply -f k8s-dcu-plugin-mig.yaml
# 验证 kubectl describe node 的 Capacity 出现 hygon.com/dcu-mig-2g-15gb: 1
```
**MIG 应用实例**（dcu-mig-pytorch-demo.yaml）：
```yaml
      resources:
        limits:
          hygon.com/dcu-mig-2g-15gb: 1   # 申请 1 个 MIG 实例
```

### 4.3 vDCU 动态切分模式（HAMi / Volcano 调度器）

集群通过 device-plugin + vDCU-Scheduler（或 Volcano）实现 vDCU **动态切分 + 自动回收**。每个 container 最多 1 张 vDCU。资源字段：`hygon.com/dcunum`（vDCU 数量）/ `hygon.com/dcucores`（算力百分比）/ `hygon.com/dcumem`（显存 MB）。

**4.3.1 基于 HAMi**
```bash
kubectl label nodes <node> dcu=on
# 镜像：dcu-device-plugin:v2.4.0 + k8s-vdcu-admission-webhook:v2.4.0 + k8s-vdcu-scheduler:v2.4.0
kubectl apply -f vdcu-admission-webhook.yaml
kubectl apply -f vdcu-scheduler.yaml
kubectl apply -f k8s-dcu-plugin-hami.yaml
# 验证 kubectl describe node 的 Capacity 出现 hygon.com/dcunum: 32 等
```
**HAMi 应用实例**（pytorch-vdcu.yaml）：
```yaml
      resources:
        limits:
          hygon.com/dcunum: 1     # 1 个 vDCU
          hygon.com/dcucores: 60  # 算力 60%
          hygon.com/dcumem: 3000 # 显存 3000MB
```

**4.3.2 基于 Volcano**
```bash
kubectl label nodes <node> dcu=on --overwrite
# 镜像：vc-scheduler:1.14 + dcu-device-plugin:v2.4.0
kubectl apply -f volcano-development.yaml
kubectl apply -f k8s-dcu-plugin-hami.yaml
```
**Volcano 应用实例**（pytorch-vdcu-vocano.yaml，注意 `schedulerName: volcano`）：
```yaml
spec:
  schedulerName: volcano
  containers:
    - name: dcu-pytorch-demo
      image: image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk25.04.2-py3.10-minimal
      resources:
        limits:
          hygon.com/dcunum: 1
          hygon.com/dcucores: 60
          hygon.com/dcumem: 3   # 显存 3G
```

> 验证容器内资源：`kubectl exec <pod> -- hy-smi`（整卡/vDCU预切分）或 `kubectl exec <pod> -- hy-smi virtual -show-device-info`（动态切分）。

### 4.4 卸载清理（动态切分模式）

除删三个组件外，需手动清遗留资源：
```bash
kubectl delete -f vdcu-admission-webhook.yaml
kubectl delete -f vdcu-scheduler.yaml
kubectl delete -f k8s-dcu-plugin-hami.yaml
# 手动清除遗留
kubectl delete secrets -n kube-system admission-webhook-dcu-certs
kubectl delete mutatingwebhookconfigurations mutating-webhook-dcu-cfg
```

### 4.5 MIG vs vDCU：概念辨析与适用边界（选型先读）

**MIG 与 vDCU 是两种平行的卡切分手段，不是同一个东西**（常见误解：把两者当同义词，如"K100 支持 MIG 是不是指 vDCU？"）。两者都做"一张物理卡 → 多个小实例"，但实现原理与适用场景不同：

| 维度 | MIG（硬件多实例） | vDCU（软件虚拟化 vdevice） |
|---|---|---|
| 本质 | 硬件级切分，实例间硬件隔离 | 软件级调度切分，无硬件隔离 |
| 切分时机 | 物理机**预切分**（`hy-smi -mig 1` + `hy-smi mig -cgi` 建 GI/CI），Pod 再申请 | 两种形态：**预切分**（`hy-smi virtual -create-vdevices`，固定规格）/ **动态切分**（HAMi/Volcano 调度时切分、Pod 退出自动回收） |
| 规格灵活度 | 固定 GI/CI 规格组合 | 预切分固定规格；动态切分算力比例+显存任意组合（`dcucores`/`dcumem`） |
| 隔离强度 | 强（硬件隔离），适合多租户/安全隔离/计量计费场景 | 弱（软件调度），适合弹性混部、提利用率 |
| K8s 接入 | 节点标签 `dcu-mode=mig`，资源字段 `hygon.com/dcu-mig-{mem}g-{gb}` | 预切分：标准插件即可（`hygon.com/dcu-share-{cu}c-{mem}g`）；动态切分：节点标签 `dcu=on` + HAMi/Volcano 调度器（`hygon.com/dcunum/dcucores/dcumem`） |
| 状态检查 | `hy-smi mig -L` | `hy-smi virtual -d <dev> -show-vdevice-info`（预切分）/ 容器内 `hy-smi virtual -show-device-info`（动态） |

**选型边界**：
- 多租户安全隔离、强隔离、计量计费 → **MIG**
- 弹性混部、利用率优先、无需强隔离 → **vDCU 动态切分**
- 只需少量固定规格小实例、不想引入额外调度器 → **vDCU 预切分**
- 不切分、一实例一卡 → 标准整卡模式（`hygon.com/dcu`）

#### 卡型支持现状（2026-10-15 开发者社区文档中心查证）

> 来源：developer.sourcefind.cn 文档中心——《DCU 虚拟化用户手册》《DCU Multi-Instance 使用手册》（PDF）、K8s 组件文档 v2（DCU-Device-Plugin 2.2 MIG 模式 / 2.3 vDCU 动态切分模式）、云原生组件 v2.3.0/v2.4.0 发布说明。**官方未发布"卡型 × 模式"完整支持矩阵**，以下为文档明示/示例卡型 + 通用约束；未列出的卡型一律以实机验证为准。

| 维度 | 官方文档口径 |
|---|---|
| vDCU 文档示例卡型 | **K100_AI**（K8s 虚拟化手册应用示例 `hygon.com/use-dcutype: "DCU-K100_AI"`） |
| MIG 文档示例卡型 | **Z100 16GB**（《DCU Multi-Instance 使用手册》全流程示例）；K8s MIG 示例输出为 60CU/32GB 卡（型号未明示） |
| vDCU 能力来源 | `hy-smi virtual -h` 原文 "The virtual is **C-3000 HCU** software virtualization"——vDCU 原为 HCU 能力后扩展到 DCU，**支持与否强绑驱动版本** |
| 每卡 vDCU 上限 | **4 个**（虚拟化手册 + HAMi 适配文档） |
| MIG 与 vDCU | **不可同时开启**（K8s 文档 2.2 节与 Volcano 节均明示） |
| 驱动要求 | vDCU：驱动 ≥ 6.3.8（Volcano 虚拟化前提）；vDCU 仅指定显存切分：驱动 ≥ 6.3.16 + vDCU-Scheduler ≥ v2.4.0；MIG：驱动需以 `-s` 参数安装（启用虚拟化支持） |
| 组件要求 | MIG 模式 device-plugin：云原生组件 ≥ v2.3.0（`RESOURCE_REGISTER_STRATEGY` 环境变量区分模式）；MIG CLI（`hy-smi mig`）仅宿主机可用，容器内不支持（MIG 实例本身可在容器内使用） |

**未明示卡型（如 BW100/BW1100 等）的验证口径**：`hy-smi mig -L`（MIG 可用/已建实例）+ `hy-smi virtual`（vDCU 现状）实机查，或 `hy-smi -mig 1` 试开启；命令不存在/报错 = 该卡型+驱动版本不支持。勿默认"文档有该模式 = 本卡型可用"。

## 5. 资源字段速查表

| 模式 | 资源字段 | 示例 | 说明 |
|---|---|---|---|
| 标准整卡 | `hygon.com/dcu` | `hygon.com/dcu: 1` | 物理卡整卡 |
| 标准预切分 | `hygon.com/dcu-share-{cu}c-{mem}g` | `hygon.com/dcu-share-20c-8g: 1` | 物理机 hy-smi 预切分后申请 |
| MIG | `hygon.com/dcu-mig-{mem}g-{gb}` | `hygon.com/dcu-mig-2g-15gb: 1` | 节点需 `dcu-mode=mig` 标签 |
| vDCU 动态 | `hygon.com/dcunum` / `dcucores` / `dcumem` | `dcunum:1, dcucores:60, dcumem:3000` | HAMi/Volcano 调度，动态切分 |

## 6. 官方文档中心云原生索引（2026-10-14 实测，未真机复测）

文档中心 `/api-doc/*` 全匿名：`GetPageInfo`（ConfigTree 大类）→ `TreeDocument?VersionUUID=<树>`（目录树）→ `DetailDocument?UUID=<叶>`（全文 md）。云原生类 ConfigTree 口径 18 篇；本节列与 K8s 直接相关的两棵版本树：

### K8s 组件发布说明（树 `559f2ba3-eae0-11f0-a600-0242ac150003`，4 叶）
- v2.1.0 `74c4a6cb-eae0-11f0-a600-0242ac150003` / v2.2.0 `7c3072b9-…` / v2.3.0 `83b4a229-…` / v2.4.0 `8ef23460-…`（各叶 uuid 前缀，DetailDocument 取全文）。
- 本 skill 正文基于 v2.4.0 内部文档手册；新版发布说明先查这棵树。

### DCU-DCGM 文档（树 `768af98e-c10d-11f0-b077-0242ac150003`，5 章 188 叶）
- 1 简介 / 2 部署 / **3 二次开发** / 4 DCGMI 功能 / 5 DCU-DCGM 服务。
- §3.2 C API 接口说明 17 组：Device / Error / Event / Memory / PCIe / Performance / Physical State / Power / System / Topo / VDevice…（**全库唯一 DCGM C API 文档源**；`dcu-opensource-platform` 只覆盖 Go 包装 hcu-dcgm + dcgmi CLI，二次开发走这棵树）。
- §5.1 服务接口（含 Policy 策略组、vDevice 虚拟设备生命周期 12 接口）——做 vGPU/切分运维自动化直接对照。

> DRA 专题：见 `dcu-opensource-platform` §k8s-hcu-dra-driver（本树发布说明中 DRA 相关叶随 v2.x 说明更新）。
- DCU 虚拟化文档树 `9169ef18-c10d-11f0-b077-0242ac150003`（12 叶：DCU 虚拟化用户手册/Kubernetes 集群 DCU 虚拟化/Multi-Instance/生态支持）——§4.2 MIG/vDCU 的官方手册源（2026-10-14 实测）。

## 7. 与兄弟 skill 关系
- `dcu-pytorch-source`：PyTorch 后端（业务镜像 `pytorch:2.5.1-...-dtk25.04.2` 即来自此栈）
- `dcu-kernel-dev`：算子开发（Container 内跑的算子底层见那里）

## 使用须知
- 35 个文档截图占位（free -m 输出、kubeadm init 红框、Grafana 仪表盘等）已剔除，内容用文字说明替代；不搬运二进制
- 不存性能基线/逐行日志；离线包百度盘链接仅作文字溯源
- 高危操作（kubeadm init/reset、节点标签）谨慎，优先测试集群验证
