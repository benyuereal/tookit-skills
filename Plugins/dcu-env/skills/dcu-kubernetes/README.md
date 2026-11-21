# dcu-kubernetes — 海光 DCU Kubernetes 部署与资源调度

海光 DCU 在 K8s 上的组件部署与资源调度：通用 K8s 集群速查（基础OS/离线yum源/Docker+daemon.json/kubeadm init 1.23.0+calico/join）、DCU 三大组件（DCU-Label-Node DaemonSet 自动打标 / DCU-Exporter 监控温度·显存·利用率+curl metrics+Prometheus+Grafana / DCU-Device-Plugin 设备注册调度）、Device-Plugin 三模式（标准 hygon.com/dcu 整卡与 dcu-share 预切分 vDCU / MIG 模式 hy-smi -mig 1 + dcu-mig-* / vDCU 动态切分 HAMi 或 Volcano 调度器 用 dcunum·dcucores·dcumem）、4 个应用实例 Pod yaml、卸载清理、资源字段速查表。35 个文档空截图占位已剔除。

详见 `SKILL.md`。
