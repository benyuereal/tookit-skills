# dcu-dtk-layout

DCU DTK 目录结构与自带工具脚本速查 skill。

- SKILL.md：激活、顶层结构、工具脚本总览（含源码要点）、lib 库分组、常用命令、驱动侧（/opt/hyhal）PTX/fatbinary 工具链（hyptxas/hycc/hyfatbinary，<device-A> 实测）
- references/dtk-layout-and-tools.md：详细目录与工具文档（每个脚本源码要点、gpurun 详解、lib 全列表）
- references/dtk-scripts/：11 个 DTK 自带工具脚本的**原样源码存档**（clang-ocl / hipgdb /
  install_precompiled_kernels.sh / test_perf.py / extractkernel / hipconfig / hipvars.pm /
  hip_embed_pch.sh / hipcc_cmake_linker_helper / rocm_agent_enumerator / gpurun）

实测基线：
- 第一基线：设备 <device_name>，容器 <container_vllm015>，镜像 vllm015-ubuntu22.04-dtk26.04-0409-modelzoo，DTK-26.04-rc4。
- 第二基线（2026-08-28 全量扫描）：<device-A> 设备（BW150 8 卡），镜像 sglang:0.5.12-ubuntu22.04-dtk2604-py3.10-20260824-0006-bw150（当时最新 dtk2604），DTK-26.04-rc4（构建 2026-03-18），HIP 6.3.26113，MIOpen 2.18.0，DUSHMEM v3.2.5。
不同镜像版本号/库版本会变化，以目标容器内 /opt/dtk/.dtk_version 为准。

> 安装：本 skill 随 dcu-toolkit 包一起放在 <agent-home>/skills/dcu/ 即可，无需额外步骤。
> 密码/凭证不进 skill 包。
