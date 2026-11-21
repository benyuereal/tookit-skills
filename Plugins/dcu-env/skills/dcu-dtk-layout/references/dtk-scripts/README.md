# DTK 自带工具脚本原始源码（实测采集）

采集自：设备 <device_name>，容器 <container_vllm015>
镜像：vllm015-ubuntu22.04-dtk26.04-0409-modelzoo，DTK-26.04-rc4
采集日期：2026-08-01
来源路径：
  - /opt/dtk/bin/{clang-ocl,hipgdb,install_precompiled_kernels.sh,test_perf.py}
  - /opt/dtk/hip/bin/{extractkernel,hipcc_cmake_linker_helper,hip_embed_pch.sh,hipconfig,hipvars.pm,rocm_agent_enumerator}
  - /opt/dtk/dcc/bin/gpurun

说明：这些是 DTK 自带的工具脚本（bash/perl/python），内容未改动，仅原样保存便于离线查阅。
逐个用途与源码要点见同级文档 `../dtk-layout-and-tools.md`。
