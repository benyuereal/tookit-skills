# 20260408 版手册互补内容

> 《RCCL通信测试手册-20260408》增量：拓扑导出 / 四机 32 卡 / 容器免密完整 docker run / launch bounds 报错修法。
> 2026-10-16 自 `env/dcu-rccl-test/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），内容原文照录。

## 20260408 版手册互补内容（增量，来自《RCCL通信测试手册-20260408》）

> 本节内容来自 20260408 版官方测试手册（file_id `<internal-file-id>`），与本文主体（260610 内部手册）互补、不重复。该手册含多张截图（指标说明图、编译结果、ssh_config、报错截图），**已按纪律剔除 base64 内联图片，仅保留文字/命令/表格**。

### 补充 A：拓扑文件导出（NCCL_TOPO_DUMP_FILE）
单机调优时可用 `NCCL_TOPO_DUMP_FILE` 把 RCCL 实际识别的拓扑导出成 xml，需针对性调优时把该文件发给海光侧分析：
```bash
export NCCL_TOPO_DUMP_FILE=${PWD}/topo-generated.xml
./build/all_reduce_perf -b 2 -e 16G -f 2 -g 8
```
> 与 §多机优化里的 `NCCL_TOPO_FILE`（指定输入拓扑）不同：`_DUMP_FILE` 是**导出**当前识别结果，`_FILE` 是**载入**海光侧给的优化拓扑。

### 补充 B：四机 32 卡测试（完整 mpirun 示例）
四机流程与双机基本相同，仅 `-np`/`-H` 进程数不同（每台仍 8 卡）：
```bash
# 已按"测试前必做"清除 NCCL_NCHANNELS_PER_PEER（四台均需先检查 unset）
mpirun --allow-run-as-root \
  --prefix /opt/mpi \
  --mca btl_tcp_if_include p14p2 \
  --mca plm_rsh_args "-p 2222" \
  -x HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -x NCCL_SOCKET_IFNAME=p14p2 \
  -x NCCL_IB_HCA="=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10" \
  -x ROCM_PATH -x LD_LIBRARY_PATH \
  -x NCCL_NET_GDR_LEVEL=PHB -x NCCL_NET_GDR_READ=1 \
  -np 32 -H node01:8,node02:8,node03:8,node04:8 \
  ./build/all_reduce_perf -b 2 -e 16G -f 2 -g 1
```
> 双机/四机共用同一套 MCA/环境变量语义（见 §双节点 16 卡测试）；`NCCL_NET_GDR_LEVEL` 通常设 `PHB`（避免拓扑识别不准引发报错/宕机），与本文主体示例的 `SYS` 视实际拓扑识别结果选用。

### 补充 C：容器 SSH 免密完整 docker run（含 /opt/hyhal 必挂）
20260408 手册给出**推荐**启动命令（DTK 26.04 镜像、SSH 端口 2222），与本文 pitfall 9（漏挂 /opt/hyhal → DCU 不可见）一致：
```bash
docker run -itd \
  --name=rccl-tests \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /root/.ssh:/root/.ssh \
  --network=host --ipc=host --shm-size=16G \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  <internal_registry>/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10 \
  bash -c "mkdir -p /run/sshd && /usr/sbin/sshd -p 2222; sleep infinity"
docker exec -it rccl-tests bash
```
> 关键点：`-v /opt/hyhal:/opt/hyhal:ro`（DCU 可见必备）、`-v /root/.ssh:/root/.ssh`（复用物理机免密）、`--network=host`（端口互通）、sshd 监听 2222 须与 mpirun `--mca plm_rsh_args "-p 2222"` 一致。容器内从头配免密时改 `/etc/ssh/sshd_config`（设端口/允许 root 登录/允许授权免密）后 `pkill sshd; /usr/sbin/sshd -p 2222`。

### 补充 D：Launch params larger than launch bounds 报错修法
编译期/运行期报 `Launch params are larger than launch bounds for kernel` 时，按编译方式二选一加 `--gpu-max-threads-per-block=1024`：
- **install.sh / cmake 编译**（在 `src/CMakeFiles.txt` 底部 `target_link_libraries(rccl_common ...)` 行后插入）：
  ```cmake
  target_compile_options(rccl_common "--gpu-max-threads-per-block=1024")
  ```
- **make 编译**（在 `src/Makefile` 约第 23 行 `HIPCUFLAGS` 变量后追加）：
  ```make
  HIPCUFLAGS := -std=c++14 --gpu-max-threads-per-block=1024
  ```
> 这与本文 pitfall 10（该 flag patch 未生效）呼应：修复后务必 `grep` 确认 HIPCCFLAGS 真正含该参数，或改用 `GPU_TARGETS=... HIPCC=hipcc make ...` 显式传参。
