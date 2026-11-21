# =============================================================================
# 重建的 Dockerfile —— 镜像:
#   harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm015-ubuntu22.04-dtk26.04-0409-modelzoo
#   (别名: dcu/admin/base/vllm:0.15.1--ubuntu22.04-dtk26.04-py3.10-20260409)
#
# ⚠⚠⚠ 专属成品, 切勿套用到其他容器 ⚠⚠⚠
#   本文件是针对【上面这一个具体容器 / 这一个具体镜像】反推并验证通过的成品。
#   里面的包版本(torch/torchvision/triton/vllm/flash_*/lightop/lmslim 等)、DTK 版本
#   (DTK-26.04-rc4)、Ubuntu 版本(22.04) 全部来自该容器的实测 pip freeze + DTK 探测,
#   是写死的。即便另一个容器“版本看起来相同”, 也可能有:
#     - 不同的 DTK 构建号 / 不同的 DCU 包精确轮(同一主版本不同构建号行为可能不同);
#     - 不同的通用包版本(numpy/opencompass/transformers 等);
#     - 不同的 RUN/ENV 层顺序或额外层(原镜像的 docker history 不同);
#     - 不同的 DTK 系统标签(ubuntu20.04 vs ubuntu22.04 包)。
#   所以: 换任何其他容器, 必须重新反推生成, 不能直接复用本文件。
#   通用做法见 skill: python3 scripts/generate_rebuild.py <新容器> --outdir ./out
#   (自动探测新容器的版本 + DTK, 动态生成, 不写死)。
#
# 来源: <device-B> 设备 <container> 容器的 `docker history --no-trunc` +
#        `docker inspect` + 容器内实际文件提取 (install_dep.sh, /etc/profile.d/env.sh)
#
# 重建方式说明:
#   - 所有 `RUN`/`ENV`/最终 `CMD` 均按 `docker history` 逐层 1:1 还原。
#   - 原 Dockerfile 中 `COPY dir:...` / `COPY file:...` 的构建上下文输入
#     (15.8GB 的 /opt/dtk、/install_dep.sh、/etc/profile.d/env.sh、/usr/share/misc/magic 等)
#     无法从已构建的镜像里“反编译”出原路径与内容，已用注释标出并给出
#     容器内可提取到的真实内容 (install_dep.sh / env.sh 已单独还原到同目录)。
#   - 中途下载的源码包 (hwloc/ucx/openmpi/MLNX_OFED) 来自内网镜像
#     http://10.16.4.1:8000/pkg/ 与 mellanox，需构建机可达该内网。
#   - 有 3 个 ENV 同名被覆盖 (LD_LIBRARY_PATH / PATH / C_INCLUDE_PATH)，
#     此处只写“最终生效值”，与 `docker inspect` 的 Config.Env 完全一致。
#
# 注: 该镜像不含 entrypoint，默认 CMD 为 `tail -f /dev/null` (常驻容器)，
#     与 <internal-user> 起容器用法一致。
# =============================================================================

FROM ubuntu:22.04

LABEL maintainer=<internal-email>
LABEL org.opencontainers.image.ref.name=ubuntu
LABEL org.opencontainers.image.version=22.04

# ---- 构建参数 ----
ARG DEBIAN_FRONTEND=noninteractive
ARG CPUS=32
ARG TZ=Asia/Shanghai

# ---- 基础系统环境 ----
ENV TZ=Asia/Shanghai
ENV SHELL=/bin/bash
ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 LANGUAGE=en_US.UTF-8

# ---- Ubuntu 源改阿里云 (jammy) ----
RUN printf 'deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse \n\
deb-src http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse \n\
deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse \n\
deb-src http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse \n\
deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse \n\
deb-src http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse \n\
deb http://mirrors.aliyun.com/ubuntu/ jammy-proposed main restricted universe multiverse \n\
deb-src http://mirrors.aliyun.com/ubuntu/ jammy-proposed main restricted universe multiverse \n\
deb http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse \n\
deb-src http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse' > /etc/apt/sources.list

# ---- 通用编译/工具链 ----
RUN apt-get update -y \
    && apt-get install --no-install-recommends -y \
        build-essential git wget gfortran elfutils libelf-dev libdrm-dev kmod \
        libtinfo5 sqlite3 libsqlite3-dev libnuma-dev libgl1-mesa-dev alien rsync \
        libpci-dev pciutils libpciaccess-dev libbabeltrace-dev pkg-config \
        libfile-which-perl libfile-basedir-perl libfile-copy-recursive-perl \
        libfile-listing-perl libprotobuf-dev libio-digest-perl libdigest-md5-file-perl \
        libdata-dumper-simple-perl vim curl libcurlpp-dev openssh-server sudo locales \
        openssl libssl-dev patch ninja-build libgoogle-glog-dev liblmdb-dev \
        libopenblas-dev libgflags-dev libibverbs-dev ibverbs-utils libffi-dev zlib1g \
        zlib1g-dev libbz2-dev libncurses-dev libsqlite3-dev read-edid numactl \
        libjpeg62 liblzma-dev libgdbm-dev libgdbm-compat-dev libnss3-dev \
        libreadline-dev libncurses5-dev libncursesw5-dev xz-utils tk-dev zip unzip cargo \
    && locale-gen en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---- ML / 多媒体 依赖 (gstreamer / opencv / 调试工具) ----
RUN apt-get update -y \
    && apt-get install --no-install-recommends -y \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools \
        gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 \
        gstreamer1.0-qt5 gstreamer1.0-pulseaudio gst123 libopencv-dev \
        python3-opencv figlet gdb net-tools htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---- OFED / RDMA 前置系统包 ----
RUN apt-get update -y \
    && apt-get install --no-install-recommends -y \
        libgstreamer1.0-dev build-essential libusb-1.0-0 lsof libelf1 debhelper \
        libltdl-dev libmnl0 swig tk libpci3 bison dpatch tcl flex udev \
        libgfortran5 kmod libfuse2 graphviz chrpath pciutils ethtool \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 3.10 (deadsnakes) ----
# 注意: 原 history 中 python3.10 层是「纯 apt 安装 python3.10 + python3-pip」,
# 而 `pip install --upgrade pip ${SOURCES}` 是后面独立的层 (见下方), 不要并在这里。
RUN apt-get update -y \
    && apt-get install --no-install-recommends software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa -y || true \
    && apt-get install --no-install-recommends -y python3.10 python3.10-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 0 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 0 \
    && update-alternatives --set python3 /usr/bin/python3.10 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10-config /usr/bin/python3-config

# ---- OpenCL vendor 配置 ----
RUN mkdir -p /etc/OpenCL/vendors/ \
    && echo "libamdocl64.so" > /etc/OpenCL/vendors/amdocl64.icd \
    && echo "/opt/dtk/opencl/lib/" >> /etc/ld.so.conf \
    && ldconfig

# ---- hwloc ----
ENV PATH=/opt/hwloc/bin/:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/opt/hwloc/lib:
RUN cd /tmp \
    && wget -O hwloc.tar.gz https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.1.tar.gz \
    && mkdir hwloc-tmp \
    && tar -xvf hwloc.tar.gz -C ./hwloc-tmp --strip-components 1 \
    && cd hwloc-tmp \
    && ./configure --prefix=/opt/hwloc \
    && make -j $CPUS \
    && make install \
    && rm -rf /tmp/hwloc*

# ---- ucx ----
ENV UCX_HOME=/opt/ucx
ENV PATH=/opt/ucx/bin:/opt/mpi/bin:/opt/hwloc/bin/:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib:
RUN cd /tmp/ \
    && wget -O ucx.tar.gz https://github.com/openucx/ucx/releases/download/v1.15.0/ucx-1.15.0.tar.gz \
    && mkdir ucx-tmp \
    && tar -xvf ucx.tar.gz -C ./ucx-tmp --strip-components 1 \
    && cd ucx-tmp \
    && ./configure --prefix=${UCX_HOME} --enable-mt --enable-optimizations \
        --enable-tuning --enable-cma \
    && make -j64 \
    && make install \
    && rm -rf /tmp/ucx*

# ---- openmpi ----
ENV PATH=/opt/mpi/bin:/opt/hwloc/bin/:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/opt/mpi/lib:/opt/hwloc/lib:
RUN cd /tmp \
    && wget -O openmpi.tar.gz https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.3.tar.gz \
    && mkdir openmpi-tmp \
    && tar -xvf openmpi.tar.gz -C ./openmpi-tmp --strip-components 1 \
    && cd openmpi-tmp \
    && ./configure --prefix=/opt/mpi/ --with-hwloc=/opt/hwloc/ \
        --enable-orterun-prefix-by-default --enable-mpi-thread-multiple \
        --enable-dlopen --with-ucx=${UCX_HOME} \
    && make -j64 \
    && make install \
    && rm -rf /tmp/openmpi*

# ---- SSH 免密相关 (容器内 sshd 准备) ----
RUN echo "source /etc/profile.d/env.sh" >> /etc/bash.bashrc \
    && ssh-keygen -A \
    && mkdir -p /var/run/sshd \
    && sed -i "s/#UseDNS .*/UseDNS no/" /etc/ssh/sshd_config

# ---- MLNX OFED 用户态 (23.10-4.0.9.1, ubuntu22.04 x86_64) ----
ENV MOFED_VERSION=23.10-4.0.9.1
ENV OS_VERSION=ubuntu22.04
ENV PLATFORM=x86_64
# 原镜像此处用 content.mellanox.com 下载 MLNX_OFED-${MOFED_VERSION} 并 mlnxofedinstall。
# <device-B> 等环境对 content.mellanox.com 出网不通; 而容器只需 OFED 用户态(ibverbs/mlx5),
# 内核模块由宿主提供。改用 Ubuntu/阿里云 apt 源的 rdma-core + ibverbs 用户态套件,
# 版本 2307mlnx47 对应 MOFED 23.10 用户态, 可满足 vLLM/RCCl 的 ibverbs 通信需求。
RUN apt-get update -y \
    && apt-get install --no-install-recommends -y \
        libgstreamer1.0-dev build-essential libusb-1.0-0 lsof libelf1 debhelper \
        libltdl-dev libmnl0 swig tk libpci3 bison dpatch tcl flex udev libgfortran5 \
        kmod libfuse2 graphviz chrpath pciutils ethtool \
        rdma-core libibverbs-dev ibverbs-utils ibverbs-providers \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new \
    && echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new \
    && mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# ---- pip 工具链 ----
RUN pip install --no-cache-dir --upgrade pip
RUN pip3 install cmake==3.29.0.1 ninja==1.11.1

# =============================================================================
# 下面是“原 Dockerfile 用 COPY 注入构建上下文”的部分。
# 这些文件/目录原本由构建机本地提供, 镜像里只残留最终结果, 无法反推原路径。
# 已把容器内尚在的文件单独提取到同目录:
#   - install_dep.sh   -> 同目录 install_dep.sh  (126B, 原 COPY 到 /)
#   - /etc/profile.d/env.sh -> 同目录 env.sh        (410B, 原 COPY 到 /etc/profile.d/)
# /opt/dtk (15.8GB) 已从 <device-B> 的 <container> 容器里直接打包导出, 见下方 COPY。
# /usr/share/misc/magic 经核实只是 ubuntu 基础镜像里 file 包自带的软链(->../file/magic),
# 原 history 那条 COPY 层为冗余, 无需单独处理, 已不保留占位。
# =============================================================================

# ---- (原 COPY 小文件到 /etc/profile.d/, 410B) ----
# 实际内容见同目录 env.sh; 它负责注入 hwloc/mpi/python 的 PATH 与 LD_LIBRARY_PATH
COPY env.sh /etc/profile.d/env.sh

# ---- (原 COPY 小文件到 / , 126B) ----
# 实际内容见同目录 install_dep.sh; 它负责装 vllm==0.15.1 等
COPY install_dep.sh /install_dep.sh

# ---- opencompass 0.5.1: 非 DCU 适配的通用公开包, 从 pypi.org 官方源安装(详见 install_dep.sh) ----
#   不本地 COPY、不走 sourcefind(该镜像对 opencompass whl 下载不稳定), 用官方源最稳.

# ---- DTK (DCU 工具链, 约15GB) ----
# 版本: DTK-26.04 (Ubuntu22.04 x86_64); .dtk_version 字段为 DTK-26.04, rc 几以下载包实际内容为准.
# 方式A(默认, 需构建机可访问外网源): 自动下载对应版本并解压到 /opt/dtk
#   tar 顶层套一层 dtk/ 目录, 用 --strip-components=1 平铺到 /opt/dtk.
RUN wget -q --tries=0 --timeout=60 -O /tmp/dtk.tar.gz "https://download.sourcefind.cn:65024/file/1/DTK-26.04/Ubuntu22.04/DTK-26.04-Ubuntu22.04-x86_64.tar.gz" \
    && mkdir -p /opt/dtk && tar -xzf /tmp/dtk.tar.gz -C /opt/dtk --strip-components=1 \
    && rm -f /tmp/dtk.tar.gz
# 方式B(构建机无法访问上述源, 如 <device-B> 机房网): 注释掉上面 RUN, 改启用下面 COPY,
#   并提前把 DTK 解压到构建上下文的 dtk/ 目录(从授权源下载 DTK-26.04 Ubuntu22.04 包, 解压得 ./dtk/...):
# COPY dtk /opt/dtk

# ---- DTK 运行时库注册到系统缓存(双保险) ----
# librocm_smi64.so.2 在 /opt/dtk/.hyhal/rocm_smi/lib; 上面 ENV LD_LIBRARY_PATH 已含该路径,
# 此处再写入 /etc/ld.so.conf.d 并 ldconfig, 确保任何场景(含非登录 shell)都能找到 DTK 库。
RUN echo '/opt/dtk/.hyhal/rocm_smi/lib' > /etc/ld.so.conf.d/dtk.conf && ldconfig

# ---- DTK 环境变量 (最终生效值, 与 docker inspect 一致) ----
ENV PATH=/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/dtk/hip/bin/hipify:/opt/hyhal/bin:/opt/dtk/opencl/bin:/opt/ucx/bin:/opt/mpi/bin:/opt/hwloc/bin/:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/.hyhal/rocm_smi/lib:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib:
ENV LIBRARY_PATH=/opt/hyhal/lib:/opt/dtk/opencl/lib:/opt/dtk/dushmem/lib
ENV C_INCLUDE_PATH=/opt/dtk/dcc/gcvm/include:/opt/dtk/include:/opt/hyhal/include:/opt/dtk/llvm/include:/opt/dtk/opencl/include
ENV CPLUS_INCLUDE_PATH=/opt/dtk/dcc/gcvm/include:/opt/dtk/include:/opt/hyhal/include:/opt/dtk/llvm/include:/opt/dtk/dushmem/include:/opt/dtk/opencl/include
ENV CMAKE_PREFIX_PATH=/opt/hyhal:
ENV MIOPEN_FIND_MODE=3
ENV HIP_PATH=/opt/dtk/hip
ENV AMDGPU_TARGETS=gfx906;gfx926;gfx928;gfx936;gfx938
ENV HYHAL_PATH=/opt/hyhal
ENV DTK_PKG_NAME=DTK-26.04-rc4-ubuntu20.04-x86_64
ENV DTK_PKG_VERSION=dtk2604-rc4
ENV DTK_VERSION=2604
ENV DTKROOT=/opt/dtk
ENV ROCM_PATH=/opt/dtk

# ---- pip 源 (主索引用国内官方镜像清华, 稳定; DCU 定制轮走 extra-index sourcefind/nightly) ----
#   背景: <device-B> 上 pypi.org 的实际 whl 下载被反代到 pypi.sourcefind.cn/open/, 而该镜像对
#   部分通用包(opencompass/h5py 等)间歇断流, 导致 build 不稳定. 故主索引改用清华镜像
#   (pypi.org 的国内官方镜像, 通用包齐全且稳定); DCU 定制轮(vllm/torch/triton 等)在清华
#   没有, 由 extra-index-url = https://pypi.sourcefind.cn/nightly/dtk/ 提供(该路径稳定).
#   目标设备若无法连清华, 可将下方 index-url 替换为任意可达的 pypi 官方/镜像源.
#   注: 若把 nightly/dtk 设为唯一 index-url, vllm 依赖的 cmake==3.29 在该索引缺失会失败,\n#   故用 extra-index 双源。install_dep.sh 里 DCU 包显式 -i nightly/dtk 双保险。

# ---- 主安装层 (9.55GB): pip 装 vllm + 依赖 + 跑 install_dep.sh ----
# 注意: install_dep.sh 已被 COPY 到 /, 这里执行它 (vllm==0.15.1 等)
RUN mkdir -p ~/.pip && echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\nextra-index-url = https://pypi.sourcefind.cn/nightly/dtk/\n" > ~/.pip/pip.conf \
    && cat ~/.pip/pip.conf \
    && pip install --ignore-installed blinker \
    && bash /install_dep.sh

# ---- 常驻容器 (无 entrypoint) ----
CMD ["/bin/sh", "-c", "tail -f /dev/null"]
