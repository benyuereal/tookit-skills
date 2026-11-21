# 源替换参考: pypi.sourcefind.cn/nightly/dtk

## pip.conf (Dockerfile 里写进 ~/.pip/pip.conf 或 /etc/pip.conf)
```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
extra-index-url = https://pypi.sourcefind.cn/nightly/dtk/
```
- `index-url=清华`: 通用包从清华(pypi.org 国内官方镜像)解析, **稳定**; 注意 <device-B> 上 pypi.org 被反代到 sourcefind `/open/` 且间歇断流, 故主索引不能用 pypi.org(见下方"<device-B> 专属网络坑")。
- `extra-index-url=nightly/dtk`: DCU 定制轮从该镜像取(含与原镜像逐字节一致的 vllm/torch/torchvision/triton/flash_attn 等精确轮); 清华无 DCU 包, 自动回退到这。
- `extra-index-url=nightly/dtk`: DCU 定制轮优先从该镜像取(含与原镜像逐字节一致的
  vllm/torch/torchvision/triton/flash_attn 等精确轮)。

## 关键事实(nightly/dtk 实测)
- vllm 0.15.1 原轮 `vllm-0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba`
  在 nightly 上有逐字节一致的副本 → 用 `==` 精确 pin。
- torch 2.9.0 系列: 原镜像 `2.9.0+das.opt1.dtk2604.20260331.g4e3c1e7` 在 nightly 上有。
- torchvision 必须同 torch 系列: `0.24.0+das.opt1.dtk2604.torch290.2604091152.g66069c`。
- **旧构建轮会下架**: flash_attn/flash_mla 的 4 月构建 nightly 不保留, 只剩 7 月 torch290
  最新可用版(主版本可差, torch 系列必须对)。
- cmake: 源无 3.29.0, 只有 3.29.0.1 → vllm 用 --no-deps 绕过, cmake 装 3.29.0.1。
- citest 0.0.8 / cinotificate 0.0.3 / opencompass 0.5.1: nightly 索引里有(用 `==` pin);
  opencompass 的 whl 在 sourcefind `/open/` 镜像**下载不稳**(2.6/3.2MB 反复断),
  改用从原容器 docker cp 出目录 + Dockerfile 直接 COPY 进 dist-packages。

## 确认"nightly 真实可得"的探查命令(勿信 INSTALLED 行)
在能联网的容器里:
```
docker exec <c> bash -c "pip index versions <pkg> -i https://pypi.sourcefind.cn/nightly/dtk/ 2>/dev/null" \
  | tr ',' '\n' | grep -v INSTALLED | grep '<版本特征如 torch290>'
```
- `INSTALLED: X` 是当前容器已装版, **不代表索引可获得**, 绝不能据此 pin。
- 取 `grep -v INSTALLED` 后命中的真实可用版本号。

## 等价测试容器(创建方式, 供换设备后验证)
```bash
docker run -d --privileged --cap-add SYS_PTRACE \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  -v /opt/hyhal:/opt/hyhal:ro -v /public -v /home -v /data \
  --shm-size=200gb --name <test> <rebuilt_image> bash
```
(DTK 容器必挂 -v /opt/hyhal:/opt/hyhal:ro, 否则 DCU 不可见 DEVCOUNT=0)

## DTK 外网直链(替代"从容器 tar", 用户最终要求)
- 可用直链(实测 wget 可下, 3.3GB 压缩 → 15GB 解压):
  ```
  https://download.sourcefind.cn:65024/file/1/DTK-26.04/Ubuntu22.04/DTK-26.04-Ubuntu22.04-x86_64.tar.gz
  ```
- **路径坑**: 该站是 z-file。浏览路径 `/1/main/...` 返回 SPA 前端壳(非文件); **真实文件直链是 `/file/1/...`**(注意 `/file/` 不是 `/1/main/`)。
- Dockerfile 写法: `RUN wget -q --tries=0 --timeout=60 -O /tmp/dtk.tar.gz "<直链>" && mkdir -p /opt/dtk && tar -xzf /tmp/dtk.tar.gz -C /opt/dtk --strip-components=1 && rm -f /tmp/dtk.tar.gz`(tar 顶层套 `dtk/`, 需 `--strip-components=1`)。
- 版本核对: 解压后 `cat /opt/dtk/.dtk_version` 应为 `DTK-26.04-rc4`(rc 几写在版本文件里, 不是可见版本号; 外网这个包就是 rc4)。
- **可达性**: <device-B> **宿主机** curl/wget 该源**超时**, 但**容器内**(含 docker build 的 RUN 容器 / docker exec)能正常下载。不要在宿主机侧探测, 直接让 Dockerfile `RUN wget` 在 build 容器里拉。
- 若构建机整网不通该源 → 退回 `COPY dtk /opt/dtk`(先下好 DTK 解压进构建上下文)。

## <device-B> 专属网络坑(pip.conf 主索引必须用清华, 不能用 pypi.org)
- <device-B> 上 `pypi.org` 的 whl 实际下载被**反代到 `pypi.sourcefind.cn/open/`**, 该镜像对部分通用包(opencompass/h5py)间歇断流。
- 故 Dockerfile pip.conf **主索引用清华**(稳定), extra-index 用 nightly(放 DCU 包):
  ```
  [global]
  index-url = https://pypi.tuna.tsinghua.edu.cn/simple
  extra-index-url = https://pypi.sourcefind.cn/nightly/dtk/
  ```
- 通用包(opencompass/numpy/h5py/setuptools 等)走清华; DCU 适配包(torch/triton/vllm/flash_*/lightop/lmslim)走 nightly。两路都避开 `/open` 断流。

## 决定性参数: --resume-retries(下载大轮不失败的关键)
- sourcefind 对大文件(torch 1GB / DTK 3.3GB)间歇断流; pip 默认 download 续传仅 5 次 → `Download failed after 6 attempts ... not enough bytes`。
- `--retries N` 是**连接**重试; `--resume-retries N` 才是**下载续传**重试(pip 报错提示里明写)。
- 所有 pip 统一: `PIP_OPT="--timeout 120 --retries 10 --resume-retries 50"`; wget 用 `--tries=0 --timeout=60`。
