---
name: dcu-image-reverse-rebuild
description: 反推 DCU 镜像为可重建的 Dockerfile（内网源换公网源，另一台设备只凭 Dockerfile 即可重建）。
version: 1.0.1
author: DCU Engineer
license: MIT
---

# DCU 镜像反推重建（Image Reverse-Rebuild）

把一个构建源来自**不可达内网**（<internal_ip_src>:9929、pypi.sourcefind.cn/open、Mellanox 内容站等）的 DCU 镜像，反推成一份**任何有外网的设备都能 `docker build`** 的 Dockerfile——做法是把每个内网依赖替换成可达的外部源。

## 适用场景
- 用户有一个 DCU 镜像（如 harbor.sourcefind.cn/.../vllm015-...-dtk26.04-...），想要一份能在另一台机器上重建它的 Dockerfile。
- 原始构建上下文 / 内网源已丢失或不可达。
- 目标："另一台设备只凭 Dockerfile 就能重建镜像"。

## 硬约束（来自本用户）
1. 只允许从原容器提取 DTK（`docker cp` / tar `/opt/dtk`）。不允许把其他任何包作为构建产物拷出容器。
2. 其余所有包必须通过 Dockerfile 内 `pip install -i <url>` 从可达外部源获取——完全自描述（禁止"从容器拷包目录"这类 hack）。
3. DCU 适配包（torch/torchvision/triton/flash_attn/flash_mla/lightop/lmslim/vllm/amdsmi）→ 只走 sourcefind/nightly/dtk；公共包（opencompass/numpy/h5py/setuptools/tokenizers）→ 走公网/官方镜像（清华）。绝不混用。
4. DTK：用户允许用外部 URL 替换。真实源
   `https://download.sourcefind.cn:65024/file/1/DTK-26.04/Ubuntu22.04/DTK-26.04-Ubuntu22.04-x86_64.tar.gz`
   是一个 z-file 实例。**关键（2026-08-06 已验证）**：该链接在**容器内**（含 `docker build` 的 RUN 阶段，~13MB/s 拉 3.3GB→15GB）可达，但从机房宿主机（<device-B>）**超时**——不要从宿主机探测它（会误判"不可达"）。默认：`RUN wget -q --tries=0 --timeout=60 -O /tmp/dtk.tar.gz <link> && tar -xzf ... --strip-components=1`。`COPY dtk /opt/dtk` 仅作为"构建机整网都到不了源"时的兜底保留。

## 工作流（已验证顺序）
完整步骤 + vLLM-0.15.1/DTK-26.04-rc4 示例见 references/rebuild-playbook.md。
**版本号绝不能写死**——用 `scripts/generate_rebuild.py <container>` 探测活容器，输出一份版本号由探测填充的 Dockerfile + install_dep.sh（见下文"通用生成器"）。worked example 的 pin 只针对 vLLM-0.15.1；对**任何**其他容器都要重新探测。
**关键**：`references/dockerfile-vllm015-final.Dockerfile` 和 `install_dep-vllm015-final.sh` 是**那一个**容器/镜像（vLLM 0.15.1 / DTK 26.04-rc4）的已验证产物。即使"版本号看起来一样"也不要复用到别的容器——其他容器的 DTK build 号、DCU wheel 构建、通用包版本、RUN/ENV 层顺序、DTK 发行版 tag（ubuntu20.04 vs 22.04）都可能不同。每个容器都要重跑生成器。
1. 恢复结构：在源设备上 `docker history --no-trunc`、`docker inspect`，并 `docker exec` 提取构建脚本（/install_dep.sh、/etc/profile.d/env.sh）。从 history 1:1 复现每个 RUN/ENV/CMD。
2. 识别构建上下文缺口：任何 `COPY dir:`/`COPY file:` 都无法反编译。DTK → 从容器 tar 或外部 wget 链接；小脚本 → 提取真实内容。标注缺口。
3. **通用生成器（版本永不写死）**：
   `python3 scripts/generate_rebuild.py <container> --outdir ./out` 在源设备上运行，对每个 DCU/关键包探测 `pip freeze` + nightly/dtk 可用性，解析到 exact 或 series 对齐版本，写出 `./out/Dockerfile`（DTK wget 链接取自 `.dtk_version`）+ `./out/install_dep.sh`（两段式 pin，所有版本动态）+ `./out/rebuild_report.txt`（逐包决策 + 不可用告警）。以输出文件为基础；把生成器留作骨架的 RUN/ENV 层从 `docker history` 补齐。
4. 源分层（成功关键——在第一次 build 前就把这些**全部**定下来，见下文"design-first"坑，不要靠一次次失败 build 去发现）：
   - pip.conf：index-url = https://pypi.tuna.tsinghua.edu.cn/simple（稳定公网镜像），extra-index-url = https://pypi.sourcefind.cn/nightly/dtk/（DCU wheel）。
   - 安装脚本 STAGE 1：先 pin 所有 DCU 包（torch/torchvision/triton/flash_*/lightop/lmslim/amdsmi 来自 nightly）以锁定 DCU torch。STAGE 2：公共包（opencompass 来自清华——其 torch>=1.13.1 已满足，不会拉官方 torch）。vLLM：`pip download --no-deps` 精确 wheel，`pip install --no-deps`，再解析 wheel METADATA 的 Requires-Dist，排除 cmake，装其余。
   - 每个 pip install/download 都带 `--resume-retries 50 --timeout 120 --retries 10`。
   - DTK：在构建容器内 `RUN wget` 那个 `/file/1/...` 链接（**不是**从宿主机）。
5. 构建与验证（见坑"在 GPU 容器上验证"）。

## 关键坑（构建前必读）
- **DESIGN-FIRST——在第一次 `docker build` 前就把整套源策略定下来。**
  本任务烧了 20+ 次迭代 build，因为每个坑（pypi.org 反向代理、sourcefind `/open` 间歇断连、`--resume-retries` 默认 5 次在 1GB torch 上失败、cmake==3.29 死路、torch 被清华拉成官方 2.x、numpy 被 opencompass 覆盖、lightop 需要真 GPU 才能 import）都是**一次失败 build 才发现一个**。用户明确顶回："你这么跑下去一直也没有结果"。修法：构建前在 Dockerfile/脚本里一次性锁死全部——(a) pip.conf 双源（清华主、nightly 辅），(b) 每个 pip 调用带 `--resume-retries 50 --timeout 120 --retries 10`，(c) vLLM 走 `--no-deps` + 排除 cmake，(d) DCU torch 在 STAGE 1 先于任何公共包 pin，(e) numpy 在**最后**重新 pin，(f) DTK 走容器内 `RUN wget /file/1/...`，(g) 在 GPU 容器上验证。然后**只 build 一次**。若 build 失败，修策略，不要打一次性补丁再盲目重 roll。
- cmake==3.29 死路：vLLM metadata 要求 cmake==3.29 但只有 3.29.0.1 存在。修法：vLLM 装 `--no-deps`，依赖里排除 cmake，单独装 cmake==3.29.0.1。
- `--resume-retries` 必须加：sourcefind（/open 和 /nightly/das_nightly）大文件（torch 1GB）会间歇断连。pip 默认下载续传 5 次 → "Download failed after 6 attempts / not enough bytes"。修法：每个 pip install/download 带 `--resume-retries 50`（另加 `--timeout 120 --retries 10`）。
- 部分网络下 pypi.org 被反向代理到 sourcefind /open：curl pypi.org/simple 返回官方 200，但实际 whl 下载走 pypi.sourcefind.cn/open（不稳定）。修法：主 index-url 用清华（不是 pypi.org），让公共包避开 /open。
- 构建进行中绝不要 `docker image prune`：可能删掉正在跑的 build 依赖的层 → build 中途 "No such image: sha256:..."。
- vLLM 要在 GPU 容器上验证，**不是** headless `docker run`：lightop（DCU 包）import 时查 GPU CU 并写 `os.environ["LIGHTOP_GPU_CUS"]`；无 GPU 容器里拿到 None → `vllm serve -h` 触发 import 链时 TypeError。这不是重建缺陷（原镜像离 GPU 也照样失败）。在容器内用 `--device=/dev/kfd --device=/dev/dri --device=/dev/mkfd -v /opt/hyhal:/opt/hyhal:ro` 验证。
- DTK 源——2026-08-06 已验证（build #26，镜像 f9e2a57c9290）：`/file/1/...` 直链**可以**写死且可用。`download.sourcefind.cn:65024` 是 z-file 实例，`/1/main/...` 返回 200 SPA 壳（不是文件），`/file/1/...` 才是真实文件 URL。它从机房**宿主机**超时，但在**容器内**（含 `docker build` RUN 阶段，~13MB/s，3.3GB 压缩→15GB 解压，`.dtk_version` = `DTK-26.04-rc4` 与原一致）下载正常。所以：Dockerfile 里默认 `RUN wget <file-link>`；`COPY dtk /opt/dtk` 仅作为"构建机整网到不了源"的兜底保留。不要以为它"需要鉴权/到处超时"就跳过 wget 方案。
- flash_attn/flash_mla 精确 build 号常常不在 nightly：用 torch290 系列里可用的同大版本最新。**用户规则（明确）："series 对齐，非 exact 时选可用的更高版本"**——漂移可接受，但优先选系列内最新 build。生成器（`generate_rebuild.py`）实现了这点：当 nightly 只有不同 build 号时，选系列内最高（max build-timestamp）的 wheel，而不是第一个匹配。
- torch build 号漂移：即使 pin 了，1GB torch 下载重试时可能回落到另一个 2.9.0+ nightly build。同系列可接受。
- numpy 会被覆盖：某个公共包（opencompass）可能把 numpy 拉到 2.x，破坏 torch。在安装脚本**末尾**重新 pin numpy==<orig>。

## 验证（"完成"的定义）
- DTK 内容一致性：重建镜像内 `cat /opt/dtk/.dtk_version` == 原（`DTK-26.04-rc4`）；顶层布局（amdgcn/hip/llvm/miopen/rccl/rocblas...）齐全；`/opt/dtk/env.sh` + `/opt/dtk/.hyhal/rocm_smi/lib/librocm_smi64.so.2` 存在；`/opt/dtk` ~15GB。（build #26 已验证：重建 DTK == 原 rc4，结构 + 关键文件 + 大小全匹配。）
- `import torch, vllm` 可用（无 GPU 也行）。
- `vllm serve -h` 在 GPU 测试容器内返回 EXIT=0（hy-smi 可见 8 HCU）。
- `pip freeze` 与原 diff：torchvision/triton/vllm/lightop/lmslim/transformers/xgrammar/numpy/amdsmi/cmake 逐字节一致；torch/flash series 对齐；无缺失 DCU 包。

## 支撑文件
- references/rebuild-playbook.md — 完整 worked example + 精确命令。
- references/version-pins.md — vLLM 0.15.1 / DTK 26.04-rc4 的已验证 nightly 版本 pin（含 DTK 外部 `/file/1/...` wget 链接，已验证在构建容器内可达）。
- references/dockerfile-vllm015-final.Dockerfile — 最终已验证 Dockerfile（build #26，镜像 f9e2a57c9290；DTK 走 wget，所有 DCU 包 pin，清华+nightly 源分层）。
- references/install_dep-vllm015-final.sh — 最终已验证安装脚本（两段式 pin，`--no-deps` vLLM + METADATA 依赖补齐，`--resume-retries 50`）。
- references/env-vllm015-final.sh — DTK env 注入（COPY 目标 /etc/profile.d/env.sh）。
- scripts/probe_nightly.py — 验证某 DCU wheel 版本真的可用（不只是 INSTALLED）。
- scripts/generate_rebuild.py — 通用生成器：给定任意容器，探测 pip freeze + nightly 可用性，输出 Dockerfile（DTK wget 链接）+ install_dep.sh（两段式 pin，所有版本由探测动态填充，永不写死）+ rebuild_report.txt。对**任何**容器都用它，而不是下面 vLLM-0.15.1 专属的 pin。
- scripts/verify_rebuild.sh — 跑 freeze-diff + GPU 容器 `vllm serve -h` 检查。
- references/cleanup-failed-builds.md — 如何真正释放 50–200 GB 失败 build 残留：匿名 exited 中间容器持有层，`prune -f` 会跳过它们；先 rm 匿名停止容器，再 prune。含"绝不要删"规则（build 中绝不 prune；绝不碰其他用户的命名容器）。

## 合并说明（历史）
本 skill 是 2026-08 由 5 个高度重叠的反推重建 skill 合并而成：
`dcu-image-reverse-rebuild`(本) + `dcu-dockerfile-reverse` + `dcu-image-reverse-build`
+ `dcu-image-rebuild` + `dcu-image-rebuild-verify` + `dcu-image-reverse-engineering`。
它们讲的全是同一件事（容器/镜像反推 Dockerfile + 换源 + 验证），且互相打架（如 DTK
源是否可达），故合并为唯一入口。被合并 skill 的独有参考文档已归档在
`references/legacy/`（from-<来源>_*.md / *.sh），仅作历史经验保留，不另维护。
主文档以本 SKILL.md + 上述支撑文件为准。
