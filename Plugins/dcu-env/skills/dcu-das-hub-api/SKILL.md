---
name: dcu-das-hub-api
description: 查 DCU 源(pypi.sourcefind.cn)包版本/下 wheel 的 JSON API 用法与踩坑
version: 1.1.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# DCU DAS Hub 包源 API（dcu-das-hub-api）

## When to Use
- 要查 DCU 源（pypi.sourcefind.cn）里某包有哪些版本 / 最新构建。
- 要"找比容器里更新的 DCU 适配构建"（跨仓库枚举）。
- 要拿 wheel 下载 URL 并下载（尤其容器无外网、需本地下载再 scp）。
- 装了 DCU kernel 轮报 `undefined symbol: _ZN3c10...`（torch ABI 不匹配）要定位。
- 预编译 kernel 有 bug 且源里无修复构建，想用 `.pth` monkey-patch 纯 PyTorch 替代。

> 用途：需要"查 DCU 源里某包有哪些版本 / 找比容器更新的构建 / 拿 wheel 下载 URL 并下载"时，按本 skill 走。
> 与 `dcu-env-packages` 的关系：那个 skill 讲 `/release/dtk/` pip 索引（直接 `pip install -i`）+ 适配包清单；本 skill 讲 DAS Hub 的 **JSON API**（枚举 / 跨仓库查 / 下载 wheel），两者互补。若 `dcu-env-packages` 被 curator adopt，本 skill 内容可并入其 `references/`。
> 姊妹 skill：**`dcu-image-hub`** 管 docker 镜像仓库（developer.sourcefind.cn servicelist API + Harbor 匿名 token + 全量快照/刷新脚本）——查镜像版本 tag、拿 `docker pull` 命令走那边；本 skill 只管 pip 包源。

## 1. 关键事实（最容易踩错）

- `pypi.sourcefind.cn` **根域名是 Vue SPA（"DAS Hub"）**，curl 裸页只拿到前端空壳 HTML，不是包索引。要机器化查包/列版本/下载，走它的 JSON API。
- **所有 JSON API 调用必须带请求头** `X-Das-Source: mirror-web-client-v1`，否则返回 `{"success":false,"error":"Access Denied: Invalid Client Source"}`。
- 该头值来自前端 bundle（`Ne.defaults.headers.common["X-Das-Source"]="mirror-web-client-v1"`），是固定常量。

## 2. API 端点（2026-08-27 实测）

| 用途 | 请求 | 返回 |
|------|------|------|
| 搜包 | `GET /api/pypi/packages/search/?keyword=<pkg>` | `{items:[{name, latest_version, file_count, version_count}]}` |
| 列仓库 | `GET /api/pypi/repos/?size=-1` | `{repos:[{label, value}]}`，`value` 是仓库路径 |
| 列某仓库某包全部版本 | `GET /api/pypi/package/<repo>/<name>?size=-1` | `{files:[{version, filename, url, download_url, upload_time}]}` |
| **列某仓库全部包**（批量包清单） | `GET /api/pypi/packages/?repo=<repo>&size=-1` | `{items:[{name, origin_repo}], total, page, size}` |

⚠ **列版本 URL 顺序是 repo 在前、name 在后**（写反会返回空 `{files:[]}`）；repo 里的 `/` 要 URL 编码成 `%2F`。

**批量包枚举（2026-09-01 实测）**：第 4 个端点列某仓库全部包，用于"找源里所有 DCU 适配包"（如判断哪些包值得整合进 skill）。两个要点：① **必须跨仓库枚举**——`release/dtk2604`（主索引，39 包）+ `customer/dtk2604-rc4`（rc4，主索引没有的 14 包），并集才是全集（53 包）；② 响应的 `origin_repo` 字段给出包的真实来源仓库（release/dtk2604 列出的包可能 origin_repo=release/das1.8.5）。

仓库路径示例（`repos` 实测）：`customer/dtk2604-rc4`、`customer/dtk2604`、`das_release/dtk2604`、`release/dtk2604`、`das_nightly/dtk2604-*` 等。

## 3. 下载 wheel（最容易踩的坑：内网 URL）

响应里的 `url`/`download_url` 指向**内网** `http://<internal_ip_pypi_storage>:9929/<repo>/+f/...`（内网存储节点 IP:端口，以实际响应为准），外网主机和 DCU 容器**都连不上**（容器通常无外网）。

**把 host 换成 `https://pypi.sourcefind.cn/<同一路径>` 即可 200 下载**（公网域名代理同一存储）。URL 末尾的 `#sha256=<hex>` 就是 wheel 校验和，下载后 `sha256sum` 核对。

```bash
H='X-Das-Source: mirror-web-client-v1'
# 1) 列某仓库某包全部版本
curl -s -H "$H" 'https://pypi.sourcefind.cn/api/pypi/package/customer%2Fdtk2604-rc4/sglang?size=-1'
# 2) 取目标版本 url，把内网 host（http://<internal_ip_pypi_storage>:9929）换成 https://pypi.sourcefind.cn 后下载
curl -s -o sglang.whl 'https://pypi.sourcefind.cn/customer/dtk2604-rc4/+f/c12/88950fc06c87e/sglang-....whl'
sha256sum sglang.whl   # 必须等于 url 里的 #sha256=
```

> 容器无外网时：本地（agent 宿主机）能访问 `pypi.sourcefind.cn`，先本地下载 + 校验，再 `scp` 进容器 `pip install --no-deps`。

## 3.5 `/nightly/dtk/` simple 索引直接下 wheel（比 JSON API 更轻的路径）

`/nightly/dtk/` 这类 pip 索引有**可直接 curl 的 simple 索引** `https://pypi.sourcefind.cn/nightly/dtk/simple/<pkg>/`（**不需要** `X-Das-Source` 头，与 §2 的 JSON API 不同），返回 HTML 里每个 wheel 的 `href` 就是真实下载 URL。

- **`pip install -i https://pypi.sourcefind.cn/nightly/dtk/ <pkg>` 在大 wheel 上会慢/卡**：实测 355MB 的 flash-attn wheel 只有 ~330KB/s，且 pip 缓存一度卡在 80K 不动（疑似并发下载分摊带宽）。
- **可靠路径**：grep simple 索引拿目标 wheel 的 `href`（真实 URL）→ `curl -sL -o <合法wheel文件名>.whl <url>` → `pip install --no-deps <wheel>`。
- **坑：`pip install <file>` 要求合法 6 段 wheel 文件名**；重命名成 `fa_new.whl` 会报 `ERROR: Invalid wheel filename (wrong number of parts)`。保留原始文件名（`flash_attn-2.8.3+...-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`）。
- **`--no-deps` 用于隔离测单包升级**：避免 pip 连带改 sgl-kernel 等依赖（测"只升 flash-attn 是否修 Bug B"时必须 `--no-deps`，否则 sgl-kernel 被一起动，变量不干净）。

```bash
# 1) 从 simple 索引拿目标 wheel 的真实 href
curl -sL 'https://pypi.sourcefind.cn/nightly/dtk/simple/flash-attn/' | \
  grep -oE 'href="[^"]*2607280958[^"]*manylinux_2_24[^"]*"' | head -1
# 2) curl 下载（保留合法 wheel 文件名）
curl -sL -o 'flash_attn-2.8.3+...gebb4be-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl' '<href里的URL>'
# 3) 隔离安装
pip install --no-deps 'flash_attn-2.8.3+...gebb4be-....whl'
```

## 4. 跨仓库找"比容器更新的构建"

`/release/dtk/` pip 索引只暴露部分仓库；DAS Hub API 的 `repos` 列表**更全**（customer/*、das_nightly/*、das_release/*、release/*）。遍历 `repos` 对每个仓库查目标包，能定位到比容器更新的构建。

实测（2026-08-27）：容器 sglang 0.5.12 / sgl-kernel 0.3.21，遍历后发现 `customer/dtk2604-rc4` 仓库有更新的 `sglang 0.5.15.post2`（8月18日）和 `sglang-kernel 0.4.6`（8月18日，与 sglang 同 commit `ga511eb`）。

```bash
# 遍历所有仓库找某包（只打印有结果的）
curl -s -H "$H" 'https://pypi.sourcefind.cn/api/pypi/repos/?size=-1' | \
  python3 -c 'import json,sys;[print(r["value"]) for r in json.load(sys.stdin)["repos"]]' | \
  while read repo; do
    enc=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$repo")
    n=$(curl -s -H "$H" "https://pypi.sourcefind.cn/api/pypi/package/$enc/<PKG>?size=-1" | \
        python3 -c 'import json,sys;print(len(json.load(sys.stdin).get("files",[])))')
    [ "$n" != "0" ] && echo "FOUND: $repo -> $n files"
  done
```

**限流 fallback 与枚举口径（2026-10-14 实测）**：JSON packages API 带头也会间歇 403（限流收紧）时，改走 **simple 索引** `https://pypi.sourcefind.cn/<repo>/simple/<pkg>/`（匿名稳定、无需 `X-Das-Source` 头，同 §3.5），逐包探测代替批量遍历。仓库总数 **107 → 120**（2026-10-14 repos API 实测）。⚠ **枚举口径**：顶层 simple 索引页会**漏包**——customer/dtk2610 顶层看不到 torch，叶子级 `/customer/dtk2610/simple/torch/` 实测有 16 个 wheel；**仓库内容一律以叶子级 `/<repo>/simple/<pkg>/` 探针为准**，顶层枚举只作仓库发现。

⚠ **遍历 100+ 仓库会触发 429 限流（2026-09-07 实测）**：`/api/pypi/repos/?size=-1` 返回 107 个仓库，逐个查包（无间隔）跑到一半开始回 `HTTP 429: Too Many Requests`，后续请求全失败。规避：① 每次请求间 `sleep 1~2s`；② 429 时退避重试（`sleep 20` 后重发，最多 3~4 次）；③ 优先用 `search` 端点缩小范围（`/api/pypi/packages/search/?keyword=<pkg>` 直接给 latest_version + 仓库），再对少数候选仓库查全版本，不要无脑遍历全部 107 仓库。

## 5. 踩坑：torch ABI 不匹配（编译型 kernel 轮）

装了"为 torch X 编译"的 kernel 轮（如 `sglang-kernel 0.4.6` 标 `torch2100`）到 torch Y（如 2.9.0）环境，import 时报：
```
ImportError: .../sgl_kernel/common_ops...so: undefined symbol: _ZN3c1013MessageLogger6streamB5cxx11Ev
```
`c10::MessageLogger::stream` 是 **torch 2.10 才有的符号**。**根因是 torch 大版本不匹配，不是包损坏**。

诊断（确认符号归属）：
```bash
nm -u <kernel>.so | grep c10                 # kernel 需要哪些 c10 符号
nm -D <torch>/lib/libc10.so | grep <符号>     # 容器 torch 有没有该符号
```
规避：kernel 轮的 `torch<标记>` 必须和环境 torch 大版本一致（见 `dcu-env-packages` §4 矩阵）。源里若没有匹配 torch 标记的构建，只能升 torch（覆盖本地 DCU 适配包，高风险，需用户拍板）或走纯 PyTorch 替代（见 §6）。

## 6. 通用技巧：`.pth` 在 import 前 monkey-patch 有 bug 的预编译 kernel

当预编译 kernel 在 DCU 上有 bug、且源里没有修好的构建时，可用 `.pth` 在 site 初始化时（早于任何业务 import）monkey-patch 成纯 PyTorch 实现，**不改任何 DCU 包、可回滚**：

- 写 `<name>_patch.py`，用环境变量开关（如 `XXX_PATCH=1` 才生效，非目标进程 no-op，避免影响其他 python 进程）。
- 在 `dist-packages` 放 `<name>_patch.pth`，内容一行 `import <name>_patch`。
- 能覆盖 `from sgl_kernel.kvcacheio import f` 这种 **import 时绑定**：因为 `.pth` 在 site 初始化阶段执行，早于 sglang import，patch 先于绑定生效。
- 纯 PyTorch 替代 gather-scatter 语义：`dst[dst_indices] = src[src_indices].to(dst.device)`。
- 回滚：删 `.pth` 即可。
- ⚠ **容器重建会丢 dist-packages 的 patch**（共享卷 `/public` 保留脚本，但 `dist-packages` 不保留），重建后需重部署。

> 注：本范式机制已验证（patch 在 3 个进程生效、小张量语义测试 PASS）；具体到某个 bug 是否被修好，需跑通对应场景的回归测试确认，不能仅凭 patch 生效就下结论。

## 验证步骤（本 skill 自检）
- [x] 根域名 SPA 空壳、`X-Das-Source` 头必需（无头 Access Denied / 有头 200）
- [x] 四个 API 端点 + repo 在前顺序（写反返回空）；第 4 端点（列某仓库全部包）2026-09-01 实测，跨仓库枚举 53 包
- [x] 内网→公网下载 URL 改写（两个 wheel 下载 + sha256 全匹配）
- [x] 跨仓库枚举定位到比容器更新的构建（sglang 0.5.15 / sglang-kernel 0.4.6）
- [x] torch ABI 不匹配符号诊断（确认 `c10::MessageLogger::stream` 容器 torch 2.9.0 缺失）
- [x] `.pth` monkey-patch 机制（3 进程生效 + 语义 PASS；具体 bug 修复未确认，已标注）
- [x] 未写入性能基线/日志（遵守 skill 内容纪律）
