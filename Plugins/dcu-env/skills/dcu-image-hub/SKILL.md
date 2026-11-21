---
name: dcu-image-hub
description: 查光合开发者社区(sourcefind.cn) DCU 镜像仓库：Web API 列镜像/列 tag、Harbor 匿名 token 直查、docker pull 拉取，附全量快照与刷新脚本。触发场景：找某个 DTK 版本的基础镜像、确认镜像 tag 是否存在、断网时先查本地快照。不适用：镜像拉下来之后怎么起容器走 dcu-docker；模型适配仓库查询走 dcu-modelzoo。
version: 1.1.2
author: DCU Engineer
license: MIT
kind: tool
---

# DCU 镜像仓库查询（sourcefind 光合开发者社区）

## When to Use

- 要找 DCU 适配的**容器镜像**：某框架（vllm/sglang/pytorch/tensorflow/migraphx/paddle/jax…）有哪些版本、配哪个 DTK/OS/python。
- 要拿某镜像的 **docker pull 完整命令**（tag 很长且带日期后缀，手猜必错）。
- 要查某镜像**最新发布**了什么 tag、镜像多大、何时 push。
- 要在**没有开发者社区账号**的情况下确认 harbor 镜像能否匿名拉取。

> 与相邻 skill 的关系：`dcu-das-hub-api` 管 **pip 包源**（pypi.sourcefind.cn，wheel 下载）；本 skill 管 **docker 镜像仓库**（harbor.sourcefind.cn:5443 + 开发者社区 Web API）。`dcu-docker` 管拉下来之后怎么起容器。
> 姊妹 skill：**`dcu-modelzoo`** 管**模型适配仓库**（GitLab group modelzoo，查某模型适配基线 DTK/框架/卡型与推荐镜像）——先 modelzoo 定模型基线，再回本 skill 查镜像 tag。

## 1. 快速查（本地快照，零网络）

`data/sourcefind_dcu_images.json`（快照日期见 `fetched_at` 字段）含全量镜像 + tag。先查快照，不够再刷：

**查某框架"最新版本"的纪律（2026-10-12 踩坑，必须遵守）**：
- 快照**过期即不可信**：`fetched_at` 早于今天就不能用它回答"最新版是哪个"——新版本随时在发布。查最新版必须先刷新快照（`scripts/fetch_sourcefind_images.py`，~22s）再查，或直接走 Web API。
- 刷新时**不要加 project 过滤**：dcu/hcu/cpu 三个项目全抓。实测 `vllm0.28.1rc1` 只发在 **hcu**/admin/base/custom，只扫 dcu 项目会漏掉整条版本线。
- 本地查快照时**不要截断 tag 列表**（`[:N]` 是事故之源）：harbor/快照的 tag 是字典序，`vllm-` 前缀杂项（gpustack/ray/vllm-omni）和旧版本变体会占满前 40 条窗口，把 `0.28/0.29` 全截到窗口外。**正确姿势是按版本号子串精确过滤**（如 `grep "0.28"`），或排序后取最大版本，永远不要"列前 N 条扫一眼"。
- 一条命令查全（刷新+按版本精确匹配，无截断）：
  ```bash
  cd <skill_dir> && python3 scripts/fetch_sourcefind_images.py /tmp/cur.json >/dev/null 2>&1 && python3 -c "
  import json,sys
  d=json.load(open('/tmp/cur.json'))
  ver=sys.argv[1] if len(sys.argv)>1 else ''
  for i in d['images']:
      for t in i['tags']:
          if ver and ver not in t['name']: continue
          print(f\"{i['project']:4s} {i['repo']}:{t['name']}\")" "0.28"
  ```
- 回答"最新版"时列出**每个项目**的命中，别只报第一个看到的。

```bash
# 全部 DCU 镜像名 + 简介一览
python3 -c "
import json
d = json.load(open('data/sourcefind_dcu_images.json'))
for i in d['images']:
    print(f\"[{i['project']}] {i['repo']:40s} {len(i['tags']):3d} tags  {i['abstract'][:40]}\")"

# 找某框架最新 tag
python3 -c "
import json
d = json.load(open('data/sourcefind_dcu_images.json'))
img = next(i for i in d['images'] if i['repo']=='admin/base/sglang')
for t in img['tags'][:5]:
    print(t['push_time'], t['size'], t['pull_url'])"
```

## 2. Web API（主通道，匿名可访问，无需任何请求头）

SPA 前端 `https://developer.sourcefind.cn/servicelist` 的数据源。三个端点全部实测可用：

| 用途 | 请求 | 关键返回字段 |
|------|------|------|
| 列镜像 | `GET /api-repo/image/ListImage?Page=1&PageSize=200` | `data.total`、`data.list[]`：`Name/RepoName/ProjectName/Accelerator/Abstract/UUID/PullCount`。⚠ **ProjectName 不是稳定键**：2026-10-14 实测 56/63 个镜像被官方从 `dcu` 翻转成 `hcu`（现存 hcu=56、cpu=7、dcu=0）；Harbor 端 `dcu/...` 与 `hcu/...` 双前缀并存均 200。**快照对比/去重一律用 `UUID`** |
| 按关键词搜 | `GET /api-repo/image/ListImage?Keyword=vllm`（也支持中文，如 `Keyword=训练`；匹配名称+简介） | 同上 |
| 列某镜像全部 tag | `GET /api-repo/image/ListTag?ImageUUID=<镜像UUID>&Page=1&PageSize=500` | `list[]`：`Name`(tag)、`Size`、`PushTime`、`PullUrl`(完整 docker pull 命令)、`Description`(HTML 发布说明) |
| 镜像详情 | `GET /api-repo/image/DetailImage?UUID=<镜像UUID>` | 同 ListImage 单条 + 完整 Introduction |

⚠ 踩坑：
- 分页参数是**大写 `Page` / `PageSize`**；`page/pageSize/limit/offset` 全部被忽略（永远返回前 10 条）。
- `ListTag` 参数名是 **`ImageUUID`**（写 `ID=`/`UUID=`/`image_uuid=` 都报 `fail.param.format.ImageUUID`），值是 ListImage 返回的镜像 UUID。
- 不带 `PageSize` 默认只回 10 条——列 tag 必须带大 PageSize。
- 基础域名是 `developer.sourcefind.cn`（不是 pypi.sourcefind.cn，也不需要 `X-Das-Source` 头——那是 pypi 包源 API 的要求）。
- 详情页 URL：`https://developer.sourcefind.cn/servicelist/detail?post_id=<UUID>`（注意参数名是 post_id 不是 uuid）。
- **刷新快照后 diff 以 `uuid` 为键**：2026-10-14 实测 56/63 行 ProjectName dcu→hcu 翻转而 uuid 全部不变（0 重复/0 新增/1 下架）；project 只当单列 diff 看，别当主键。

## 3. 一键刷新快照

```bash
python3 scripts/fetch_sourcefind_images.py data/sourcefind_dcu_images.json
# ~1min 拉全量（63 镜像 / 1234 tags @2026-10-14；09-02 首测 64/1205，1 个镜像下架），失败自动重试 3 次
```

脚本只依赖 python3 标准库。输出 JSON 结构：`{fetched_at, source, images:[{id, uuid, name, repo, project, accelerator, abstract, pull_count, updated, detail_url, tags:[{name, size, pull_url, push_time, release_type}]}]}`，tags 按 push 时间倒序（最新在前）。

## 4. Harbor Registry 匿名直查（第二通道，绕过 Web 站）

知道 repo 名、想直接确认 registry 里的 tag/manifest 时用（Harbor 匿名用户可拉公开项目）：

```bash
H='harbor.sourcefind.cn:5443'
R='dcu/admin/base/vllm'
# 1) 匿名 token
TOKEN=$(curl -s "https://$H/service/token?service=harbor-registry&scope=repository:$R:pull" | python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])')
# 2) tag 列表（Web API 分页/字段齐全；此通道胜在绕过站点直接对 registry）
curl -s -H "Authorization: Bearer $TOKEN" "https://$H/v2/$R/tags/list"
# 3) manifest（查大小/digest/架构）
curl -s -H "Authorization: Bearer $TOKEN" \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json' \
  "https://$H/v2/$R/manifests/<tag>"
# 4) Harbor 搜索接口（按项目搜 repo）
curl -s "https://$H/api/v2.0/search?q=vllm"
```

- `/v2/_catalog` 匿名返回 401（禁匿名列全库），但单 repo 的 token + tags/list + manifest 匿名可用。
- **`docker pull harbor.sourcefind.cn:5443/<RepoName>:<tag>` 无需 docker login**（docker 客户端自动走匿名 token 流程，2026-09-02 实测 manifest 可匿名获取）。

## 5. 仓库与 tag 命名规律（选型速查）

- repo 路径：`harbor.sourcefind.cn:5443/<project>/admin/base/<name>`。2026-10-14 实测 ListImage 返回 project 只剩 `hcu`（56，原 dcu 卡镜像批量改名）+ `cpu`（7）；09-02 时为 dcu 56/cpu 7/hcu 1。Harbor registry 端 `dcu/`、`hcu/` 双前缀并存均可 pull（同 repo 双 200）。
- tag 命名：`<框架版本>-<OS>-<DTK版本>[-das版本][-rocblas标记][-日期]-py<py版本>[-rN/-rcN/-beta]`
  例：`0.18.1-ubuntu22.04-dtk26.04-py3.10`、`0.9.2-ubuntu22.04-dtk25.04.2-rocblas1121-das1.7-py3.10-20251203`
- 选型看三段：**框架版本 + DTK 版本 + OS/python**；`r1/r5`、`-rcN`、日期后缀是同版本的重建/变体（如 axolotl tag 里 `vllm018` 表示复用 vLLM 0.18 运行环境）。
- 大类：推理（vllm/sglang/sginfer/tf-serving/migraphx）、训练（pytorch/tensorflow(+horovod/hyperopt)/paddlepaddle/oneflow/jax/axolotl/llamafactory）、IDE（vscode-*/jupyterlab-*/rstudio-* × dtk/pytorch/tensorflow/…）、DTK 基础（dtk/rockylinux/dev）、K8s 组件（dcu-device-plugin[-share]/k8s-vdcu-*/dcu-exporter/dcu-label-node）。

## 6. 拉取后启动

拉下来怎么起容器（设备透传/挂 hyhal/shm）见 `dcu-docker` 的标准启动模板。典型：

```bash
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.18.1-ubuntu22.04-dtk26.04-py3.10  # 2026-10-14 起 ListImage 返回 hcu/；dcu/ 前缀仍可拉（双项目并存，均实测 200）
# 然后按 dcu-docker 模板 docker run（--device=/dev/kfd,/dev/mkfd,/dev/dri + -v /opt/hyhal:/opt/hyhal:ro + --shm-size）
```

## Verification Checklist（本 skill 自检，2026-09-02）

- [x] ListImage 全量：`Page=1&PageSize=200` → 64/64（dcu 56 / cpu 7 / hcu 1）
- [x] ListTag：vllm 29 tags、pytorch 105、dtk 62、tensorflow 101、custom(dcu) 246，PullUrl 完整可用
- [x] Keyword 中文/英文过滤实测（训练→10、推理→5、k8s→5、onnx→0）
- [x] 分页参数大小写坑（小写被忽略）
- [x] Harbor 匿名 token → tags/list + manifest 200
- [x] `docker` 未登录状态下 manifest 可匿名访问（即 pull 无需 login）
- [x] 快照脚本全量跑通（22s，0 失败）
- [x] 2026-10-14 复测：63 镜像/1234 tags，uuid 零重复零新增；56/63 ProjectName dcu→hcu 翻转（官方改名），cpu 7 个不动；Harbor `hcu/admin/base/vllm` 与 `dcu/admin/base/vllm` tags/list 均 200
