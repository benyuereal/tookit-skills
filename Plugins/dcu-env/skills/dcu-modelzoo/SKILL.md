---
name: dcu-modelzoo
description: 查光合开发者社区 ModelZoo（GitLab group modelzoo，626 个 DCU 模型适配仓库）：GitLab API v4 匿名列仓库/搜模型/model.properties 元数据/README 提取 DTK 版本与推荐镜像，附全量快照与刷新脚本。触发场景：某模型在 DCU 上有没有人适配过、适配到哪个 DTK、推荐用哪个镜像。不适用：查基础镜像 tag 走 dcu-image-hub；已有部署配方速查走 dcu-model-deploy-catalog。
version: 1.0.1
author: DCU Engineer
license: MIT
kind: tool
---

# DCU ModelZoo 模型适配仓库查询（sourcefind 光合开发者社区）

## When to Use

- 要确认**某模型在 DCU 上有没有官方适配仓库**（qwen/glm/deepseek/kimi/minimax/ernie/ocr/多模态/训练版 _pytorch/推理版 _vllm…）。
- 要查某模型适配用的 **DTK 版本 / 推理框架版本 / 推荐容器镜像 / 支持卡型（BW1000/K100AI/BW1100）与最低卡数**。
- **旧项目上新设备前查适配基线**：README 环境依赖表里的 DTK 版本就是该项目验证时的基线——DTK 24/25 系的老项目在新设备（26.04 生态）上需要重新适配验证，不能直接跑。
- 要批量筛选"某卡型 + 某框架"的全部适配模型（model.properties 机器可读元数据可直接过滤）。

> 与相邻 skill 的关系：`dcu-image-hub` 管**镜像仓库**（docker 镜像 tag/拉取）；`dcu-das-hub-api` 管 **pip 包源**（wheel）；本 skill 管**模型适配仓库**（模型代码+README+元数据，GitLab group `modelzoo`）。三者互补：选模型 → 本 skill 查适配基线 → dcu-image-hub 找对应镜像 → das-hub-api 补包。

> **官方部署文档库（GitHub，非 ModelZoo）**：HYGON-AI/inference-cookbook-das，docs/model-deployment/sglang 与 vllm 下的 <model>.md 是海光官方按模型的**部署配方文档**——含每模型 env+CLI+参考镜像+启动命令，与 ModelZoo 的 model.properties 元数据**同源互补**。要问"这个模型在 DCU 上官方怎么跑/怎么调"（尤其 GLM-5/5.1/5.2/5.3、DeepSeek-V4、Kimi、MiniMax、Qwen3.x 一代），**优先查这里**；ModelZoo 偏"有没有适配仓库 + 适配基线元数据（DTK/镜像/卡型）"。二者互相印证。

## 1. 快速查（本地快照，零网络）

`data/modelzoo_index.json`（快照日期见 `fetched_at`，2026-09-02 首抓 622 → 2026-10-14 复测 626 仓库）每条含：
`path/name/path_with_namespace/description/default_branch/web_url/last_activity/star_count/props{modelCode,processType,appCategory,frameType,accelerateType}/readme_hints{dtks,images,cards}[]`。**快照 diff/合并以 `path_with_namespace` 全串为键**（脚本 2026-10-14 起输出该字段；子组仓库短名会撞车，group 改名也不受影响）。

```bash
# 按名字搜模型（含 props 里的框架/卡型/DTK）
python3 - <<'EOF'
import json, re
d = json.load(open('data/modelzoo_index.json'))
kw = 'qwen3'
for p in d['projects']:
    blob = json.dumps(p, ensure_ascii=False).lower()
    if kw in blob:
        pr = p.get('props') or {}
        h = p.get('readme_hints') or {}
        print(f"{p['path']:40s} frame={pr.get('frameType','-'):20s} card={pr.get('accelerateType','-'):16s} DTK={','.join(h.get('dtks',[])) or '-':10s} {p['last_activity']}")
EOF

# 找"新设备不可直接用"的老 DTK 项目（DTK < 26.04）
python3 - <<'EOF'
import json
d = json.load(open('data/modelzoo_index.json'))
for p in d['projects']:
    dtks = (p.get('readme_hints') or {}).get('dtks') or []
    if dtks and all(v.split('-')[0] < '26.04' for v in dtks):
        print(p['path'], dtks, p['last_activity'], '→ 需重新适配/验证')
EOF
```

## 2. GitLab API v4（主通道，匿名可访问，无需 token）

基址：`https://developer.sourcefind.cn/codes/api/v4`（页面是 GitLab，ModelZoo 是 group `modelzoo`，visibility=public）。

| 用途 | 请求 | 说明 |
|------|------|------|
| 列全量仓库 | `GET /groups/modelzoo/projects?per_page=100&page=N&order_by=last_activity_at&sort=desc` | 响应头 `X-Total`（626 @2026-10-14）/`X-Total-Pages`（7）控制翻页；返回 GitLab project 全字段 |
| 按名搜 | 同上 + `&search=qwen3` | 匹配 path/name/description |
| 元数据 | `GET /projects/modelzoo%2F<path>/repository/files/model.properties/raw?ref=<default_branch>` | 机器可读：modelCode/modelName/processType(推理\|训练)/appCategory/frameType/accelerateType |
| README 原文 | `GET /projects/modelzoo%2F<path>/repository/files/README.md/raw?ref=<default_branch>` | 标准模板：环境依赖表(DTK/框架版本)、推荐镜像、支持的 DCU 型号+最低卡数、推理启动命令 |
| 文件树 | `GET /projects/modelzoo%2F<path>/repository/tree?ref=<branch>` | 看仓库里有什么文件 |
| 分支/标签 | `.../repository/branches`、`.../repository/tags` | 适配版本 tag（如 v1.0~v1.3） |

⚠ 踩坑：
- `model.properties` **约 5% 仓库没有**（2026-10-14 复测 626 中 32 个 404=594 有 properties，与 09-02 一致，多为早期/特殊仓库如 deepstream-app、comfyui）——这些靠 README 提取。
- `frameType`/`accelerateType` 是人工填的，有脏值（`pytorch，vllm` 中文逗号、`BW1000.K100AI` 点分隔、模板占位文字没删干净），过滤时做归一化。
- `model.properties` 里的 `accelerateType` 与 README「支持的DCU型号」可能不一致（properties 忘更新）——**以 README 为准**，两者都抓。
- 仓库 path 大小写敏感（如 `Qwen2.5-vl_pytorch`），URL 里 `modelzoo%2F` 后跟原样 path。
- ref 用项目 `default_branch`（main/master 都有），猜错会 404。

## 3. 一键刷新快照

```bash
python3 scripts/fetch_modelzoo.py data/modelzoo_index.json --with-readme
# 默认只抓列表+properties（~2min）；--with-readme 加抓 README 提取 DTK/镜像/卡型（8 并发，~10min）
```

输出 JSON：`{fetched_at, source, total, projects:[...]}`（--with-readme 时含 `readme_hints`）。

## 4. README 标准模板（读适配信息的位置）

每个 ModelZoo 仓库 README 按社区模板写，关键字段：

| 字段 | 位置 | 用途 |
|------|------|------|
| 环境依赖表 | `| DTK | 26.04 |` 等 | **适配基线 DTK/框架版本**，上新设备前先对基线 |
| 推荐镜像 | `image.sourcefind.cn:5000/...` 或 `harbor.sourcefind.cn:5443/...` | 直接可 `docker pull` 的完整镜像（镜像 tag 查 dcu-image-hub 核对最新版） |
| 支持的DCU型号+最低卡数 | 权重表 `K100AI,BW1000` + `最低卡数需求` | 选型/排卡；**FP8 权重仅 BW1100/BW1101 支持**，其他卡用 BF16 |
| 推理启动命令 | vLLM/SGLang/Transformers 分节 | 单机/多机启动方式 |

## 5. 旧项目新设备适配警示（实测分布 @2026-09-02）

- 622 仓库中 README 带明确 DTK 基线的约 220 个（2026-09-02 --with-readme 快照口径，10-14 复测沿用该 readme_hints 未重抓）：DTK 26.04 系 105（多为 2026-06 后活跃的新项目）、25.04.x 系 65、24.04.x 15、25.04/23.04 各 1。
- **DTK≤25.04 的老项目上 26.04 新设备需重新适配/验证**（老镜像 tag 还在镜像仓库里可拉，但框架+DTK 组合与新设备驱动生态可能不兼容）；判断规则见 §1 第二个查询。
- 两个镜像域名并存：老 `image.sourcefind.cn:5000`（README 多处引用，Registry v2 匿名探测 401，可能需登录或在 DCU 内网环境拉）、新 `harbor.sourcefind.cn:5443`（匿名 token 可查可拉，见 dcu-image-hub §4）。README 里给的老域名镜像拉不动时，去 dcu-image-hub 按框架名找 harbor 新地址的等价 tag。

## 6. 克隆与使用

```bash
git clone https://developer.sourcefind.cn/codes/modelzoo/<path>.git   # 公开仓库匿名可 clone（http_url_to_repo 字段）
# SSH: ssh://git@developer.sourcefind.cn:2222/modelzoo/<path>.git
```

## Verification Checklist（本 skill 自检，2026-09-02）

- [x] 列表翻页 X-Total=622、7 页全拉通
- [x] 2026-10-14 复测 X-Total=626（+4：OUI、NeoHorse、Nex-N2.5、MiniCPM5，0 移除）、594/626 有 properties，脚本键改 path_with_namespace 全串
- [x] search=qwen3 返回 20 条过滤结果
- [x] model.properties 抓取 590/622 成功（32 个 404 已核实为仓库无此文件；10-14 复测 594/626）
- [x] README raw 抓取 + DTK/镜像/卡型正则提取（595 仓库带推荐镜像）
- [x] template 仓库 README 确认模板结构（环境依赖表/镜像/卡数表/推理命令）
- [x] image.sourcefind.cn:5000 匿名 /v2/ 401（与 harbor:5443 匿名可拉形成对比）
- [x] 快照脚本全量跑通（622 项目 + --with-readme，约 10 分钟，0 失败；10-14 复测 626 项目默认模式 0 失败，readme_hints 沿用 09-02 抓取结果按 path 键合并）
