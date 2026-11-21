# sourcefind 文档中心（/api-doc）总索引

> 2026-10-14 匿名实测：ConfigTree 7 大类 106 篇口径、按版本树递归展开 **23 树 × 553 叶**。本文件是包级总索引（树→技能映射），各树细节已挂进对应技能；9 棵已挂对应技能（✅ 列），OneScience 与空树留档。

## API 三端点（全匿名）

- host 必须 `developer.sourcefind.cn`（www 前缀 → 404）。
- `GET /api-doc/document/GetPageInfo` — ConfigTree 大类清单。
- `GET /api-doc/document/TreeDocument?VersionUUID=<树uuid>` — 目录树（叶含 uuid/path/type）。
- `GET /api-doc/document/DetailDocument?UUID=<叶uuid>` — 全文 markdown（Type=md）。
- `SearchDocument` — 全文搜索端点存在；参数名未固化，以实时调用为准。

## 23 树清单与技能映射

| 树 UUID | 叶数 | 内容 | 落点 |
|---|---|---|---|
| 0a12cd95-1b62 | 8 | DCU 直通用户手册（按机型） | ✅ dcu-csv-passthrough §2.6.2 |
| 42c18d04-420d | 1 | RCCL-TEST 指导手册 | ✅ dcu-rccl-test References |
| 559f2ba3-eae0 | 4 | K8s 组件发布说明 v2.1.0→v2.4.0 | ✅ dcu-kubernetes §6 |
| 768af98e-c10d | 188 | DCU-DCGM（5 章，§3.2 C API 17 组） | ✅ dcu-kubernetes §6 |
| 7541efc4-54b1 | 98 | dcu-operator（10 章） | ✅ dcu-opensource-platform |
| ad998204-4e7c | 56 | DCC 编译器（8 章） | ✅ dcu-triton-source/references/dcc-compiler-docs.md |
| 5ce53abd-1b65 | 12 | HyQual 手册（BW1000/BW1100/K100-AI） | ✅ dcu-hyqual-acceptance |
| 74da671f-1b65 | 22 | HY-SMI CTL 使用手册 | ✅ dcu-hy-smi References |
| 9169ef18-c10d | 12 | DCU 虚拟化/Multi-Instance/K8s 虚拟化 | ✅ dcu-kubernetes §6 |
| 5fca5126-16a8 | 12 | DCU-Container-Toolkit（安装/运行时集成/CLI） | ✅ dcu-docker References |
| 4fd79e96-ad7c | 60 | DTK 软件栈（C++/Fortran/AI 应用） | ✅ dcu-dtk References |
| 5b77899d-ad7c | 21 | DCU 环境/Toolkit 部署/作业调度/HPC | ✅ dcu-env 文末指针 |
| 678457f6-268f | 5 | DCU 硬件环境部署手册/服务器适配列表 | ✅ dcu-env 文末指针 |
| d9126ccd-226f | 13 | BW/K100-AI/Z100L·K100 机型基础环境部署 | ✅ dcu-env 文末指针 |
| a2e964e8-a67d | 25 | DCU 编程实战 2026（概述/架构/编程方法/移植优化/AI 应用；配套 Cat7 SampleChapter3-6.zip） | ✅ dcu-cuda-hip-port 文末指针 |
| 5f2cd48c-ad7c | 10 | AI 框架手册（MIGraphX/TF/PyTorch 常见网络/调试/扩展/第三方库转码） | ✅ dcu-dtk References（叶名实测改正：非 HPC） |
| 60480536-5283 | 6 | OneScience（数据集说明/热点模型/使用文档/开发案例） | 不沉淀（AI4S 非主线，I2 口径） |
| be04831c-6e1e / 84bd1421-81a6 / 4d1181f9-80c6 / 87ee5c5b-c10d / 5ceaa38a-5a6b | 0 | 空标题树（TreeDocument 有 UUID 无内容） | 留档防重复调研 |

## 已判定不沉淀（防重复调研）

- 数据集 36 个：ListDataset 公开、DetailDataset 需登录（fail.login），不建技能。
- 技能市场 3 包（dcu-coding-skill 1.2.1 / dcu-master-router 1.0.2 / dcu-code-migrator 1.0.2）：版本号随实时页面查，不进库。
- OOB 带外管理文档：体量小，已并入 dcu-hyqual-acceptance 一行 + Cat13 下载位。