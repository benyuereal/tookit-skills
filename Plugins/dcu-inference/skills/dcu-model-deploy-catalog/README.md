# dcu-model-deploy-catalog — README

## 定位
按模型查 DCU 部署配方的速查目录：96 个已验证模型（LLM/VLM/Embedding/CV/图像视频生成/时序/科学计算）
的镜像、框架、容器/依赖/启动命令、坑。命令摘自各模型《海光DCU模型部署手册》原文。

## 与相邻 skill 分工
- 本 skill：**查现成模型的部署配方**（某模型用哪个镜像/框架/怎么起）。
- `dcu-model-portability`：评估**全新模型**能否即插即用 + 从零产出部署手册（方法学）。
- `dcu-model-deploy-verify`：部署前**真机实测验证**纪律。
- `dcu-vllm-deploy`：vLLM 服务**调优/已知坑**。

## references
- `index.md`：机器可读总索引（模型 → 镜像 tag → 框架 → 原文 内部文档 链接）。
- `llm.md` / `vlm.md` / `embedding.md` / `cv.md` / `image-gen.md` / `video.md` / `audio.md` / `timeseries.md` / `science.md`：
  按类别的完整部署配方（含 docker run / 依赖 / 启动 / 测试命令）。

## 使用
「某模型在 DCU 上怎么部署」→ 先翻 `index.md` 定位镜像/框架，再进对应类别文件取完整命令。
「评估新模型 / 写部署手册」→ 转 `dcu-model-portability`。

## 来源
《<某能源集团客户>2026年网络设备项目测试项汇总》「模型适配支持度」sheet
（内部文档 `<internal-doc-link>`）部署文档交付链接列，96 篇部署手册。
