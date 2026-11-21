# vLLM + MTP 工具调用参数结尾漂移（tool-call argument drift）

## 现象
vLLM 部署开 MTP 投机解码（num_speculative_tokens: 2 等）后，Claude Code / OpenAI SDK
客户端的 tool call 偶发失败：约 10% 概率报 Invalid tool parameters / json.loads
失败（Unterminated string）。关闭 MTP 即恢复正常，但 MTP 的推理加速不可放弃。

注意：漂移对较长的参数串更敏感（结尾 token 越靠后，MTP 猜错概率越高）。

## 根因
MTP 一次预测多个 token，在生成 tool call 参数 JSON 的结尾处偶发多吐一对 {}，
把本该收尾的 "} 顶掉，导致 JSON 字符串未闭合。

  正常: {"command":"ls","description":"测试 10"}   <- 结尾 " } 合法闭合
  畸形: {"command":"ls","description":"测试 10{}   <- 多了 {}，引号丢失，字符串未终止

- 错误发生在客户端的本地 schema 校验环节，不是网络或服务端 bug。
- 转发层（LiteLLM / nginx）日志全是 200 OK —— 它们只转发不校验参数合法性。
- vLLM 也认为生成成功 —— 它不知道结尾多吐了 {}。
- 漂移形态高度规律：永远是结尾 "} 被 {} 取代（或衍生形态，见下）。
  这让它可以被确定性修复（把结尾 {} 改回 "}）。

来源案例：benyuereal/llm-optimize（Anthropic 链路，Claude Code -> LiteLLM -> vLLM）。
本笔记把它推广到任意「客户端 -> vLLM(OpenAI 接口, MTP 开)」链路。

## 畸形形态（按出现频率）
1. 结尾 {} 取代 "}（最常见）：{"cmd":"ls","desc":"x{} -> 修复结尾为 "}
2. 缺闭合符：{"cmd":"ls" -> 补 "}
3. 对象未闭合就嵌入下一个 {：{"a":1{"a":1} -> 删掉多余 { 及之后，补 } 闭合
4. 整段重复 / 尾掺垃圾：{"a":1}{"a":1} 或 {"a":1}GARBAGE -> 取第一个完整合法 JSON 前缀

所有分支均以 json.loads 成功为最终判据，避免误伤合法 JSON。

## 修复方案：旁路转发修复代理（推荐，零改动 vLLM / 客户端）
在「客户端 <-> vLLM」之间插一层轻量转发代理（FastAPI + httpx），拦截 SSE 流，
把畸形 tool call 的 arguments 原地修复成合法 JSON 再发回客户端。MTP 照开、客户端零改动。

- 修复核心函数 try_repair_input（纯字符串 + 括号状态机 + json.loads 验证）直接复用
  benyuereal/llm-optimize 的成熟实现，已覆盖上述 4 种形态。
- 协议层按 OpenAI 流式格式重写：choices.delta.tool_calls[].function.arguments 按
  index 分片累积，缓冲到流结束修复后补发（保留打字机效果，额外延迟极小）。
- 非流式：收完整体修复，带重试兜底。

可直接用的 OpenAI 版代理模板见 templates/openai_repair_proxy.py（已端到端验证：
非流式 / 流式畸形 {} 漂移均被修复为合法 JSON，客户端最终 json.loads 成功）。

### 已知限制（旁路代理）
- 代理补发的「完整 arguments delta」用 finish_reason: null、choice index 0 的 delta
  事件。绝大多数 OpenAI SDK（openai-python / langchain）按 tool_calls[].index 累加
  arguments，能正确合并。少数对「流结束后再补发 arguments」有特殊校验的严格客户端
  可能报错 —— 此时改用缓冲版（收完再发，无打字机效果）。
- 旁路打补丁，治标不治本。根治应在 vLLM 侧：MTP 解码在结构结尾处不强行多吐 token，
  或在 tool call 序列化出口加 JSON 合法性兜底。

## 验证脚本配方（不依赖真实 vLLM）
用 FastAPI TestClient + httpx.ASGITransport 把代理 httpx 重定向到本地 fake vLLM
（返回畸形分片），即可在单测里真实执行流式/非流式修复链路，断言客户端拿到合法 JSON。
做法：monkey-patch httpx.AsyncClient 为 ASGITransport(app=fake)，再
TestClient(proxy_app).post("/v1/chat/completions", json={...stream:True/False})。

## 触发此问题的典型部署
- 自建 GPU/DCU 推理：vLLM 跑 glm-5.2 等模型，开 MTP 加速，客户端走 tool use。
- 客户端：Claude Code（Anthropic 协议经 LiteLLM 转换）、或任意 OpenAI 兼容 SDK。
- DCU 实测环境参考：vLLM 0.18.1（HCU 定制版）、glm-5.2、deepseek_mtp、
  num_speculative_tokens: 2。
