"""
OpenAI 接口修复代理（MTP tool-call 参数漂移）：客户端 -> 本代理(:8080) -> vLLM(:8000, MTP 开)。

解决 vLLM + MTP(num_speculative_tokens>1) 场景下，tool_calls 的 arguments JSON
结尾漂移（结尾 "}" 被 "{}" 顶替 / 缺闭合符 / 整段重复等）导致客户端
json.loads 失败、tool call 丢弃的问题。

修复核心 try_repair_input 来自 benyuereal/llm-optimize（已覆盖结尾 {}、缺闭合符、
对象未闭合嵌 {、整段重复/尾掺垃圾等形态），本文件只重写协议层：
Anthropic 的 content_block_delta.input_json_delta -> OpenAI 的
choices.delta.tool_calls[].function.arguments。

用法：
  pip install fastapi uvicorn httpx
  python openai_repair_proxy.py
然后把客户端的 base_url 指向 http://127.0.0.1:8080 （保持 /v1 路径不变）。

配置在文件顶部 UPSTREAM / LISTEN_PORT / LOG_PATH。
"""
import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
import httpx

UPSTREAM = "http://127.0.0.1:8000"   # vLLM OpenAI 接口
LISTEN_PORT = 8080
LOG_PATH = "/tmp/openai_repair_proxy.log"
MAX_RETRIES = 3

app = FastAPI()


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def sse_line(payload: dict) -> str:
    """OpenAI SSE 事件只有 data: 行，没有 event: 行（和 Anthropic 不同）。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------- 畸形修复（直接复用仓库逻辑） ----------
_REPAIR_TAIL_CANDIDATES = [
    ('{}',    '"}'),
    ("{}'",   '"}'),
    ('{}"',   '"}'),
    ('{',     '"}'),
    ('}',     '"}'),
]


def _find_embedded_brace(raw: str):
    """定位'第一个对象未闭合就嵌入下一个 {'的多余 { 位置。找不到返回 None。"""
    depth = 0
    in_str = False
    esc = False
    started = False
    for i, ch in enumerate(raw):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if not started:
                started = True
                depth = 1
            else:
                if depth == 1:
                    return i
                depth += 1
        elif ch in "[":
            if started:
                depth += 1
        elif ch in "}]":
            if started:
                depth -= 1
    return None


def try_repair_input(raw: str):
    """尝试修复畸形 input JSON。成功返回修复后字符串，失败/已合法返回 None。"""
    if not raw or not raw.strip():
        return None
    try:
        json.loads(raw)
        return None  # 本就合法
    except Exception:
        pass
    for bad_tail, good_tail in _REPAIR_TAIL_CANDIDATES:
        if raw.endswith(bad_tail):
            cand = raw[: -len(bad_tail)] + good_tail
            try:
                json.loads(cand)
                return cand
            except Exception:
                continue
    for suffix in ['"}', '"}', '"]}', '"]', '}']:
        cand = raw + suffix
        try:
            json.loads(cand)
            return cand
        except Exception:
            continue
    stripped = raw.rstrip("\"'{}[]:, \t\r\n")
    for suffix in ['"}', '"}', '}']:
        cand = stripped + suffix
        try:
            json.loads(cand)
            return cand
        except Exception:
            continue
    embed_pos = _find_embedded_brace(raw)
    if embed_pos is not None:
        cand = raw[:embed_pos] + "}"
        try:
            json.loads(cand)
            return cand
        except Exception:
            pass
    s = raw.lstrip()
    if s.startswith("{") or s.startswith("["):
        for i in range(1, len(raw)):
            if raw[i] in "}]":
                cand = raw[: i + 1]
                try:
                    json.loads(cand)
                    if raw[i + 1:].strip():
                        return cand
                except Exception:
                    continue
    return None


# ---------- 非流式修复 ----------
def non_stream_repair(data: dict) -> tuple[dict, bool]:
    """就地修复非流式响应里的 tool_calls.arguments。返回 (data, ok)。"""
    ok = True
    for choice in data.get("choices", []) or []:
        msg = choice.get("message", {}) or {}
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            args = fn.get("arguments")
            if isinstance(args, str) and args.strip():
                try:
                    json.loads(args)
                except Exception:
                    fixed = try_repair_input(args)
                    if fixed is not None:
                        fn["arguments"] = fixed
                        log(f"REPAIR(non-stream) name={fn.get('name')} OK")
                        log(f"  before = {args!r}")
                        log(f"  after  = {fixed!r}")
                    else:
                        ok = False
                        log(f"REPAIR(non-stream) name={fn.get('name')} FAILED (原样)")
                        log(f"  args   = {args!r}")
    return data, ok


# ---------- 混合流式（文字实时转，tool args 缓冲修复） ----------
async def hybrid_stream(body: bytes, headers: dict):
    """OpenAI 流式：arguments 按 tool_calls[].index 分片累积，流结束修复后补发。"""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{UPSTREAM}/v1/chat/completions",
                                 content=body, headers=headers) as resp:
            args_parts: dict[int, list[str]] = {}   # index -> [partial arguments]
            args_name: dict[int, str] = {}           # index -> function name

            async for line in resp.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data: "):
                    yield line + "\n"
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    yield "data: [DONE]\n\n"
                    continue
                try:
                    ev = json.loads(payload)
                except Exception:
                    yield line + "\n"
                    continue

                choices = ev.get("choices", []) or []
                modified = False
                for ch in choices:
                    delta = ch.get("delta", {}) or {}
                    for tc in delta.get("tool_calls", []) or []:
                        idx = tc.get("index")
                        fn = tc.get("function", {}) or {}
                        if idx is None:
                            continue
                        if fn.get("name") is not None and idx not in args_parts:
                            args_parts[idx] = []
                            args_name[idx] = fn.get("name", "?")
                        if fn.get("arguments") is not None:
                            if idx not in args_parts:
                                args_parts[idx] = []
                                args_name[idx] = fn.get("name", "?")
                            args_parts[idx].append(fn["arguments"])
                            fn["arguments"] = ""
                            modified = True

                if modified:
                    keep = False
                    for ch in choices:
                        d = ch.get("delta", {}) or {}
                        if d.get("role") or d.get("content") or d.get("name") is not None:
                            keep = True
                        for tc in d.get("tool_calls", []) or []:
                            tcf = tc.get("function", {}) or {}
                            if tcf.get("name") is not None:
                                keep = True
                    if keep:
                        yield sse_line(ev)
                    continue

                yield sse_line(ev)

            # 流结束：把每个缓冲的 tool args 修复后，补发携带完整 arguments 的 delta
            for idx, parts in args_parts.items():
                raw_args = "".join(parts)
                name = args_name.get(idx, "?")
                final_args = raw_args
                if raw_args.strip():
                    try:
                        json.loads(raw_args)
                    except Exception:
                        fixed = try_repair_input(raw_args)
                        if fixed is not None:
                            final_args = fixed
                            log(f"REPAIR(stream) name={name} OK")
                            log(f"  before = {raw_args!r}")
                            log(f"  after  = {fixed!r}")
                        else:
                            log(f"REPAIR(stream) name={name} FAILED (原样发出)")
                            log(f"  args   = {raw_args!r}")
                patch = {
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": idx,
                                "function": {"arguments": final_args},
                            }],
                        },
                        "finish_reason": None,
                    }]
                }
                yield sse_line(patch)


# ---------- 路由 ----------
@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    body = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    try:
        is_stream = json.loads(body).get("stream", False)
    except Exception:
        is_stream = False

    if is_stream:
        return StreamingResponse(
            hybrid_stream(body, headers),
            media_type="text/event-stream",
        )

    async with httpx.AsyncClient(timeout=None) as client:
        last_data = None
        last_status = 500
        last_headers = {}
        for attempt in range(1, MAX_RETRIES + 1):
            resp = await client.post(f"{UPSTREAM}/v1/chat/completions",
                                     content=body, headers=headers)
            try:
                data = resp.json()
            except Exception:
                data = {}
            fixed_data, ok = non_stream_repair(data)
            if ok:
                log(f"attempt {attempt}: OK (non-stream)")
                return JSONResponse(content=fixed_data, status_code=resp.status_code,
                                     headers=dict(resp.headers))
            last_data, last_status, last_headers = fixed_data, resp.status_code, dict(resp.headers)
            log(f"attempt {attempt}: BAD (non-stream, unrepairable), retry")
        log("retries exhausted, returning last response")
        return JSONResponse(content=last_data, status_code=last_status, headers=last_headers)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(path: str, request: Request):
    body = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.request(request.method, f"{UPSTREAM}/{path}",
                                     content=body, headers=headers)
    return Response(content=resp.content, status_code=resp.status_code,
                    headers=dict(resp.headers))


if __name__ == "__main__":
    import uvicorn
    open(LOG_PATH, "a", encoding="utf-8").close()
    log(f"OpenAI 修复代理启动，监听 :{LISTEN_PORT}，上游 {UPSTREAM}，日志 {LOG_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="warning")
