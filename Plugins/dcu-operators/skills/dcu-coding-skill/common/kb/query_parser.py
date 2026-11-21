"""
查询预处理模块 —— ParsedQuery

从用户 query 中提取：
- API 符号列表（正则匹配 hip*/cuda*/roc* 等）
- 自然语言查询部分（去符号/代码块后剩余文本）
- 代码块（fenced block）
- 错误摘要（自修复场景）

为 search_api / search_fts / search_hybrid 分通道检索提供输入。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# API 符号前缀（匹配HIP/CUDA/ROCm等生态的常见符号）
_API_PREFIXES = [
    r"hip[A-Z][a-zA-Z0-9]*",       # hipMalloc, hipMemcpyAsync
    r"hip[A-Z][a-zA-Z0-9]*_v2",    # hipMalloc_v2 (版本化)
    r"roc[A-Z][a-zA-Z0-9]*",       # rocblas_* , rocrand_*
    r"cuda[A-Z][a-zA-Z0-9]*",      # cudaMalloc, cudaEventCreate
    r"cudnn[A-Z][a-zA-Z0-9]*",     # cudnnConvolutionDescriptor
    r"cublas[A-Z][a-za-zA-Z0-9]*", # cublasCreate
    r"cufft[A-Z][a-zA-Z0-9]*",     # cufftPlan1d
    r"curand[A-Z][a-zA-Z0-9]*",    # curandGenerator
    r"nccl[A-Z][a-zA-Z0-9]*",      # ncclCommInitAll
    r"miopen[A-Z][a-zA-Z0-9]*",    # miopenConvolutionForward
    r"rccl[A-Z][a-zA-Z0-9]*",      # rcclAllReduce
    r"hsa[A-Z][a-zA-Z0-9]*",       # hsaInit
]

# 合并为一个正则（忽略大小写匹配时外部控制）
_API_PATTERN = re.compile(
    r"|".join(_API_PREFIXES),
    re.IGNORECASE
)

# fenced code block 正则（```...``` 或 ~~~...~~~）
_CODE_BLOCK_PATTERN = re.compile(
    r"(```[\s\S]*?```|~~~[\s\S]*?~~~)",
    re.MULTILINE
)

# 错误堆栈/报错信息提取（自修复场景）
_ERROR_PATTERN = re.compile(
    r"(error|failed|exception|invalid|undefined|未定义|错误|失败)",
    re.IGNORECASE
)


@dataclass
class ParsedQuery:
    """
    用户查询的结构化解析结果。

    Attributes:
        raw: 原始用户输入
        symbols: 提取的 API 符号列表（如 ["cudaMalloc", "hipMemcpyAsync"]）
        nl_query: 自然语言查询部分（去掉符号和代码块后的文本）
        code_blocks: 提取的代码块列表
        error_snippet: 自修复场景下提取的关键错误信息（截断后的摘要）
    """
    raw: str
    symbols: List[str] = field(default_factory=list)
    nl_query: str = ""
    code_blocks: List[str] = field(default_factory=list)
    error_snippet: str = ""


def parse_query(query: str, *, extract_errors: bool = False) -> ParsedQuery:
    """
    解析用户 query，提取符号、自然语言、代码块和错误信息。

    Args:
        query: 用户原始输入
        extract_errors: 是否在自修复场景下提取错误摘要

    Returns:
        ParsedQuery 结构体
    """
    raw = query

    # 1. 提取代码块（从原文中移除，避免干扰符号提取）
    code_blocks = _CODE_BLOCK_PATTERN.findall(raw)
    text_without_code = _CODE_BLOCK_PATTERN.sub("", raw)

    # 2. 提取 API 符号
    symbols = _API_PATTERN.findall(text_without_code)
    symbols = _deduplicate_symbols(symbols)

    # 3. 生成自然语言查询（去掉符号后的剩余文本）
    nl_query = _API_PATTERN.sub("", text_without_code)
    nl_query = _clean_whitespace(nl_query)

    # 4. 错误摘要（可选，仅在 extract_errors=True 时提取）
    error_snippet = ""
    if extract_errors:
        error_snippet = _extract_error_snippet(raw)

    return ParsedQuery(
        raw=raw,
        symbols=symbols,
        nl_query=nl_query,
        code_blocks=code_blocks,
        error_snippet=error_snippet,
    )


def _deduplicate_symbols(symbols: List[str]) -> List[str]:
    """去重并保持顺序"""
    seen = set()
    result = []
    for s in symbols:
        normalized = s.lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(s)
    return result


def _clean_whitespace(text: str) -> str:
    """清理多余空白字符，返回单行压缩后的文本"""
    # 将多个空白字符替换为单个空格
    cleaned = re.sub(r"\s+", " ", text)
    # 去除首尾空白
    cleaned = cleaned.strip()
    return cleaned


def _extract_error_snippet(text: str, max_len: int = 200) -> str:
    """
    从错误/堆栈信息中提取关键摘要。

    策略：
    1. 找到包含 error/failed/exception 等关键词的行
    2. 保留该行及其前后各 1 行
    3. 截断到 max_len
    """
    lines = text.split("\n")
    error_lines = []

    for i, line in enumerate(lines):
        if _ERROR_PATTERN.search(line):
            # 取当前行及前后各 1 行（避免截断到不完整的上下文）
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            error_lines.extend(lines[start:end])

    if not error_lines:
        # 如果没有明确错误信息，取前 max_len 字符
        return text[:max_len]

    snippet = "\n".join(error_lines)
    if len(snippet) > max_len:
        snippet = snippet[:max_len] + "..."
    return snippet


# ---- 可选的便捷入口 ----


def extract_symbols(query: str) -> List[str]:
    """仅提取符号的便捷函数"""
    return parse_query(query).symbols


def extract_nl_query(query: str) -> str:
    """仅提取自然语言查询的便捷函数"""
    return parse_query(query).nl_query