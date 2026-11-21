#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def pick(pattern: str, text: str) -> str:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_candidates(raw_text: str):
    # 折叠空白，处理“整行粘连/换行丢失”
    flat = normalize_spaces(raw_text)
    # 从 hipblaslt-bench 开始，截到 --activation_type XXX 结束
    pattern = re.compile(r"(hipblaslt-bench\b.*?--activation_type\s+\S+)", re.IGNORECASE)
    return [normalize_spaces(x) for x in pattern.findall(flat)]


def build_key(cmd: str):
    # 用关键字段做去重键，避免空格差异导致重复
    return (
        pick(r"--api_method\s+(\S+)", cmd),
        pick(r"(?:^|\s)-m\s+(\d+)", cmd),
        pick(r"(?:^|\s)-n\s+(\d+)", cmd),
        pick(r"(?:^|\s)-k\s+(\d+)", cmd),
        pick(r"--lda\s+(\d+)", cmd),
        pick(r"--ldb\s+(\d+)", cmd),
        pick(r"--ldc\s+(\d+)", cmd),
        pick(r"--ldd\s+(\d+)", cmd),
        pick(r"--stride_a\s+(\d+)", cmd),
        pick(r"--stride_b\s+(\d+)", cmd),
        pick(r"--stride_c\s+(\d+)", cmd),
        pick(r"--stride_d\s+(\d+)", cmd),
        pick(r"--alpha\s+(\S+)", cmd),
        pick(r"--beta\s+(\S+)", cmd),
        pick(r"--transA\s+([TN])", cmd).upper(),
        pick(r"--transB\s+([TN])", cmd).upper(),
        pick(r"--batch_count\s+(\d+)", cmd),
        pick(r"--scaleA\s+(\S+)", cmd),
        pick(r"--scaleB\s+(\S+)", cmd),
        pick(r"--a_type\s+(\S+)", cmd),
        pick(r"--b_type\s+(\S+)", cmd),
        pick(r"--c_type\s+(\S+)", cmd),
        pick(r"--d_type\s+(\S+)", cmd),
        pick(r"--scale_type\s+(\S+)", cmd),
        pick(r"--bias_type\s+(\S+)", cmd),
        pick(r"--compute_type\s+(\S+)", cmd),
        pick(r"--activation_type\s+(\S+)", cmd),
    )


def is_valid_key(key):
    # 关键字段至少要有这些，否则视为截断命令
    required_idx = [0, 1, 2, 3, 14, 15, 19, 20, 23, 25, 26]
    return all(key[i] for i in required_idx)


def main():
    parser = argparse.ArgumentParser(
        description="从原始混合日志提取并去重 hipblaslt-bench 完整命令。"
    )
    parser.add_argument("-i", "--input", required=True, help="原始日志文件路径")
    parser.add_argument("-o", "--output", default="sample_dedup.sh", help="输出去重后的命令文件")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_text = input_path.read_text(encoding="utf-8", errors="ignore")
    candidates = extract_candidates(raw_text)

    seen = set()
    uniq = []
    invalid = 0

    for cmd in candidates:
        key = build_key(cmd)
        if not is_valid_key(key):
            invalid += 1
            continue
        if key in seen:
            continue
        seen.add(key)
        uniq.append(cmd)

    output_path.write_text("\n".join(uniq) + ("\n" if uniq else ""), encoding="utf-8")

    print(f"[OK] candidates: {len(candidates)}")
    print(f"[OK] invalid/skipped: {invalid}")
    print(f"[OK] unique: {len(uniq)}")
    print(f"[OK] output: {output_path}")


if __name__ == "__main__":
    main()
