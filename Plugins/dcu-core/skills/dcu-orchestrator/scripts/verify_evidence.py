#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCU orchestrator —— 校验某次运行收集的 evidence 是否满足任务契约。

仅依赖 Python 3 标准库（JSON + 极简 YAML）。

用法：
    python3 verify_evidence.py <evidence.json> [--task <task.yaml>]
    python3 verify_evidence.py <evidence.json> --fingerprint device dtk image model ttft tpot throughput

功能：
    - 校验 evidence JSON 可解析，含 evidence.path 与 evidence.fingerprint。
    - 若给定 --task <task.yaml>，读取任务契约的 evidence（task/evidence 的 path + fingerprint），
      校验收集的 evidence 是否覆盖契约要求的指纹字段，且关键字段非空、非占位符（<...>）。
    - 输出校验报告，退出码：0=通过，1=不通过。

契约（fingerprint）公共字段：device / dtk / image / model
serving 追加：ttft / tpot / throughput
kernel 追加：latency / bandwidth
"""
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

PLACEHOLDER_RE = re.compile(r'^<.*>$')


# ---- 极简 YAML 解析（与 render_workflow 同源，避免第三方依赖）----
def _strip_comment(line):
    idx = line.find(' #')
    return line[:idx].rstrip() if idx >= 0 else line.rstrip()

def _split_kv(line):
    if ':' in line:
        k, v = line.split(':', 1)
        return k.strip(), v.strip()
    return None, None

def _unquote(v):
    if v is None:
        return None
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
        return v[1:-1]
    return v

def parse_yaml(text):
    lines = []
    for raw in text.split('\n'):
        line = _strip_comment(raw)
        if line.strip() == '' or line.strip().startswith('#'):
            continue
        lines.append(line)

    def parse_block(idx, indent):
        container = None
        while idx < len(lines):
            line = lines[idx]
            if line.strip() == '':
                idx += 1
                continue
            cur = len(line) - len(line.lstrip())
            if cur < indent:
                break
            if cur > indent:
                raise ValueError('缩进异常: %r' % line)
            stripped = line.strip()
            if stripped.startswith('- '):
                if container is None:
                    container = []
                elif not isinstance(container, list):
                    raise ValueError('混合 map/list: %r' % line)
                item_txt = stripped[2:].strip()
                k, v = _split_kv(item_txt)
                if k is not None and v not in (None, ''):
                    item = {k: _unquote(v)}
                    idx += 1
                    if idx < len(lines):
                        ni = len(lines[idx]) - len(lines[idx].lstrip())
                        if ni == cur + 2 and not lines[idx].strip().startswith('- '):
                            sub, idx = parse_block(idx, ni)
                            if isinstance(sub, dict):
                                item.update(sub)
                    container.append(item)
                elif k is not None:
                    idx += 1
                    sub, idx = parse_block(idx, cur + 2)
                    item = {k: sub}
                    if idx < len(lines):
                        ni = len(lines[idx]) - len(lines[idx].lstrip())
                        if ni == cur + 2 and not lines[idx].strip().startswith('- '):
                            sub2, idx = parse_block(idx, ni)
                            if isinstance(sub2, dict):
                                item.update(sub2)
                    container.append(item)
                else:
                    container.append(_unquote(item_txt))
                    idx += 1
            else:
                k, v = _split_kv(line)
                if container is None:
                    container = {}
                elif not isinstance(container, dict):
                    raise ValueError('混合 list/map: %r' % line)
                if v in (None, ''):
                    idx += 1
                    sub, idx = parse_block(idx, cur + 2)
                    container[k] = sub if sub is not None else {}
                else:
                    container[k] = _unquote(v)
                    idx += 1
        return container, idx

    if not lines:
        return {}
    result, _ = parse_block(0, 0)
    return result if isinstance(result, dict) else result


def get(mapping, *keys, default=None):
    cur = mapping
    for k in keys:
        for part in str(k).split('/'):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
    return cur


# 缺失/占位值集合（大小写不敏感）：'MISSING' 是 dcu_distill.py 对缺失字段的哨兵值，必须按缺失拦截，
# 否则全 MISSING 的指纹会骗过校验（op-distill.yaml 验收：fingerprint 无缺失、非占位符）。
MISSING_VALUES = ('', 'MISSING', 'N/A', 'NA', 'NONE', 'NULL', 'TBD', 'TODO', 'PENDING', 'YOUR_..._HERE')


def is_placeholder(v):
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip()
        return PLACEHOLDER_RE.match(s) is not None or s.upper() in MISSING_VALUES
    return False


def main():
    argv = sys.argv[1:]
    if not argv:
        print('用法: python3 verify_evidence.py <evidence.json> [--task <task.yaml>] '
              '[--fingerprint f1 f2 ...]')
        return 2
    ev_path = argv[0]
    task_file = None
    extra_fp = []
    i = 1
    while i < len(argv):
        if argv[i] == '--task' and i + 1 < len(argv):
            task_file = argv[i + 1]
            i += 2
        elif argv[i] == '--fingerprint':
            extra_fp = argv[i + 1:]
            break
        else:
            i += 1

    ev_path = os.path.expanduser(ev_path)
    if task_file:
        task_file = os.path.expanduser(task_file)
    if not os.path.exists(ev_path):
        print('✗ evidence 文件不存在: %s' % ev_path)
        return 1
    try:
        with open(ev_path, 'r', encoding='utf-8') as f:
            ev = json.load(f)
    except Exception as e:
        print('✗ evidence 不是合法 JSON: %s' % e)
        return 1

    # 取 evidence 对象（顶层直接是 evidence，或包装在 evidence 键）
    ev_obj = ev if ('path' in ev or 'fingerprint' in ev) else (ev.get('evidence') or {})
    epath = ev_obj.get('path')
    fp = ev_obj.get('fingerprint') or {}

    issues = []
    ok = True
    if not epath:
        issues.append('缺少 evidence.path')
        ok = False

    # status↔校验联动：生产方（dcu_distill.py）自报 status=unverified 时，即使字段齐全也不得判为通过
    estatus = str(fp.get('status') or ev_obj.get('status') or '').strip().lower()
    if estatus == 'unverified':
        issues.append('evidence status=unverified（生产方标记指纹不完整），不得判为通过')
        ok = False

    req_fp = []
    if task_file:
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                ty = parse_yaml(f.read())
            contract = get(ty, 'task/evidence/fingerprint', default={}) or {}
            if isinstance(contract, dict):
                req_fp = list(contract.keys())
            elif isinstance(contract, list):
                req_fp = contract
        except FileNotFoundError:
            issues.append('任务文件不存在: %s' % task_file)
            ok = False
        except Exception as e:
            issues.append('解析任务文件失败: %s' % e)
            ok = False
    for f in extra_fp:
        if f not in req_fp:
            req_fp.append(f)

    known_domain = ['device', 'dtk', 'image', 'model', 'ttft', 'tpot',
                    'throughput', 'latency', 'bandwidth']
    check_list = req_fp if req_fp else known_domain

    covered = {}
    for f in check_list:
        present = f in fp and not is_placeholder(fp.get(f))
        covered[f] = present
        if not present:
            issues.append('fingerprint 字段 %s 缺失或为占位符（值: %r）' % (f, fp.get(f)))
            ok = False

    print('===== evidence 校验报告 =====')
    print('evidence 文件 : %s' % ev_path)
    print('evidence.path : %s' % epath)
    print('fingerprint   : %s' % json.dumps(fp, ensure_ascii=False))
    print('需覆盖指纹   : %s' % (', '.join(check_list) if check_list else '(未指定，按领域默认)'))
    print('--- 逐项 ---')
    for f in check_list:
        mark = '✓' if covered.get(f) else '✗'
        print('  %s %s' % (mark, f))
    for iss in issues:
        print('  - %s' % iss)
    print('-----------------------------')
    print('结论: %s' % ('通过' if ok else '不通过'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
