#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCU orchestrator —— 渲染 workflow 就绪节点与执行顺序。

仅依赖 Python 3 标准库（无需 PyYAML / 第三方包）。

用法：
    python3 render_workflow.py workflows/infer-optimize.yaml
    python3 render_workflow.py workflows/infer-optimize.yaml --tasks-dir tasks
    python3 render_workflow.py workflows/infer-optimize.yaml --json

功能：
    - 解析 workflow YAML（name / max_parallel / nodes[]：id + task + depends_on + retries + proceed）。
    - 按依赖拓扑（Kahn 算法）算就绪节点与执行批次（受 max_parallel 约束）。
    - 解析 node 引用的 task 文件（tasks/<task>.yaml），校验任务契约必需字段
      （task/type, task/goal, task/evidence, task/acceptance 等）。
    - 输出每批次可执行的节点及依赖/状态；--json 输出机器可读结构。
"""
import json
import os
import sys
from collections import deque, OrderedDict

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ---- 极简 YAML 子集解析（仅覆盖本项目 workflow/task 用到的结构）----
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
    """极简 YAML -> Python 结构。支持 map/list/scalar、缩进、行内注释、引号。
    不支持多行块、锚点、块标量（| >）。对本仓库 workflow/task 足够。"""
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
                    # "- key: value"
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
                    # "- key:" 子块
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


def coerce_scalar(s):
    """把字符串标量转成 bool/int/list，用于 flow 列表 [a, b]。"""
    if s is None:
        return None
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1].strip()
        if inner == '':
            return []
        return [coerce_scalar(x) for x in inner.split(',')]
    if s.lower() in ('true',):
        return True
    if s.lower() in ('false',):
        return False
    if s.lower() in ('null', 'none'):
        return None
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    if ',' in s:
        return [coerce_scalar(x) for x in s.split(',')]
    return s


def get(mapping, *keys, default=None):
    """按 keys 逐级取 dict 值；'a/b' 表示 '/' 分隔的嵌套路径。"""
    cur = mapping
    for k in keys:
        for part in str(k).split('/'):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
    return cur


def _node_deps(n):
    """安全取节点 depends_on：容忍 key 缺失/空值/字符串/flow 列表/块列表，绝不崩溃。"""
    raw = get(n, 'depends_on', default=None)
    if not raw:
        return []
    if isinstance(raw, str):
        raw = coerce_scalar(raw)
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [d for d in raw if d]
    return [raw]


def main():
    argv = sys.argv[1:]
    json_out = '--json' in argv
    argv = [a for a in argv if a != '--json']
    tasks_dir = 'tasks'
    if '--tasks-dir' in argv:
        i = argv.index('--tasks-dir')
        tasks_dir = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if not argv:
        print('用法: python3 render_workflow.py <workflow.yaml> [--tasks-dir <dir>] [--json]')
        return 2
    wf_path = argv[0]

    # tasks 目录默认相对 workflow 文件定位：先找同级 tasks/，再找 skill 根（workflows 的上一级）下的 tasks/
    if '--tasks-dir' not in sys.argv[1:]:
        wf_dir = os.path.dirname(os.path.abspath(wf_path))
        for cand in (os.path.join(wf_dir, 'tasks'),
                     os.path.join(os.path.dirname(wf_dir), 'tasks')):
            if os.path.isdir(cand):
                tasks_dir = cand
                break

    with open(wf_path, 'r', encoding='utf-8') as f:
        wf = parse_yaml(f.read())

    name = get(wf, 'name', default='<unnamed>')
    max_parallel = int(get(wf, 'max_parallel', default=1) or 1)
    nodes = get(wf, 'nodes', default=[]) or []

    node_map = OrderedDict()
    for n in nodes:
        nid = get(n, 'id', default='')
        if not nid:
            continue
        node_map[nid] = n

    # Kahn 拓扑分层
    in_deg = {nid: 0 for nid in node_map}
    adj = {nid: [] for nid in node_map}
    for nid, n in node_map.items():
        deps = _node_deps(n)
        for d in deps:
            if d in node_map:
                adj[d].append(nid)
                in_deg[nid] += 1

    ready = deque([nid for nid, d in in_deg.items() if d == 0])
    batches = []
    while ready:
        batch = []
        for _ in range(len(ready)):
            nid = ready.popleft()
            batch.append(nid)
            for nxt in adj[nid]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    ready.append(nxt)
        while len(batch) > max_parallel:
            batches.append(batch[:max_parallel])
            batch = batch[max_parallel:]
        if batch:
            batches.append(batch)

    reachable = set(x for b in batches for x in b)
    report = {
        'workflow': name,
        'max_parallel': max_parallel,
        'cyclic': len(reachable) < len(node_map),
        'unreachable': [nid for nid in node_map if nid not in reachable],
        'batches': [],
    }
    for batch in batches:
        binfo = []
        for nid in batch:
            n = node_map[nid]
            task_ref = get(n, 'task', default='')
            task_meta = {'task_file': '', 'type': '?', 'goal': '', 'missing_fields': []}
            tpath = '%s/%s.yaml' % (tasks_dir, task_ref)
            task_meta['task_file'] = tpath
            try:
                with open(tpath, 'r', encoding='utf-8') as tf:
                    task_yaml = parse_yaml(tf.read())
                required = ['task/type', 'task/goal', 'task/evidence', 'task/acceptance']
                miss = [r for r in required if get(task_yaml, r, default=None) in (None, '')]
                task_meta['type'] = get(task_yaml, 'task/type', default='?') or '?'
                task_meta['goal'] = get(task_yaml, 'task/goal', default='') or ''
                task_meta['missing_fields'] = miss
            except FileNotFoundError:
                task_meta['missing_fields'] = ['<task file not found>']
            binfo.append({
                'id': nid,
                'task': task_ref,
                'depends_on': _node_deps(n),
                'retries': int(get(n, 'retries', default=0) or 0),
                'proceed': get(n, 'proceed', default='auto') or 'auto',
                'task_meta': task_meta,
            })
        report['batches'].append(binfo)

    # 门禁语义：结构问题（依赖环/缺任务文件/缺契约字段）即校验失败，退出码非零，
    # 使本脚本可直接用作 CI/前置门禁（只警告却返回 0 会让坏工作流静默通过）。
    errors = []
    if report['cyclic']:
        errors.append('依赖环或不可达节点: %s' % ','.join(report['unreachable']))
    for batch in report['batches']:
        for n in batch:
            if n['task_meta']['missing_fields']:
                errors.append('节点 %s (task=%s) 缺失契约字段: %s' % (
                    n['id'], n['task'], ','.join(str(m) for m in n['task_meta']['missing_fields'])))
    report['errors'] = errors

    if json_out:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if errors else 0

    print('工作流: %s | max_parallel=%d' % (name, max_parallel))
    if report['cyclic']:
        print('⚠ 检测到依赖环 / 存在不可达节点: %s' % report['unreachable'])
    print('节点总数: %d' % len(node_map))
    for bi, batch in enumerate(report['batches']):
        print('')
        print('--- 批次 %d (可并行 %d) ---' % (bi + 1, len(batch)))
        for n in batch:
            tm = n['task_meta']
            print('  [%s] %s | task=%s | deps=%s | retries=%s | proceed=%s' % (
                n['id'], tm['type'], n['task'], ','.join(str(d) for d in n['depends_on']) or '-',
                n['retries'], n['proceed']))
            print('      goal: %s' % tm['goal'])
            if tm['missing_fields']:
                print('      ⚠ 缺失契约字段: %s' % tm['missing_fields'])
    if errors:
        print('')
        print('✗ 工作流校验失败（%d 个问题）:' % len(errors))
        for e in errors:
            print('  - ' + e)
    else:
        print('')
        print('✓ 工作流校验通过（无环、任务契约齐全）')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
