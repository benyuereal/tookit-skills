#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCU orchestrator —— 工作流状态文件：断点续跑、产物校验与收尾门禁。

仅依赖 Python 3 标准库（无 PyYAML / 无第三方包）。

背景：dcu-orchestrator 的其余脚本都是无状态的，工作流跑到一半断了只能从头来。
本脚本把「每个节点走到哪一步、产出的是哪个版本的产物、最后一次收尾为什么被拒」
落成一个可读的 JSON 状态文件，给后续节点 / 人类 / CI 提供恢复依据。

用法：
    python3 workflow_state.py init    --run-dir DIR --workflow NAME [--nodes a,b,c] [--force]
    python3 workflow_state.py mark    --run-dir DIR --node ID --status STATUS [--artifact PATH ...] [--note TEXT]
    python3 workflow_state.py inspect --run-dir DIR [--json]
    python3 workflow_state.py verify  --run-dir DIR
    python3 workflow_state.py finish  --run-dir DIR --status success|failed [--reason TEXT] [--force]

公共选项：
    --run-dir DIR   运行目录（必填）。状态文件默认落在 <DIR>/workflow-state.json。
    --state PATH    覆盖状态文件路径（默认 <run-dir>/workflow-state.json）。
    -h, --help      显示帮助。

状态文件 schema（<run-dir>/workflow-state.json）：
    {
      "schema_version": 1,
      "workflow": "op-optimize",
      "run_dir": "/abs/run",
      "created_at": "2026-10-28T03:00:00.000000Z",
      "updated_at": "...",
      "nodes": {
        "<node-id>": {
          "status": "pending|running|done|failed|skipped|stale",
          "attempts": 1,                     # 进入 running 次数；直接标记终态时至少记 1
          "started_at": "..."|null,          # 首次进入 running 的时刻
          "finished_at": "..."|null,         # 进入 done/failed/skipped 的时刻
          "updated_at": "..."|null,
          "note": "..."|null,                # --note 最近一次写入
          "stale_reason": "..."|null,        # verify 判 stale 的原因
          "artifacts": {                     # 绝对路径 -> 产物指纹
            "/abs/path/result.json": {"sha256": "...", "size": 123, "mtime": 1761000000}
          }
        }
      },
      "result": {                            # finish 写入的收尾结论
        "status": "success|failed",
        "forced": false,                     # 是否用 --force 绕过门禁
        "reason": null,                      # --reason 原文（强制收尾必填）
        "bypassed": [],                      # 被 --force 绕过的错误码列表
        "finished_at": "..."
      }
    }
    - nodes 的顺序即工作流声明顺序（inspect 的「卡在哪个节点」按此顺序取第一个非 done）。
    - done 节点必须至少有一个 artifact；每次 mark --status done 以本次 --artifact 覆盖旧记录。

门禁（finish --status success）：
    1. 未完成节点门禁：判定前先就地重算所有 done 节点 artifact 的 sha256，失效即降级 stale；
       随后存在任何非 done 节点（含 stale）→ 拒绝，错误码 UNFINISHED_NODES。
    2. 两级天花板门禁：读 <run-dir>/roofline/result.json（找不到再退到 <run-dir>/../roofline/），要求
       一级 ceiling_bound == true 且 二级 measured_us <= ceiling_us * 1.5，两级同时满足才放行。
       缺文件/不可解析 → MISSING_ROOFLINE_EVIDENCE；任一级不满足 → UNRESOLVED_CEILING。
    3. --force --reason TEXT 可绕过上述门禁，但必须在状态文件 result 里记 forced=true + reason + bypassed。
       只给 --force 不给 --reason 视为用法错误，拒绝执行。

并发安全：
    - 写状态一律同目录 tmpfile + os.replace（同文件系统，原子替换）。
    - 读改写全过程持 fcntl.flock 排他锁（锁文件 <state>.lock）；inspect 取共享锁。
    - 因此同一时刻多个 mark 不会丢更新。

退出码：
    0  成功 / 门禁通过
    1  状态问题（init 冲突、未知节点、artifact 缺失致 failed、verify 发现 stale）
    2  用法错误 / finish 收尾门禁拒绝（MISSING_ROOFLINE_EVIDENCE、UNRESOLVED_CEILING、UNFINISHED_NODES）
"""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

STATE_FILENAME = 'workflow-state.json'
SCHEMA_VERSION = 1
CEILING_FACTOR = 1.5

# 契约（tasks/<node>.yaml）与本脚本的相对位置。找不到就不做字段级校验，
# **但必须在输出里说明「为什么没查」**，不能静默降级成「绿色的 done」。
HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.normpath(os.path.join(HERE, '..', 'tasks'))
WORKFLOWS_DIR = os.path.normpath(os.path.join(HERE, '..', 'workflows'))

MARK_STATUSES = ('pending', 'running', 'done', 'skipped', 'failed')
FINAL_STATUSES = ('success', 'failed')
# finish success 被拒时输出的错误码（均返回退出码 2）。
GATE_CODES = ('MISSING_ROOFLINE_EVIDENCE', 'UNRESOLVED_CEILING', 'UNFINISHED_NODES')

STATUS_MARKS = {
    'done': '✓',
    'failed': '✗',
    'running': '▶',
    'pending': '·',
    'skipped': '⊘',
    'stale': '⚠',
}

USAGE = """用法: workflow_state.py <子命令> [选项]

DCU orchestrator 工作流状态文件：断点续跑、产物校验与收尾门禁。

子命令:
  init     初始化状态文件（已存在时不覆盖，需 --force 才重置）
           --run-dir DIR --workflow NAME [--nodes a,b,c] [--force]

  mark     标记节点状态（done 至少需要一个真实存在的 --artifact）
           --run-dir DIR --node ID --status pending|running|done|skipped|failed
           [--artifact PATH ...] [--note TEXT]

  inspect  查看进度：节点状态、最后更新、卡点、artifact 是否仍有效
           --run-dir DIR [--json]

  resume   算出断点续跑计划：哪些不要重跑、哪些要重做、下一个是谁（只读不改）
           --run-dir DIR [--json]
           已完成节点的产物会重新算 sha256；running 视为中断；failed 按
           workflows/<wf>.yaml 的 retries 判「还能重试」还是「需人工」

  verify   复核所有 done 节点的 artifact（sha256），失效节点降级为 stale
           --run-dir DIR

  finish   收尾。success 受两级天花板门禁 + 未完成节点门禁约束
           --run-dir DIR --status success|failed [--reason TEXT] [--force]

公共选项:
  --state PATH   状态文件路径（默认 <run-dir>/workflow-state.json）
  -h, --help     显示本帮助

退出码: 0 成功 / 1 状态问题 / 2 用法错误或 finish 收尾门禁拒绝
"""


# ---------------------------------------------------------------- 基础设施
class UsageError(Exception):
    """命令行用法错误（退出码 2）。"""


class StateMissing(Exception):
    """状态文件缺失或不可解析（退出码 1）。"""


def utc_now():
    """当前 UTC 时间，ISO8601 带 Z 后缀（含微秒，便于排序）。"""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def sha256_file(path):
    """分块计算文件 sha256。"""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact_meta(path):
    """产物指纹：sha256 + size + mtime（mtime 仅作参考，失效判定只看 sha256/存在性）。"""
    st = os.stat(path)
    return {
        'sha256': sha256_file(path),
        'size': int(st.st_size),
        'mtime': int(st.st_mtime),
    }


@contextmanager
def state_lock(state_path, exclusive=True):
    """对 <state>.lock 加 flock；排他锁覆盖整个读改写过程，避免丢更新。"""
    directory = os.path.dirname(os.path.abspath(state_path)) or '.'
    if not os.path.isdir(directory):
        raise StateMissing('状态文件目录不存在: %s（请先执行 init）' % directory)
    lock_path = state_path + '.lock'
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def atomic_write_json(path, data):
    """同目录 tmpfile + fsync + os.replace，保证读者永远看到完整 JSON。"""
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    fd, tmp = tempfile.mkstemp(prefix='.' + os.path.basename(path) + '.', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_state(path):
    """读取并做最小结构校验；缺失/损坏一律抛 StateMissing。"""
    if not os.path.isfile(path):
        raise StateMissing('状态文件不存在: %s（请先执行 init）' % path)
    try:
        with open(path, 'r', encoding='utf-8') as stream:
            data = json.load(stream)
    except Exception as e:
        raise StateMissing('状态文件不是合法 JSON: %s (%s)' % (path, e))
    if not isinstance(data, dict) or not isinstance(data.get('nodes'), dict):
        raise StateMissing('状态文件结构异常（缺少 nodes 对象）: %s' % path)
    return data


def new_node():
    """新节点的初始结构。"""
    return {
        'status': 'pending',
        'attempts': 0,
        'started_at': None,
        'finished_at': None,
        'updated_at': None,
        'note': None,
        'stale_reason': None,
        'artifacts': {},
    }


def state_path_of(opts):
    """解析 --run-dir / --state，返回 (run_dir, state_path) 绝对路径。"""
    run_dir = opts.get('run-dir')
    if not run_dir:
        raise UsageError('缺少 --run-dir')
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    state = opts.get('state')
    if state:
        return run_dir, os.path.abspath(os.path.expanduser(state))
    return run_dir, os.path.join(run_dir, STATE_FILENAME)


def parse_argv(argv):
    """手动解析 --key value / --flag / --artifact v1 v2 ...，风格对齐同目录脚本。"""
    bool_flags = ('force', 'json', 'help', 'skip-evidence-check')
    value_flags = ('run-dir', 'workflow', 'nodes', 'node', 'status', 'note', 'reason', 'state', 'task')
    opts = {'artifact': []}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ('-h', '--help', 'help'):
            opts['help'] = True
            i += 1
            continue
        if not arg.startswith('--'):
            raise UsageError('无法识别的参数: %s' % arg)
        key = arg[2:]
        if key in bool_flags:
            opts[key] = True
            i += 1
            continue
        if key == 'artifact':
            i += 1
            start = i
            while i < len(argv) and not argv[i].startswith('--'):
                opts['artifact'].append(argv[i])
                i += 1
            if i == start:
                raise UsageError('--artifact 需要一个路径')
            continue
        if key not in value_flags:
            raise UsageError('未知选项: %s' % arg)
        if i + 1 >= len(argv):
            raise UsageError('选项 --%s 缺少取值' % key)
        opts[key] = argv[i + 1]
        i += 2
    return opts


def artifact_issues(node):
    """重算节点 artifact：存在性 + sha256，返回问题列表。"""
    issues = []
    artifacts = node.get('artifacts') or {}
    if not artifacts:
        issues.append({'path': None, 'reason': '缺少 artifact 记录'})
        return issues
    for path, meta in artifacts.items():
        if not os.path.isfile(path):
            issues.append({'path': path, 'reason': '文件不存在'})
            continue
        try:
            digest = sha256_file(path)
        except OSError as e:
            issues.append({'path': path, 'reason': '无法读取: %s' % e})
            continue
        if digest != meta.get('sha256'):
            issues.append({'path': path, 'reason': 'sha256 已变化（产物被改写，旧保存点失效）'})
    return issues


# ---------------------------------------------------------------- 子命令
def cmd_init(opts):
    """初始化状态文件；默认不覆盖既有进度。"""
    run_dir, state_path = state_path_of(opts)
    workflow = opts.get('workflow')
    if not workflow:
        raise UsageError('init 需要 --workflow NAME')
    node_ids = []
    for chunk in (opts.get('nodes') or '').split(','):
        node_id = chunk.strip()
        if node_id and node_id not in node_ids:
            node_ids.append(node_id)

    os.makedirs(run_dir, exist_ok=True)
    state_dir = os.path.dirname(state_path) or '.'
    os.makedirs(state_dir, exist_ok=True)

    with state_lock(state_path):
        existed = os.path.exists(state_path)
        if existed and not opts.get('force'):
            print('✗ 状态文件已存在: %s' % state_path)
            print('  为保护已有进度不覆盖；如需重置（会清空所有节点状态）请加 --force。')
            return 1
        now = utc_now()
        state = {
            'schema_version': SCHEMA_VERSION,
            'workflow': workflow,
            'run_dir': run_dir,
            'created_at': now,
            'updated_at': now,
            'nodes': {node_id: new_node() for node_id in node_ids},
            'result': None,
        }
        atomic_write_json(state_path, state)

    if existed:
        print('⚠ --force：已重置状态文件，原有进度被清空: %s' % state_path)
    print('✓ 初始化工作流状态: %s' % workflow)
    print('  状态文件: %s' % state_path)
    print('  节点(%d): %s' % (len(node_ids), ', '.join(node_ids) if node_ids else '(空，可用 --nodes 补充)'))
    return 0


def node_task_path(node_id):
    """节点 → 契约文件。节点 id 与 tasks/<name>.yaml 同名是本项目的约定
    （`workflows/op-optimize.yaml` 里每个 `id` 都对应一个同名 task）。"""
    p = os.path.join(TASKS_DIR, '%s.yaml' % node_id)
    return p if os.path.isfile(p) else None


def verify_artifacts_content(opts, node_id, paths):
    """标记 done 前校验产物**内容**（D4 / T10 P1-5）。

    只验「文件存在」不够：一个 `{}` 也能标绿，evidence 问题全压到收尾才爆。

    契约从哪来，按优先级：
      1. `--task <task.yaml>` 显式指定
      2. 没给就从 `tasks/<node_id>.yaml` 自动取（节点 id == 契约名，这是本项目的约定）
      3. 都没有：只做通用检查（JSON 产物不得为空对象），并**在返回值里说明没做字段级校验**

    校验本身交给 verify_evidence.py —— 字段清单的权威点在任务契约的
    `evidence.fingerprint`，不在这里再写一份（一个概念一个权威点）。
    返回人类可读的问题串；空串表示通过。
    """
    if opts.get('skip-evidence-check'):
        return ''

    task = opts.get('task')
    if not task:
        task = node_task_path(node_id)
        if task:
            opts = dict(opts, task=task)
    problems = []

    # 通用检查：JSON 可解析但内容为空 = 没有 evidence
    # （纯文本产物如日志在无契约时不判，避免把「格式」当成「内容」）
    if not task:
        for path in paths:
            try:
                with open(path, encoding='utf-8') as f:
                    text = f.read()
            except OSError as e:
                problems.append('%s: 无法读取（%s）' % (os.path.basename(path), e))
                continue
            try:
                data = json.loads(text)
            except ValueError:
                continue                     # 非 JSON，无契约时不做内容判定
            if not isinstance(data, dict) or not data:
                problems.append('%s: JSON 为空对象，没有 evidence 内容'
                                % os.path.basename(path))
        if not problems:
            problems.append('节点 %s 没有任务契约（tasks/%s.yaml 不存在），'
                            '只做了「非空 JSON」检查，字段级契约未生效'
                            % (node_id, node_id))
        return '; '.join(problems)

    # 有任务契约就交给 verify_evidence.py
    # 但**只对契约声明的那个 evidence 文件**做指纹校验：契约的 evidence.path 只命名
    # 一个产物，其余 artifact（工具原始输出、日志）本来就没有 fingerprint，拿契约去套
    # 它们会把合法产物误判掉。
    verifier = os.path.join(HERE, 'verify_evidence.py')
    if not os.path.isfile(verifier):
        return '找不到 verify_evidence.py，无法按契约校验（用 --skip-evidence-check 显式跳过）'
    want = contract_evidence_basename(task)
    for path in paths:
        if want and os.path.basename(path) != want:
            continue
        cmd = [sys.executable, '-I', verifier, path, '--task', task]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as e:
            return '无法运行 verify_evidence: %s' % e
        if proc.returncode != 0:
            tail = [ln for ln in (proc.stdout or '').splitlines() if ln.strip()][-4:]
            problems.append('%s: verify_evidence 不通过 — %s'
                            % (os.path.basename(path), ' | '.join(tail)))
    return '; '.join(problems)


def contract_evidence_basename(task_path):
    """从任务契约里抠出 `evidence: path:` 的文件名；抠不到返回 ''（= 全部都要校验）。"""
    try:
        with open(task_path, encoding='utf-8') as f:
            lines = f.read().splitlines()
    except OSError:
        return ''
    in_evidence = False
    for raw in lines:
        line = raw.split('#', 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped.startswith('evidence:'):
            in_evidence = True
            continue
        if in_evidence:
            # evidence 段里第一个 path: 就是它；缩进退回到 task 级说明段结束
            if indent <= 2 and not stripped.startswith('path:'):
                break
            if stripped.startswith('path:'):
                value = stripped.split(':', 1)[1].strip().strip('"').strip("'")
                return os.path.basename(value)
    return ''


def cmd_mark(opts):
    """标记节点状态；done 必须有真实存在的产物，否则节点置 failed。"""
    run_dir, state_path = state_path_of(opts)
    node_id = opts.get('node')
    status = opts.get('status')
    if not node_id:
        raise UsageError('mark 需要 --node ID')
    if status not in MARK_STATUSES:
        raise UsageError('无效 --status: %r（可选: %s）' % (status, '|'.join(MARK_STATUSES)))

    now = utc_now()
    with state_lock(state_path):
        state = load_state(state_path)
        nodes = state.setdefault('nodes', {})
        if node_id not in nodes:
            print('✗ 未知节点: %s（当前节点: %s）' % (node_id, ', '.join(nodes) or '(空)'))
            return 1
        node = nodes[node_id]
        node.setdefault('artifacts', {})

        given = opts.get('artifact') or []
        recorded = {}
        missing = []
        for raw in given:
            path = os.path.abspath(os.path.expanduser(raw))
            if os.path.isfile(path):
                recorded[path] = artifact_meta(path)
            else:
                missing.append(path)

        # done 的硬要求（按「先便宜后昂贵」的顺序拦，别让后面的检查吞掉前面的原因）：
        #   1. 至少要给一个 --artifact
        #   2. 每个 artifact 都必须真实存在（缺失的进 recorded 之前就被剔除，
        #      若放到内容校验**之后**才查，空 recorded 会先撞上「无契约」分支，
        #      把「文件不存在」误报成「字段级契约未生效」）
        #   3. 内容校验（D4）：只验存在会让空 JSON 也能标绿，把所有 evidence 问题
        #      压到收尾一次性爆掉，中途 inspect 看到的全是绿的。
        if status == 'done' and not given:
            node['status'] = 'failed'
            node['attempts'] = max(1, int(node.get('attempts') or 0))
            node['finished_at'] = now
            node['stale_reason'] = None
            node['updated_at'] = now
            reason = '标记 done 被拒：未提供 --artifact（done 至少需要一个产物）'
            node['note'] = ('%s；%s' % (opts['note'], reason)) if opts.get('note') else reason
            state['updated_at'] = now
            atomic_write_json(state_path, state)
            print('✗ 节点 %s 标记 done 被拒：未提供 --artifact。' % node_id)
            print('  已将节点置为 failed。')
            return 1
        if status == 'done' and missing:
            node['status'] = 'failed'
            node['attempts'] = max(1, int(node.get('attempts') or 0))
            node['finished_at'] = now
            node['stale_reason'] = None
            node['updated_at'] = now
            reason = '标记 done 被拒：artifact 不存在或不是文件: %s' % ', '.join(missing)
            node['note'] = ('%s；%s' % (opts['note'], reason)) if opts.get('note') else reason
            state['updated_at'] = now
            atomic_write_json(state_path, state)
            print('✗ 节点 %s 标记 done 被拒：' % node_id)
            for path in missing:
                print('  - 不存在或不是文件: %s' % path)
            print('  已将节点置为 failed。')
            return 1
        if status == 'done' and recorded:
            bad = verify_artifacts_content(opts, node_id, list(recorded))
            if bad:
                node['status'] = 'failed'
                node['attempts'] = max(1, int(node.get('attempts') or 0))
                node['finished_at'] = now
                node['stale_reason'] = None
                node['updated_at'] = now
                reason = '标记 done 被拒：产物未通过 evidence 校验 — %s' % bad
                node['note'] = ('%s；%s' % (opts['note'], reason)) if opts.get('note') else reason
                state['updated_at'] = now
                atomic_write_json(state_path, state)
                print('✗ 节点 %s 标记 done 被拒：产物内容未通过 evidence 校验。' % node_id)
                print('  %s' % bad)
                print('  （--skip-evidence-check 可绕过，但会在状态文件留痕）')
                return 1

        # 正常推进
        if status == 'running':
            node['attempts'] = int(node.get('attempts') or 0) + 1
            if not node.get('started_at'):
                node['started_at'] = now
            node['finished_at'] = None
        elif status in ('done', 'skipped'):
            if int(node.get('attempts') or 0) == 0:
                node['attempts'] = 1
            node['finished_at'] = now
        elif status == 'failed':
            # failed **必须累加**，否则「失败 → 重试 → 再失败」永远停在 attempts=1，
            # workflows/*.yaml 里的 retries 上限就形同虚设（resume 靠它判「还能不能重试」）。
            # done/skipped 不累加：它们不消耗重试预算。
            node['attempts'] = int(node.get('attempts') or 0) + 1
            node['finished_at'] = now
        node['status'] = status
        node['stale_reason'] = None
        if status == 'done':
            node['artifacts'] = recorded
        elif recorded:
            merged = dict(node.get('artifacts') or {})
            merged.update(recorded)
            node['artifacts'] = merged
        if opts.get('note'):
            node['note'] = opts['note']
        if status == 'done' and opts.get('skip-evidence-check'):
            # 提示语承诺「会在状态文件留痕」，必须真留，否则等于无声放行
            stamp = 'evidence 校验被 --skip-evidence-check 显式跳过'
            node['note'] = ('%s；%s' % (node['note'], stamp)) if node.get('note') else stamp
            node['evidence_skipped'] = True
        node['updated_at'] = now
        state['updated_at'] = now
        atomic_write_json(state_path, state)

    print('✓ 节点 %s -> %s (attempts=%d, artifacts=%d)' % (
        node_id, status, node.get('attempts') or 0, len(node.get('artifacts') or {})))
    if missing:
        print('  ⚠ 以下 --artifact 不存在，未记录（非 done 状态不拦截）: %s' % ', '.join(missing))
    return 0


def build_inspect(state, state_path):
    """组装 inspect 的机器可读报告（节点顺序 = 声明顺序）。"""
    nodes = state.get('nodes') or {}
    report_nodes = []
    stuck = None
    done_count = 0
    for node_id, node in nodes.items():
        status = node.get('status') or 'pending'
        if status == 'done':
            done_count += 1
        elif stuck is None:
            stuck = node_id
        issues = artifact_issues(node) if (node.get('artifacts') or status in ('done', 'stale')) else []
        bad = {i['path']: i['reason'] for i in issues if i['path']}
        artifacts = []
        for path, meta in (node.get('artifacts') or {}).items():
            artifacts.append({
                'path': path,
                'sha256': meta.get('sha256'),
                'size': meta.get('size'),
                'mtime': meta.get('mtime'),
                'valid': path not in bad,
                'reason': bad.get(path),
            })
        report_nodes.append({
            'id': node_id,
            'status': status,
            'attempts': int(node.get('attempts') or 0),
            'started_at': node.get('started_at'),
            'finished_at': node.get('finished_at'),
            'updated_at': node.get('updated_at'),
            'note': node.get('note'),
            'stale_reason': node.get('stale_reason'),
            'artifacts': artifacts,
            'artifact_issues': issues,
        })
    return {
        'workflow': state.get('workflow'),
        'state_file': state_path,
        'run_dir': state.get('run_dir'),
        'created_at': state.get('created_at'),
        'updated_at': state.get('updated_at'),
        'nodes_total': len(nodes),
        'nodes_done': done_count,
        'stuck': stuck,
        'result': state.get('result'),
        'nodes': report_nodes,
    }


def print_inspect_human(report):
    """人类可读的进度视图。"""
    print('工作流: %s | 状态文件: %s' % (report['workflow'] or '<unknown>', report['state_file']))
    print('运行目录: %s' % (report['run_dir'] or '-'))
    print('创建: %s | 更新: %s' % (report['created_at'] or '-', report['updated_at'] or '-'))
    if report['stuck']:
        stuck_node = next(n for n in report['nodes'] if n['id'] == report['stuck'])
        print('节点: %d | 已完成: %d | 卡在: %s (%s)' % (
            report['nodes_total'], report['nodes_done'], report['stuck'], stuck_node['status']))
    else:
        print('节点: %d | 已完成: %d | 卡在: -（全部 done）' % (
            report['nodes_total'], report['nodes_done']))
    print('')
    for node in report['nodes']:
        mark = STATUS_MARKS.get(node['status'], '?')
        line = '  %s %-16s %-8s attempts=%d' % (mark, node['id'], node['status'], node['attempts'])
        if node['updated_at']:
            line += ' updated=%s' % node['updated_at']
        artifacts = node['artifacts']
        if artifacts:
            valid = sum(1 for a in artifacts if a['valid'])
            line += ' artifacts=%d(有效 %d)' % (len(artifacts), valid)
        print(line)
        if node['note']:
            print('      note: %s' % node['note'])
        if node['stale_reason']:
            print('      stale: %s' % node['stale_reason'])
        for issue in node['artifact_issues']:
            if issue['path']:
                print('      ! %s: %s' % (issue['path'], issue['reason']))
            else:
                print('      ! %s' % issue['reason'])
    result = report.get('result')
    if result:
        print('')
        tail = ''
        if result.get('forced'):
            tail += ' (forced)'
        if result.get('reason'):
            tail += ' | reason: %s' % result['reason']
        print('结果: %s%s' % (result.get('status'), tail))


def cmd_inspect(opts):
    """查看进度；--json 输出机器可读结构。"""
    _run_dir, state_path = state_path_of(opts)
    with state_lock(state_path, exclusive=False):
        state = load_state(state_path)
        report = build_inspect(state, state_path)
    if opts.get('json'):
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_inspect_human(report)
    return 0


def workflow_node_limits(state):
    """从工作流定义里取每个节点的 `retries`（重试上限的**唯一权威点**）。

    `retries` 写在 `workflows/<wf>.yaml` 的节点上（op-profile 是 2，其余是 1），
    不在这里另设一个全局常量 —— 那就是第二个权威点，早晚会和 YAML 漂开。
    拿不到工作流定义时返回空 dict：续跑计划退化成「不设上限」，并在输出里说明，
    不静默假设一个数字。
    """
    name = state.get('workflow')
    if not name:
        return {}, None
    path = os.path.join(WORKFLOWS_DIR, '%s.yaml' % name)
    if not os.path.isfile(path):
        return {}, '找不到工作流定义 %s，重试上限未知' % path
    try:
        sys.path.insert(0, HERE)
        import render_workflow  # 同目录脚本；解析器只此一份，不另写一个
        with open(path, encoding='utf-8') as stream:
            wf = render_workflow.parse_yaml(stream.read())
    except Exception as e:
        return {}, '解析工作流定义失败（%s: %s），重试上限未知' % (type(e).__name__, e)
    limits = {}
    for node in render_workflow.get(wf, 'nodes', default=[]) or []:
        if not isinstance(node, dict) or not node.get('id'):
            continue
        try:
            limits[str(node['id'])] = int(render_workflow.get(node, 'retries', default=0) or 0)
        except (TypeError, ValueError):
            limits[str(node['id'])] = 0
    return limits, None


def plan_resume(state):
    """断点续跑的核心：算出「新会话该怎么接着走」（T10 / P1-7）。

    续跑的难点不是「读回状态」——状态本来就在 json 里。难点是三件容易搞错的事：

      1. **done 不等于可以跳过**：产物可能被改写/删掉（sha256 变了）。不重算就跳过
         = 拿一份已经不存在的结果往下走。
      2. **running 不等于还在跑**：进程被 kill 之后状态文件永远停在 running，
         没人会把它翻回来。必须当成「中断了、要重做」。
      3. **failed 要区分能不能重试**：同一个节点反复失败还继续跑，就是在烧时间。
         上限读工作流定义里的 `retries`，不在本脚本另设常量。

    本函数只**读**状态、只**算**计划，不改任何文件 —— 让 `resume` 可以在任何
    时刻安全地跑（包括真的还有别的会话在跑的时候）。
    """
    nodes = state.get('nodes') or {}
    limits, limits_error = workflow_node_limits(state)
    todo, blocked, keep = [], [], []

    # 先按「产物是否仍然有效」重新判定每个 done 节点，而不是信 status 字段
    effective = {}
    for node_id, node in nodes.items():
        status = node.get('status') or 'pending'
        if status == 'done':
            issues = artifact_issues(node)
            effective[node_id] = ('stale' if issues else 'done'), issues
        else:
            effective[node_id] = status, []

    for node_id, node in nodes.items():
        status, issues = effective[node_id]
        attempts = int(node.get('attempts') or 0)
        limit = limits.get(node_id)
        entry = {
            'id': node_id,
            'status': status,
            'recorded_status': node.get('status'),
            'attempts': attempts,
            'retries': limit,
            'note': node.get('note'),
            'stale_reason': ('; '.join('%s: %s' % (i['path'] or '<无记录>', i['reason'])
                                       for i in issues) or None),
        }
        if status in ('done', 'skipped'):
            # done 且产物有效 / 显式跳过：都是作者的决定，不重跑
            keep.append(entry)
        elif status == 'running':
            entry['why'] = '状态停在 running（进程被 kill 时不会自动翻回来）→ 重做'
            todo.append(entry)
        elif status == 'stale':
            entry['why'] = '产物已失效（%s）→ 重做' % entry['stale_reason']
            todo.append(entry)
        elif status == 'failed':
            if limit is not None and attempts > limit:
                entry['why'] = ('已失败 %d 次，超过工作流允许的 retries=%d → '
                                '不再自动重试，需人工判断' % (attempts, limit))
                blocked.append(entry)
            else:
                entry['why'] = '上一轮失败（第 %d 次，上限 %s）→ 重试' % (
                    attempts, limit if limit is not None else '未知')
                todo.append(entry)
        else:  # pending 或未知
            entry['why'] = '尚未开始'
            todo.append(entry)

    return {
        'workflow': state.get('workflow'),
        'run_dir': state.get('run_dir'),
        'finished': state.get('result'),
        'limits_error': limits_error,
        'keep': keep,          # 已完成/显式跳过，**不要重跑**
        'todo': todo,          # 要重做的（按声明顺序）
        'blocked': blocked,    # 达重试上限，需人工
        'next': todo[0]['id'] if todo else None,
    }


def print_resume_human(plan):
    print('断点续跑计划：%s' % (plan['workflow'] or '<unknown>'))
    print('运行目录：%s\n' % (plan['run_dir'] or '-'))
    if plan.get('limits_error'):
        print('! %s（按「不设上限」处理）\n' % plan['limits_error'])

    if plan['keep']:
        print('已完成，**不要重跑**（产物 sha256 复核有效）：')
        for e in plan['keep']:
            print('  %s %-18s %s' % (STATUS_MARKS.get(e['status'], '?'), e['id'], e['status']))
        print('')

    if plan['todo']:
        print('要重做：')
        for e in plan['todo']:
            print('  %s %-18s %s' % (STATUS_MARKS.get(e['status'], '?'), e['id'], e['why']))
        print('')

    if plan['blocked']:
        print('达上限，需人工判断（不会自动重跑）：')
        for e in plan['blocked']:
            print('  ✗ %-18s %s' % (e['id'], e['why']))
        print('')

    if plan.get('finished'):
        print('注意：本次 run 已收尾（%s）—— 继续跑会改写已定稿的结论。\n'
              % plan['finished'].get('status'))

    if plan['next']:
        print('下一步：%s' % plan['next'])
    elif plan['blocked']:
        print('下一步：无（%d 个节点卡在上限，需人工处理）' % len(plan['blocked']))
    else:
        print('下一步：无（全部完成）')


def cmd_resume(opts):
    """从状态文件算出续跑计划（只读，不改状态）。"""
    _run_dir, state_path = state_path_of(opts)
    with state_lock(state_path, exclusive=False):
        state = load_state(state_path)
        plan = plan_resume(state)
    if opts.get('json'):
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        print_resume_human(plan)
    # 有可做的工作就返回 0；只剩人工阻塞或已全完成时返回 1，方便脚本判断
    return 0 if plan['next'] else 1


def cmd_verify(opts):
    """复核 done 节点产物；任何失效 -> 降级 stale 并返回非零。"""
    _run_dir, state_path = state_path_of(opts)
    now = utc_now()
    stale_nodes = []
    changed = False
    with state_lock(state_path):
        state = load_state(state_path)
        nodes = state.get('nodes') or {}
        print('===== artifact 校验 =====')
        checked = 0
        for node_id, node in nodes.items():
            status = node.get('status')
            if status not in ('done', 'stale'):
                continue
            checked += 1
            issues = artifact_issues(node)
            if issues:
                detail = '; '.join('%s: %s' % (i['path'] or '<无记录>', i['reason']) for i in issues)
                if status != 'stale' or node.get('stale_reason') != detail:
                    node['status'] = 'stale'
                    node['stale_reason'] = detail
                    node['updated_at'] = now
                    changed = True
                else:
                    node['status'] = 'stale'
                stale_nodes.append(node_id)
                print('  ✗ %s -> stale' % node_id)
                for issue in issues:
                    print('      - %s: %s' % (issue['path'] or '<无记录>', issue['reason']))
            else:
                if status == 'stale':
                    node['status'] = 'done'
                    node['stale_reason'] = None
                    node['updated_at'] = now
                    changed = True
                    print('  ✓ %s -> done（产物已恢复一致，保存点重新有效）' % node_id)
                else:
                    print('  ✓ %s（artifact 有效）' % node_id)
        if changed:
            state['updated_at'] = now
            atomic_write_json(state_path, state)
    print('-------------------------')
    if checked == 0:
        print('无已 done 节点，无需校验。')
        return 0
    if stale_nodes:
        print('✗ 结论: %d 个节点产物失效（已降级 stale）: %s' % (len(stale_nodes), ', '.join(stale_nodes)))
        return 1
    print('✓ 结论: %d 个已 done 节点产物全部有效' % checked)
    return 0


def roofline_field(data, key):
    """从 roofline result.json 取字段：顶层优先，其次 result / roofline 包装层。"""
    if not isinstance(data, dict):
        return None
    for container in (data, data.get('result'), data.get('roofline')):
        if isinstance(container, dict) and key in container:
            return container[key]
    return None


def evaluate_second_level(measured, ceiling):
    """二级天花板：measured_us <= ceiling_us * 1.5。返回 (是否通过, 说明)。"""
    try:
        measured_f = float(measured)
        ceiling_f = float(ceiling)
    except (TypeError, ValueError):
        return False, 'measured_us/ceiling_us 缺失或非数值（measured_us=%r, ceiling_us=%r）' % (measured, ceiling)
    if ceiling_f <= 0:
        return False, 'ceiling_us 非正数（%r），无法判定' % (ceiling,)
    limit = ceiling_f * CEILING_FACTOR
    ok = measured_f <= limit
    return ok, 'measured_us=%g, ceiling_us=%g, 上限=ceiling_us*%.1f=%g → %s' % (
        measured_f, ceiling_f, CEILING_FACTOR, limit, '通过' if ok else '超限')


def cmd_finish(opts):
    """收尾；success 受两级天花板门禁与未完成节点门禁约束。"""
    run_dir, state_path = state_path_of(opts)
    status = opts.get('status')
    reason = opts.get('reason')
    force = bool(opts.get('force'))
    if status not in FINAL_STATUSES:
        raise UsageError('finish 需要 --status success|failed（收到: %r）' % (status,))
    if force and not (reason and reason.strip()):
        print('✗ 使用 --force 必须同时提供 --reason TEXT（强制收尾必须留痕）。')
        return 1

    with state_lock(state_path):
        state = load_state(state_path)
        nodes = state.get('nodes') or {}
        now = utc_now()

        if status == 'failed':
            state['result'] = {
                'status': 'failed',
                'forced': force,
                'reason': reason,
                'bypassed': [],
                'finished_at': now,
            }
            state['updated_at'] = now
            atomic_write_json(state_path, state)
            print('✗ 工作流收尾: failed%s' % ((' | reason: %s' % reason) if reason else ''))
            return 0

        # ---- success 门禁 ----
        # 先就地复核 done 节点的产物：sha256 失效即降级 stale（与 verify 同一判定）。
        # 否则只要没人跑过 verify，被改写的产物仍会凭旧的 done 状态混过门禁。
        downgraded = []
        for node_id, node in nodes.items():
            if node.get('status') != 'done':
                continue
            issues = artifact_issues(node)
            if issues:
                node['status'] = 'stale'
                node['stale_reason'] = '; '.join(
                    '%s: %s' % (i['path'] or '<无记录>', i['reason']) for i in issues)
                node['updated_at'] = now
                downgraded.append(node_id)

        unfinished = [node_id for node_id, node in nodes.items() if node.get('status') != 'done']
        failures = []  # [(错误码, 说明)]
        if downgraded:
            print('⚠ 产物失效已降级 stale（finish 就地复核）: %s' % ', '.join(downgraded))
        if unfinished:
            failures.append(('UNFINISHED_NODES',
                             '仍有未 done 节点（%d/%d）: %s' % (
                                 len(unfinished), len(nodes), ', '.join(unfinished))))

        # roofline 证据两种布局都要认：节点产物落在 run 目录内（<run-dir>/roofline/），
        # 或 run 目录本身是某个节点的子目录（<run-dir>/../roofline/）。
        # 先找就近的那个，找不到再找上一级；都不在才算缺证据。
        candidates = [
            os.path.join(run_dir, 'roofline', 'result.json'),
            os.path.join(run_dir, os.pardir, 'roofline', 'result.json'),
        ]
        roof_path = next((os.path.normpath(p) for p in candidates if os.path.isfile(p)),
                         os.path.normpath(candidates[0]))
        roof = None
        if not os.path.isfile(roof_path):
            failures.append(('MISSING_ROOFLINE_EVIDENCE',
                             '缺少 roofline 证据（试过 %s）' % ' 与 '.join(
                                 os.path.normpath(p) for p in candidates)))
        else:
            try:
                with open(roof_path, 'r', encoding='utf-8') as stream:
                    roof = json.load(stream)
            except Exception as e:
                failures.append(('MISSING_ROOFLINE_EVIDENCE',
                                 'roofline 证据不可解析: %s (%s)' % (roof_path, e)))

        print('===== finish 门禁 =====')
        print('未完成节点: %d/%d%s' % (
            len(unfinished), len(nodes),
            (' (%s)' % ', '.join(unfinished)) if unfinished else ''))
        print('roofline 证据: %s' % roof_path)

        if roof is not None:
            ceiling_bound = roofline_field(roof, 'ceiling_bound')
            measured_us = roofline_field(roof, 'measured_us')
            ceiling_us = roofline_field(roof, 'ceiling_us')
            level1_ok = ceiling_bound is True
            level2_ok, level2_msg = evaluate_second_level(measured_us, ceiling_us)
            print('  一级天花板 (ceiling_bound == true)          : %s（ceiling_bound=%r）' % (
                '通过' if level1_ok else '未通过', ceiling_bound))
            print('  二级天花板 (measured_us <= ceiling_us*1.5)  : %s（%s）' % (
                '通过' if level2_ok else '未通过', level2_msg))
            if not level1_ok:
                failures.append(('UNRESOLVED_CEILING',
                                 '一级天花板未满足：ceiling_bound=%r，须为 true' % (ceiling_bound,)))
            if not level2_ok:
                failures.append(('UNRESOLVED_CEILING', '二级天花板未满足：%s' % level2_msg))
        else:
            print('  一级天花板 (ceiling_bound == true)          : 未评估（缺证据）')
            print('  二级天花板 (measured_us <= ceiling_us*1.5)  : 未评估（缺证据）')

        seen_codes = {code for code, _msg in failures}
        codes = [code for code in GATE_CODES if code in seen_codes]

        if failures and not force:
            print('')
            print('✗ 拒绝收尾为 success：')
            seen = set()
            for code, msg in failures:
                if (code, msg) in seen:
                    continue
                seen.add((code, msg))
                print('  - %s: %s' % (code, msg))
            print('错误码: %s' % ', '.join(codes))
            print('（如确认要强行收尾，用 --force --reason <说明>，会在状态文件里留痕。）')
            if downgraded:
                # 即便拒绝收尾，也要把 stale 降级落盘，避免下次凭旧状态重蹈覆辙
                state['updated_at'] = now
                atomic_write_json(state_path, state)
            # finish success 的所有门禁拒绝统一返回 2（与错误码字符串一起构成机读契约），
            # 便于 CI 区分「收尾被门禁拦下」与「状态文件本身有问题（1）」。
            return 2

        bypassed = codes if force else []
        state['result'] = {
            'status': 'success',
            'forced': force,
            'reason': reason,
            'bypassed': bypassed,
            'finished_at': now,
        }
        state['updated_at'] = now
        atomic_write_json(state_path, state)

    if force and bypassed:
        print('')
        print('⚠ --force：已绕过门禁（%s），reason: %s' % (', '.join(bypassed), reason))
    print('✓ 工作流收尾: success（已写入状态文件）')
    return 0


_COMMANDS = {
    'init': cmd_init,
    'mark': cmd_mark,
    'inspect': cmd_inspect,
    'resume': cmd_resume,
    'verify': cmd_verify,
    'finish': cmd_finish,
}


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(USAGE)
        return 0
    cmd = argv[0]
    if cmd not in _COMMANDS:
        print('✗ 未知子命令: %s' % cmd)
        print(USAGE)
        return 2
    try:
        opts = parse_argv(argv[1:])
    except UsageError as e:
        print('✗ %s' % e)
        print(USAGE)
        return 2
    if opts.get('help'):
        print(USAGE)
        return 0
    try:
        return _COMMANDS[cmd](opts)
    except UsageError as e:
        print('✗ %s' % e)
        return 2
    except StateMissing as e:
        print('✗ %s' % e)
        return 1
    except OSError as e:
        print('✗ 文件操作失败: %s' % e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
