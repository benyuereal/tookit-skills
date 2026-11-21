#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dcu-orchestrator workflow_state.py 的 CLI 契约测试。

只通过 subprocess 调脚本（不 import 内部函数），保证命令行契约本身被测到。
仅依赖标准库 unittest + tempfile + subprocess。

覆盖：
    - init 幂等（重复 init 不覆盖进度，退出码非零）
    - mark done 缺 artifact / artifact 文件不存在 -> 拒绝并置 failed
    - mark done 带真实文件 -> sha256/size 写入
    - 产物被改写 -> verify 降级 stale 且非零退出
    - inspect --json 定位第一个未完成节点
    - finish success 的三种拒绝（缺证据 / ceiling_bound=false / measured 超限）
    - finish success 两级满足 + 全部 done -> 放行
    - 未完成节点门禁（UNFINISHED_NODES）
    - --force --reason 绕过并在状态文件留痕；--force 缺 --reason 被拒
    - 并发 mark 不丢更新
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'scripts', 'workflow_state.py')


def run(*args, cwd=None):
    """调用 CLI，返回 CompletedProcess（文本模式）。"""
    return subprocess.run(
        [sys.executable, '-I', SCRIPT] + list(args),
        capture_output=True, text=True, cwd=cwd)


class WorkflowStateTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.run_dir = os.path.join(self.tmp, 'run')
        self.roof_dir = os.path.join(self.tmp, 'roofline')
        os.makedirs(self.roof_dir)
        self.state_file = os.path.join(self.run_dir, 'workflow-state.json')

    def tearDown(self):
        self._tmp.cleanup()

    # ---- 辅助 ----
    def init(self, nodes='a,b,c', *extra):
        return run('init', '--run-dir', self.run_dir,
                   '--workflow', 'op-optimize', '--nodes', nodes, *extra)

    def artifact(self, name, content):
        path = os.path.join(self.tmp, name)
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write(content)
        return path

    def mark(self, node, status, *artifacts, note=None):
        args = ['mark', '--run-dir', self.run_dir, '--node', node, '--status', status]
        if artifacts:
            args += ['--artifact'] + list(artifacts)
        if note:
            args += ['--note', note]
        return run(*args)

    def mark_done(self, node, *artifacts):
        """**只用于铺垫**（sha256 / verify / finish / 并发这些用例的脚手架）。

        探针节点名 a/b/c 在本仓库没有同名任务契约，而 done 的字段级校验是失败关闭的
        （无契约即拒，见 test_done_rejects_node_without_contract），所以这里显式跳过，
        并在状态文件留痕。evidence 字段契约本身由本文件末尾的 D4/P1-5 用例覆盖。
        """
        result = run('mark', '--run-dir', self.run_dir, '--node', node,
                     '--status', 'done', '--skip-evidence-check',
                     *(['--artifact'] + list(artifacts) if artifacts else []))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def state(self):
        with open(self.state_file, 'r', encoding='utf-8') as stream:
            return json.load(stream)

    def write_roofline(self, **fields):
        payload = {'ceiling_bound': True, 'measured_us': 100.0, 'ceiling_us': 100.0}
        payload.update(fields)
        with open(os.path.join(self.roof_dir, 'result.json'), 'w', encoding='utf-8') as stream:
            json.dump(payload, stream)
        return payload

    def finish_all_done(self):
        self.init()
        for node in ('a', 'b', 'c'):
            self.mark_done(node, self.artifact('%s.json' % node, node))

    # ---- init ----
    def test_init_is_idempotent_and_does_not_overwrite(self):
        result = self.init()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        state = self.state()
        self.assertEqual(state['workflow'], 'op-optimize')
        self.assertEqual(sorted(state['nodes']), ['a', 'b', 'c'])
        self.assertEqual({n['status'] for n in state['nodes'].values()}, {'pending'})

        result = self.mark('a', 'running')
        self.assertEqual(result.returncode, 0, result.stdout)

        result = self.init()
        self.assertNotEqual(result.returncode, 0, '重复 init 应返回非零')
        self.assertEqual(self.state()['nodes']['a']['status'], 'running',
                         '重复 init 不得覆盖已有进度')

    def test_init_force_resets(self):
        self.init()
        self.mark('a', 'running')
        result = self.init('a,b,c', '--force')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.state()['nodes']['a']['status'], 'pending')

    # ---- mark done 的 artifact 硬要求 ----
    def test_mark_done_without_artifact_is_rejected(self):
        self.init()
        result = self.mark('a', 'done')
        self.assertNotEqual(result.returncode, 0)
        node = self.state()['nodes']['a']
        self.assertEqual(node['status'], 'failed')
        self.assertIn('--artifact', node['note'])

    def test_mark_done_with_missing_artifact_file_is_rejected(self):
        self.init()
        missing = os.path.join(self.tmp, 'nope.json')
        result = self.mark('a', 'done', missing)
        self.assertNotEqual(result.returncode, 0)
        node = self.state()['nodes']['a']
        self.assertEqual(node['status'], 'failed')
        self.assertIn('nope.json', node['note'])

    def test_mark_done_records_artifact_sha256(self):
        self.init()
        path = self.artifact('a.json', 'payload')
        self.mark_done('a', path)
        node = self.state()['nodes']['a']
        self.assertEqual(node['status'], 'done')
        self.assertEqual(node['attempts'], 1)
        self.assertIsNotNone(node['finished_at'])
        meta = node['artifacts'][os.path.abspath(path)]
        with open(path, 'rb') as stream:
            expected = hashlib.sha256(stream.read()).hexdigest()
        self.assertEqual(meta['sha256'], expected)
        self.assertEqual(meta['size'], len(b'payload'))
        self.assertIn('mtime', meta)

    # ---- verify ----
    def test_verify_marks_tampered_artifact_stale(self):
        self.init()
        path = self.artifact('a.json', 'v1')
        self.mark_done('a', path)
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write('v2-changed')

        result = run('verify', '--run-dir', self.run_dir)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.state()['nodes']['a']['status'], 'stale')

        result = run('inspect', '--run-dir', self.run_dir, '--json')
        self.assertEqual(result.returncode, 0, result.stdout)
        report = json.loads(result.stdout)
        artifact = report['nodes'][0]['artifacts'][0]
        self.assertFalse(artifact['valid'])
        self.assertIn('sha256', artifact['reason'])

    def test_verify_passes_on_untouched_artifacts(self):
        self.init()
        self.mark_done('a', self.artifact('a.json', 'stable'))
        result = run('verify', '--run-dir', self.run_dir)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.state()['nodes']['a']['status'], 'done')

    def test_verify_restores_stale_when_content_restored(self):
        self.init()
        path = self.artifact('a.json', 'v1')
        self.mark_done('a', path)
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write('v2')
        self.assertNotEqual(run('verify', '--run-dir', self.run_dir).returncode, 0)
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write('v1')
        self.assertEqual(run('verify', '--run-dir', self.run_dir).returncode, 0)
        self.assertEqual(self.state()['nodes']['a']['status'], 'done')

    # ---- inspect ----
    def test_inspect_json_reports_first_unfinished_node(self):
        self.init('a,b,c')
        self.mark_done('a', self.artifact('a.json', 'a'))

        result = run('inspect', '--run-dir', self.run_dir, '--json')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['stuck'], 'b')
        self.assertEqual(report['nodes_done'], 1)
        self.assertEqual(report['nodes_total'], 3)
        self.assertEqual([n['id'] for n in report['nodes']], ['a', 'b', 'c'])
        self.assertEqual({n['id']: n['status'] for n in report['nodes']},
                         {'a': 'done', 'b': 'pending', 'c': 'pending'})

    def test_inspect_json_rehashes_artifacts_without_verify(self):
        """inspect 自己重算 sha256，不依赖先前 verify 的结论。"""
        self.init('a,b')
        path = self.artifact('a.json', 'v1')
        self.mark_done('a', path)
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write('v2-changed')

        result = run('inspect', '--run-dir', self.run_dir, '--json')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        artifact = json.loads(result.stdout)['nodes'][0]['artifacts'][0]
        self.assertFalse(artifact['valid'])
        self.assertIn('sha256', artifact['reason'])

    def test_inspect_human_output_mentions_stuck_node(self):
        self.init('a,b')
        result = run('inspect', '--run-dir', self.run_dir)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('卡在: a', result.stdout)

    # ---- finish 门禁 ----
    def test_finish_success_rejected_without_roofline_evidence(self):
        self.finish_all_done()
        result = run('finish', '--run-dir', self.run_dir, '--status', 'success')
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('MISSING_ROOFLINE_EVIDENCE', result.stdout)
        self.assertIsNone(self.state()['result'], '被拒时不得写入 success')

    def test_finish_success_rejected_when_ceiling_bound_false(self):
        self.finish_all_done()
        self.write_roofline(ceiling_bound=False)
        result = run('finish', '--run-dir', self.run_dir, '--status', 'success')
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('UNRESOLVED_CEILING', result.stdout)
        self.assertIn('一级天花板', result.stdout)

    def test_finish_success_rejected_when_measured_exceeds_ceiling(self):
        self.finish_all_done()
        # 100 * 1.5 = 150 上限，150.1 超限
        self.write_roofline(ceiling_bound=True, ceiling_us=100.0, measured_us=150.1)
        result = run('finish', '--run-dir', self.run_dir, '--status', 'success')
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('UNRESOLVED_CEILING', result.stdout)
        self.assertIn('二级天花板', result.stdout)

    def test_finish_success_passes_when_both_levels_satisfied(self):
        self.finish_all_done()
        self.write_roofline(ceiling_bound=True, ceiling_us=100.0, measured_us=140.0)
        result = run('finish', '--run-dir', self.run_dir, '--status', 'success')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = self.state()['result']
        self.assertEqual(payload['status'], 'success')
        self.assertFalse(payload['forced'])
        self.assertEqual(payload['bypassed'], [])

    def test_finish_success_rejected_with_unfinished_nodes(self):
        self.init('a,b')
        self.mark_done('a', self.artifact('a.json', 'a'))
        self.write_roofline()
        result = run('finish', '--run-dir', self.run_dir, '--status', 'success')
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('UNFINISHED_NODES', result.stdout)

    def test_finish_success_rechecks_artifacts_before_gate(self):
        """没跑过 verify 也不能让被改写的产物混过 success 门禁。"""
        self.finish_all_done()
        self.write_roofline()
        with open(os.path.join(self.tmp, 'b.json'), 'w', encoding='utf-8') as stream:
            stream.write('tampered-after-done')
        result = run('finish', '--run-dir', self.run_dir, '--status', 'success')
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('UNFINISHED_NODES', result.stdout)
        self.assertIn('b', result.stdout)
        self.assertEqual(self.state()['nodes']['b']['status'], 'stale')
        self.assertIsNone(self.state()['result'])

    def test_finish_failed_is_always_allowed(self):
        self.init('a,b')
        result = run('finish', '--run-dir', self.run_dir,
                     '--status', 'failed', '--reason', '用户叫停')
        self.assertEqual(result.returncode, 0, result.stdout)
        payload = self.state()['result']
        self.assertEqual(payload['status'], 'failed')
        self.assertEqual(payload['reason'], '用户叫停')

    # ---- force 留痕 ----
    def test_force_bypasses_ceiling_and_records_trail(self):
        self.finish_all_done()
        self.write_roofline(ceiling_bound=False)
        result = run('finish', '--run-dir', self.run_dir, '--status', 'success',
                     '--force', '--reason', '用户拍板强行收尾')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = self.state()['result']
        self.assertEqual(payload['status'], 'success')
        self.assertTrue(payload['forced'])
        self.assertEqual(payload['reason'], '用户拍板强行收尾')
        self.assertIn('UNRESOLVED_CEILING', payload['bypassed'])

    def test_force_bypasses_unfinished_nodes_and_records_trail(self):
        self.init('a,b')
        self.mark_done('a', self.artifact('a.json', 'a'))
        self.write_roofline()
        result = run('finish', '--run-dir', self.run_dir, '--status', 'success',
                     '--force', '--reason', '用户拍板先收尾')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = self.state()['result']
        self.assertTrue(payload['forced'])
        self.assertEqual(payload['reason'], '用户拍板先收尾')
        self.assertIn('UNFINISHED_NODES', payload['bypassed'])

    def test_force_without_reason_is_rejected(self):
        self.finish_all_done()
        result = run('finish', '--run-dir', self.run_dir, '--status', 'success', '--force')
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(self.state()['result'])

    # ---- 并发 ----
    def test_concurrent_marks_do_not_lose_updates(self):
        nodes = ['a', 'b', 'c']
        self.init(','.join(nodes))
        artifacts = {n: self.artifact('%s.json' % n, n) for n in nodes}

        # 每节点 2 个并发 mark，共 6 个进程同时抢状态文件
        procs = []
        for node in nodes:
            for _ in range(2):
                procs.append(subprocess.Popen(
                    [sys.executable, '-I', SCRIPT, 'mark',
                     '--run-dir', self.run_dir, '--node', node,
                     '--status', 'done', '--skip-evidence-check',
                     '--artifact', artifacts[node]],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for proc in procs:
            out, err = proc.communicate()
            self.assertEqual(proc.returncode, 0, out + err)

        state = self.state()
        for node in nodes:
            entry = state['nodes'][node]
            self.assertEqual(entry['status'], 'done', '节点 %s 的更新被丢了' % node)
            self.assertEqual(list(entry['artifacts']), [os.path.abspath(artifacts[node])])
        # 顶层 updated_at 必须存在且是最后一次写入的时间
        self.assertIsNotNone(state['updated_at'])

    # ---- 用法 ----
    def test_help_exits_zero(self):
        result = run('--help')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('用法', result.stdout)

    def test_unknown_status_is_usage_error(self):
        self.init()
        result = self.mark('a', 'bogus')
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertEqual(self.state()['nodes']['a']['status'], 'pending')

    # ---- D4: done 必须校验产物内容，不能只看文件存在 ----
    def test_done_rejects_empty_json_without_task(self):
        """一个 `{}` 不是 evidence。旧行为会标绿（D4）。"""
        self.init()
        path = self.artifact('empty.json', '{}')
        result = self.mark_done_expect_fail('a', path)
        self.assertIn('空对象', result.stdout)
        self.assertEqual(self.state()['nodes']['a']['status'], 'failed')

    def test_done_rejects_json_with_content_but_no_contract(self):
        """D4 的另一半：没契约就等于没做字段级校验，不能标绿。

        `{"measured_us": 1.0}` 看着像 evidence，但没人能说出「该有哪些键」。
        本仓库 5 条工作流的每个节点都有同名 tasks/<id>.yaml，无契约是异常。
        """
        self.init()
        path = self.artifact('ok.json', '{"measured_us": 1.0}')
        result = self.mark_done_expect_fail('a', path)
        self.assertIn('没有任务契约', result.stdout)
        self.assertEqual(self.state()['nodes']['a']['status'], 'failed')

    def test_done_rejects_text_artifact_without_contract(self):
        """日志产物同样无契约可验 —— 拒绝而不是「不是 JSON 就放行」。"""
        self.init()
        path = self.artifact('run.log', 'step 1\nstep 2\n')
        result = self.mark_done_expect_fail('a', path)
        self.assertIn('没有任务契约', result.stdout)

    def test_done_with_task_rejects_contract_violation(self):
        """有契约时交给 verify_evidence：产物缺 fingerprint 必须被拒。"""
        self.init()
        path = self.artifact('result.json', '{"node": "b"}')
        args = ['mark', '--run-dir', self.run_dir, '--node', 'b', '--status', 'done',
                '--artifact', path, '--task', self.make_task()]
        result = run(*args)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.state()['nodes']['b']['status'], 'failed')

    def test_contract_check_only_applies_to_declared_evidence_file(self):
        """契约只声明了 result.json；同批的工具原始输出不该被指纹规则误杀。"""
        self.init()
        evidence = self.artifact('result.json', '{"node": "c"}')
        raw = self.artifact('tool-raw.json', '{"anything": 1}')
        result = run('mark', '--run-dir', self.run_dir, '--node', 'c', '--status', 'done',
                     '--artifact', evidence, raw, '--task', self.make_task())
        # evidence 文件本身不合契约 -> 仍要拒
        self.assertNotEqual(result.returncode, 0, result.stdout)
        # 换成合规 evidence + 不合规原始输出 -> 应通过
        good = self.artifact('result.json', json.dumps({
            'path': 'x', 'fingerprint': {'device': 'd', 'dtk': 't',
                                         'image': 'i', 'model': 'm'}}))
        result = run('mark', '--run-dir', self.run_dir, '--node', 'c', '--status', 'done',
                     '--artifact', good, raw, '--task', self.make_task())
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_skip_evidence_check_escapes(self):
        self.init()
        path = self.artifact('empty2.json', '{}')
        result = run('mark', '--run-dir', self.run_dir, '--node', 'a', '--status', 'done',
                     '--artifact', path, '--skip-evidence-check')
        self.assertEqual(result.returncode, 0, result.stdout)

    def mark_done_expect_fail(self, node, *artifacts):
        result = self.mark(node, 'done', *artifacts)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def make_task(self):
        """最小任务契约：要求 evidence.path + 四个公共指纹字段。"""
        path = os.path.join(self.tmp, 'task.yaml')
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write(
                'task:\n'
                '  type: normal\n'
                '  goal: "测试"\n'
                '  evidence:\n'
                '    path: "runs/x/result.json"\n'
                '    fingerprint:\n'
                '      device: "<device>"\n'
                '      dtk: "<dtk>"\n'
                '      image: "<image>"\n'
                '      model: "<model>"\n'
                '  acceptance:\n'
                '    - "占位"\n')
        return path


class ContractAutoLookupTest(unittest.TestCase):
    """P1-5：`mark done` 自己就能找到节点的任务契约，不必每次手喂 `--task`。

    用**仓库里真实的** `tasks/op-roofline.yaml`，不另写一份假的 —— 要证明的正是
    「发出去的那份契约会生效」，用假契约测不出来。
    """

    NODE = 'op-roofline'
    FULL = {
        'device': 'BW1100 (HCU5)', 'dtk': '26.04', 'image': 'n/a（裸机）',
        'model': 'sparse-mla@decode-glm53', 'op': 'sparse-mla',
        'shapes': 'decode-glm53', 'cost_bound': 'compute',
        'ceiling_us': 77.9, 'measured_us': 329.5,
        'floor_attainment': 0.2364, 'ceiling_bound': False,
    }

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.run_dir = os.path.join(self.tmp, 'run')

    def tearDown(self):
        self._tmp.cleanup()

    def mark(self, payload, *extra):
        path = os.path.join(self.tmp, 'result.json')   # 契约里 evidence.path 的文件名
        # verify_evidence 硬性要求 evidence.path 存在（真实交付件都带），补上再写。
        if isinstance(payload, dict) and 'path' not in payload:
            payload = dict(payload, path='runs/op-optimize/<run>/roofline/result.json')
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write(payload if isinstance(payload, str) else json.dumps(payload))
        if not os.path.exists(os.path.join(self.run_dir, 'workflow-state.json')):
            run('init', '--run-dir', self.run_dir, '--workflow', 'op-optimize',
                '--nodes', self.NODE)
        return run('mark', '--run-dir', self.run_dir, '--node', self.NODE,
                   '--status', 'done', '--artifact', path, *extra)

    def test_complete_fingerprint_passes(self):
        result = self.mark({'fingerprint': self.FULL})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_deleting_one_key_is_rejected(self):
        """删掉 fingerprint.floor_attainment —— 旧行为下这份产物能标绿（D4）。"""
        fp = {k: v for k, v in self.FULL.items() if k != 'floor_attainment'}
        result = self.mark({'fingerprint': fp})
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('floor_attainment', result.stdout + result.stderr)

    def test_placeholder_value_is_rejected(self):
        """键在但值是契约里的占位符 = 没填，不算 evidence。"""
        fp = dict(self.FULL, ceiling_us='<theoretical-floor-us>')
        result = self.mark({'fingerprint': fp})
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('ceiling_us', result.stdout + result.stderr)

    def test_fingerprint_at_top_level_is_not_enough(self):
        """字段放错层也算缺 —— 契约的权威点是 `fingerprint.*`，不是「文件里有这个字符串」。"""
        result = self.mark(dict(self.FULL))
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('fingerprint', result.stdout + result.stderr)

    def test_explicit_task_overrides_auto_lookup(self):
        """`--task` 显式指定时以它为准（临时契约优先于按节点名找到的那份）。"""
        task = os.path.join(self.tmp, 'task.yaml')
        with open(task, 'w', encoding='utf-8') as stream:
            stream.write('task:\n  type: normal\n  goal: "x"\n  evidence:\n'
                         '    path: "runs/x/result.json"\n    fingerprint:\n'
                         '      device: "<device>"\n')
        result = self.mark({'fingerprint': {'device': 'BW1100 (HCU5)'}}, '--task', task)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ResumeTest(unittest.TestCase):
    """P1-7 断点续跑：kill 之后重开，能不能确定性地接着走。

    续跑真正的风险不是「读不回状态」，是三种误判：
      - done 就跳过（产物其实已被改写）
      - running 当成还在跑（进程早被 kill 了）
      - failed 无限重试（同一处反复烧时间）
    这三条各有一个用例。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.run_dir = os.path.join(self.tmp, 'run')

    def tearDown(self):
        self._tmp.cleanup()

    def init(self, nodes='op-profile,op-roofline,op-constrain',
             workflow='op-optimize'):
        return run('init', '--run-dir', self.run_dir,
                   '--workflow', workflow, '--nodes', nodes)

    def resume_json(self):
        cp = run('resume', '--run-dir', self.run_dir, '--json')
        self.assertIn(cp.returncode, (0, 1), cp.stdout + cp.stderr)
        return json.loads(cp.stdout), cp.returncode

    def artifact(self, name, content):
        path = os.path.join(self.tmp, name)
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write(content)
        return path

    def mark(self, node, status, *artifacts):
        return run('mark', '--run-dir', self.run_dir, '--node', node,
                   '--status', status, '--skip-evidence-check',
                   *(['--artifact'] + list(artifacts) if artifacts else []))

    def test_fresh_run_points_at_first_node(self):
        self.init()
        plan, rc = self.resume_json()
        self.assertEqual(rc, 0)
        self.assertEqual(plan['next'], 'op-profile')
        self.assertEqual(len(plan['todo']), 3)
        self.assertEqual(plan['keep'], [])

    def test_done_node_is_not_rerun(self):
        """产物 sha256 复核有效 -> 不要重跑（这是续跑省时间的全部意义）。"""
        self.init()
        path = self.artifact('p.out', 'ok')
        self.mark('op-profile', 'done', path)
        plan, _ = self.resume_json()
        self.assertEqual([e['id'] for e in plan['keep']], ['op-profile'])
        self.assertEqual(plan['next'], 'op-roofline')

    def test_tampered_artifact_forces_redo(self):
        """done 不等于可以跳过：产物被改写就必须重做，否则拿一份不存在的结果往下走。"""
        self.init()
        path = self.artifact('p.out', 'ok')
        self.mark('op-profile', 'done', path)
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write('tampered')
        plan, _ = self.resume_json()
        self.assertEqual([e['id'] for e in plan['keep']], [])
        entry = next(e for e in plan['todo'] if e['id'] == 'op-profile')
        self.assertEqual(entry['status'], 'stale')
        self.assertIn('sha256', entry['why'])

    def test_deleted_artifact_forces_redo(self):
        self.init()
        path = self.artifact('p.out', 'ok')
        self.mark('op-profile', 'done', path)
        os.unlink(path)
        plan, _ = self.resume_json()
        entry = next(e for e in plan['todo'] if e['id'] == 'op-profile')
        self.assertIn('文件不存在', entry['why'])

    def test_running_is_treated_as_interrupted(self):
        """进程被 kill 时状态永远停在 running，没人会翻回来 —— 必须当成要重做。"""
        self.init()
        self.mark('op-profile', 'running')
        plan, _ = self.resume_json()
        entry = next(e for e in plan['todo'] if e['id'] == 'op-profile')
        self.assertEqual(entry['recorded_status'], 'running')
        self.assertIn('kill', entry['why'])

    def test_failed_within_retries_is_retried(self):
        self.init()
        self.mark('op-profile', 'failed')
        plan, _ = self.resume_json()
        self.assertEqual(plan['blocked'], [])
        self.assertIn('op-profile', [e['id'] for e in plan['todo']])

    def test_failed_beyond_retries_is_blocked(self):
        """重试上限读 workflows/<wf>.yaml 的 retries（op-roofline 是 1）。"""
        self.init()
        for _ in range(3):
            self.mark('op-roofline', 'failed')
        plan, _ = self.resume_json()
        blocked = [e['id'] for e in plan['blocked']]
        self.assertEqual(blocked, ['op-roofline'])
        self.assertIn('retries=1', plan['blocked'][0]['why'])
        self.assertNotIn('op-roofline', [e['id'] for e in plan['todo']])

    def test_attempts_accumulate_on_repeated_failure(self):
        """P1-7 依赖这个计数：不累加的话 retries 上限永远触发不了。"""
        self.init()
        for i in (1, 2, 3):
            self.mark('op-roofline', 'failed')
            with open(os.path.join(self.run_dir, 'workflow-state.json'),
                      encoding='utf-8') as stream:
                state = json.load(stream)
            self.assertEqual(state['nodes']['op-roofline']['attempts'], i)

    def test_skipped_is_not_rerun(self):
        """显式跳过是作者的决定，不是中断。"""
        self.init()
        self.mark('op-constrain', 'skipped')
        plan, _ = self.resume_json()
        self.assertEqual([e['id'] for e in plan['keep']], ['op-constrain'])
        self.assertNotIn('op-constrain', [e['id'] for e in plan['todo']])

    def test_all_done_exits_one(self):
        self.init('op-profile')
        self.mark('op-profile', 'done', self.artifact('p.out', 'ok'))
        plan, rc = self.resume_json()
        self.assertIsNone(plan['next'])
        self.assertEqual(rc, 1, '没有可做的工作时返回 1，方便脚本判断')

    def test_unknown_workflow_does_not_crash(self):
        """拿不到工作流定义 -> 说明「重试上限未知」，不静默假设一个数字。"""
        self.init(workflow='no-such-workflow')
        self.mark('op-profile', 'failed')
        plan, _ = self.resume_json()
        self.assertIn('找不到工作流定义', plan['limits_error'])
        self.assertIn('op-profile', [e['id'] for e in plan['todo']])

    def test_resume_does_not_modify_state(self):
        """resume 是只读的：它只算计划，不能顺手改状态文件。"""
        self.init()
        self.mark('op-profile', 'running')
        state_path = os.path.join(self.run_dir, 'workflow-state.json')
        with open(state_path, 'rb') as stream:
            before = stream.read()
        run('resume', '--run-dir', self.run_dir)
        with open(state_path, 'rb') as stream:
            self.assertEqual(stream.read(), before)

    def test_resume_works_after_real_kill(self):
        """真中断：起一个跑着 mark --status running 的进程，SIGKILL 掉，再续跑。"""
        self.init()
        state_path = os.path.join(self.run_dir, 'workflow-state.json')
        proc = subprocess.Popen(
            [sys.executable, '-I', SCRIPT, 'mark', '--run-dir', self.run_dir,
             '--node', 'op-profile', '--status', 'running'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.kill()                      # SIGKILL：没有任何清理机会
        proc.wait()

        plan, _ = self.resume_json()
        # 不管 mark 是否来得及落盘，续跑都必须把这个节点当成「要做」
        self.assertIn('op-profile', [e['id'] for e in plan['todo']])
        self.assertNotIn('op-profile', [e['id'] for e in plan['keep']])
        self.assertTrue(os.path.isfile(state_path), '状态文件不能被 kill 弄坏/删掉')
        with open(state_path, encoding='utf-8') as stream:
            json.load(stream)            # 仍是合法 JSON


if __name__ == '__main__':
    unittest.main()
