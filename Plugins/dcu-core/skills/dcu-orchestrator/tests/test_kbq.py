#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kbq.py 的单测（T10 / P1-6 知识回流可检索）。

被测脚本：<repo>/Plugins/dcu-core/skills/dcu-orchestrator/scripts/kbq.py
无卡环境可跑：只碰临时目录里的 markdown。

钉住的几条纪律：
  - 五段（结论/前提/已证伪/复现命令/口径）缺一段就报缺，不猜标题
  - 代码围栏里的 `## 结论` 不算真标题（示例不能骗过检查器）
  - promotion-*.md 是提案骨架，不套五段规则
  - 缺段落只标记、**不阻断检索**（历史卡不能因为格式老旧就查不到）
  - `--json` 写在子命令前后都必须生效（argparse parents 的经典坑）
  - 生成方 dcu_distill.py 产出的台账必须过本检查器（否则格式必然漂）
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 注意层数：本文件在 Plugins/dcu-core/skills/dcu-orchestrator/tests/ 下，
# 比 Plugins/ops/tests/ 深一层，所以是 parents[4] 而不是 [3]。
REPO = HERE.parents[4]
SCRIPT = REPO / 'Plugins' / 'dcu-core' / 'skills' / 'dcu-orchestrator' / 'scripts' / 'kbq.py'
DISTILL = SCRIPT.parent / 'dcu_distill.py'

sys.path.insert(0, str(SCRIPT.parent))
import kbq  # noqa: E402


FULL = """# 卡片：{name}

## 结论

- 结论一句话。

## 前提

- 前提一句话。

## 已证伪

- 曾以为 X，实测不是。

## 复现命令

```bash
python3 foo.py
```

## 口径

- 分母是 128 MB。
"""


def write_kb(root, rel, text):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')
    return p


def run_kbq(*args):
    return subprocess.run([sys.executable, '-I', str(SCRIPT)] + list(args),
                          capture_output=True, text=True)


def render(*args, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = kbq.main(list(args))
    return rc, buf.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.kb = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()


class FormatCheckTest(Base):
    def test_complete_card_is_ok(self):
        write_kb(self.kb, 'op-opt/a.md', FULL.format(name='a'))
        rc, out = render('--kb', self.kb, 'index')
        self.assertEqual(rc, 0)
        self.assertIn('✓', out)
        self.assertIn('共 1 条，不合规 0 条', out)

    def test_missing_one_section_is_reported(self):
        text = FULL.format(name='a').replace('## 口径', '## 别的东西')
        write_kb(self.kb, 'op-opt/a.md', text)
        rc, out = render('--kb', self.kb, 'index')
        self.assertEqual(rc, 0, '缺段不是错误退出码')
        self.assertIn('口径', out)
        self.assertIn('不合规 1 条', out)

    def test_each_section_independently_required(self):
        """五段是**各自**必填，不是「有五段就行」。逐个删一遍。"""
        for want in ('结论', '前提', '已证伪', '复现命令', '口径'):
            with self.subTest(section=want):
                text = FULL.format(name='x').replace('## %s' % want, '## 其它')
                write_kb(self.kb, 'op-opt/x.md', text)
                lines = kbq.read_lines(str(Path(self.kb) / 'op-opt' / 'x.md'))
                self.assertIn(want, kbq.missing_sections(lines))

    def test_fenced_heading_does_not_count(self):
        """代码块里的 `## 结论` 是示例，不能当成真标题。"""
        text = FULL.format(name='a').replace('## 前提', '## 其它')
        text += '\n```markdown\n## 前提\n```\n'
        write_kb(self.kb, 'op-opt/a.md', text)
        lines = kbq.read_lines(str(Path(self.kb) / 'op-opt' / 'a.md'))
        self.assertIn('前提', kbq.missing_sections(lines))

    def test_alias_forms_count(self):
        """「复现脚本」与「复现命令」是同一概念；措辞不同不该判不合规。"""
        text = (FULL.format(name='a')
                .replace('## 复现命令', '## 复现脚本')
                .replace('## 已证伪', '## 已证伪 / 已排除'))
        write_kb(self.kb, 'op-opt/a.md', text)
        lines = kbq.read_lines(str(Path(self.kb) / 'op-opt' / 'a.md'))
        self.assertEqual(kbq.missing_sections(lines), [])

    def test_promotion_file_is_not_a_ledger(self):
        write_kb(self.kb, 'op-opt/promotion-x-20261028.md', '# 晋升提案\n\n## 拟并入\n')
        rc, out = render('--kb', self.kb, 'index')
        self.assertEqual(rc, 0)
        self.assertIn('不合规 0 条', out, '提案骨架不套五段规则')

    def test_backup_and_json_files_are_skipped(self):
        write_kb(self.kb, 'op-opt/a.md.bak', 'garbage')
        write_kb(self.kb, 'op-opt/a.json', '{}')
        write_kb(self.kb, 'op-opt/README.md', 'readme')
        rc, out = render('--kb', self.kb, 'index')
        self.assertIn('没有任何条目', out)


class SearchTest(Base):
    def setUp(self):
        super().setUp()
        write_kb(self.kb, 'op-opt/good.md', FULL.format(name='good'))
        bad = FULL.format(name='bad').replace('## 口径', '## 其它')
        write_kb(self.kb, 'op-opt/bad.md', bad)

    def test_hit_reports_file_and_section(self):
        rc, out = render('--kb', self.kb, 'search', '结论一句话')
        self.assertEqual(rc, 0)
        self.assertIn('good.md', out)
        self.assertIn('§ 结论', out, '命中要标出属于哪一段')

    def test_missing_caliber_card_is_flagged(self):
        """缺「口径」的卡要标红/标缺 —— 这是 P1-6 的验收点。"""
        rc, out = render('--kb', self.kb, '--no-color', 'search', '结论一句话')
        self.assertEqual(rc, 0)
        self.assertIn('✗ op-opt/bad.md', out)
        self.assertIn('缺: 口径', out)

    def test_missing_section_does_not_block_search(self):
        """格式老的卡照样能检索到，只是被标记。"""
        rc, out = render('--kb', self.kb, 'search', '分母是 128 MB')
        self.assertEqual(rc, 0)
        self.assertIn('bad.md', out)

    def test_no_hit_exits_one(self):
        rc, out = render('--kb', self.kb, 'search', '这句话不存在zzz')
        self.assertEqual(rc, 1)
        self.assertIn('没有命中', out)

    def test_case_insensitive(self):
        write_kb(self.kb, 'op-opt/c.md', FULL.format(name='c') + '\n- MFMA 发射率\n')
        rc, out = render('--kb', self.kb, 'search', 'mfma')
        self.assertEqual(rc, 0)
        self.assertIn('c.md', out)

    def test_dir_scopes_search(self):
        write_kb(self.kb, 'issue/i.md', FULL.format(name='i') + '\n- 结论一句话 extra\n')
        rc, out = render('--kb', self.kb, '--dir', 'issue', 'search', '结论一句话')
        self.assertEqual(rc, 0)
        self.assertIn('issue/i.md', out)
        self.assertNotIn('op-opt/good.md', out)


class CliTest(Base):
    def test_json_flag_works_before_and_after_subcommand(self):
        """`search kw --json` 是最自然的写法；argparse parents 会把它吞掉。"""
        write_kb(self.kb, 'op-opt/a.md', FULL.format(name='a'))
        for args in (['--kb', self.kb, '--json', 'index'],
                     ['--kb', self.kb, 'index', '--json']):
            with self.subTest(args=args):
                cp = run_kbq(*args)
                self.assertEqual(cp.returncode, 0, cp.stderr)
                json.loads(cp.stdout)   # 必须是合法 JSON

    def test_search_json_shape(self):
        write_kb(self.kb, 'op-opt/a.md', FULL.format(name='a'))
        cp = run_kbq('--kb', self.kb, 'search', '结论一句话', '--json')
        data = json.loads(cp.stdout)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['missing_sections'], [])
        self.assertEqual(data[0]['hits'][0]['section'], '结论')

    def test_piped_output_has_no_ansi(self):
        write_kb(self.kb, 'op-opt/a.md', FULL.format(name='a').replace('## 口径', '## x'))
        cp = run_kbq('--kb', self.kb, 'search', '结论')
        self.assertNotIn('\033[', cp.stdout, '非 tty 不许混进 ANSI 转义')

    def test_missing_kb_dir_is_not_a_crash(self):
        cp = run_kbq('--kb', os.path.join(self.kb, 'nope'), 'index')
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertIn('没有任何条目', cp.stdout)


class DistillContractTest(Base):
    """生成方 ↔ 检查方的契约：dcu_distill.py 产出的台账必须过 kbq 五段检查。

    这是「一个概念一个权威点」的落点 —— 五段清单只在 kbq.SECTIONS 写一份，
    生成方必须满足它。生成方改模板后忘了同步，这条测试就红。
    """

    def test_distill_ledger_passes_format_check(self):
        run = Path(self.kb) / 'validate-result.json'
        run.write_text(json.dumps({
            'op': 'demo-op', 'device': 'BW1100', 'gfx': 'gfx938', 'dtk': '6.3',
            'framework': 'tilelang 0.1.12', 'image': 'n/a',
            'baseline': {'median_ms': 2.0}, 'metrics': {'median_ms': 1.5},
            'gain': 33.3, 'accuracy': 'ok',
        }), encoding='utf-8')
        kbdir = Path(self.kb) / 'kb'
        cp = subprocess.run([sys.executable, '-I', str(DISTILL),
                             '--run', str(run), '--kb', str(kbdir)],
                            capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)

        ledgers = [p for p in (kbdir / 'op-opt').glob('*.md')
                   if not p.name.startswith('promotion-')]
        self.assertEqual(len(ledgers), 1, cp.stdout)
        lines = kbq.read_lines(str(ledgers[0]))
        self.assertEqual(kbq.missing_sections(lines), [],
                         'dcu_distill.py 生成的台账缺段 —— 生成方与 kbq.SECTIONS 漂了')

    def test_handwritten_template_passes_format_check(self):
        """人工模板是第三个权威点，同样必须过检查器（否则手写卡一落地就红）。"""
        tpl = REPO / 'data' / 'kb' / 'templates' / 'op-opt-ledger-template.md'
        self.assertTrue(tpl.is_file(), tpl)
        self.assertEqual(kbq.missing_sections(kbq.read_lines(str(tpl))), [])


if __name__ == '__main__':
    unittest.main()
