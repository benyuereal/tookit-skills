#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kbq — 用户知识库（`~/.config/dcu-toolkit/kb/`）的索引与检索入口（T10 / P1-6）。

问题
----
沉淀写了很多，但**下次要用的时候找不到**。`kb/op-opt/` 里躺着的结论，只有
「记得文件名」或「全文 grep」两条路；而 grep 出来的是一堆行，看不出哪个文件、
哪一段、这张卡是不是已经过期/缺口径。

做法
----
两个子命令，都只读、不改任何文件：

    index             列出全部条目 + 格式合规性（一张表）
    search <kw>...    按关键词检索，命中处标出「属于哪一段」

**格式合规性**是 P1-6 的另一半：沉淀统一成五段
`结论 / 前提 / 已证伪 / 复现命令 / 口径`。
缺 `口径` 的卡在检索输出里**标红**——不是拒绝检索，是让「这条结论没写口径」看得见。
口径的含义：同一条数据换个分母/频率档/标称峰值，结论可能反转；不写清楚就无法复用。

五段清单的权威点在这里（`SECTIONS`）。生成方 `dcu_distill.py` 与人工模板
`data/kb/templates/op-opt-ledger-template.md` 都不各写一份，靠
`tests/test_kbq.py::test_distill_ledger_passes_format_check` 把三者钉在一起：
生成物过不了本检查器，测试就红。**没挂到机器上的规则等于没规则。**

刻意不做
--------
- **不改文件、不建索引文件**：`kb/` 是用户态沉淀，install.sh 永不覆盖它；
  再往里塞一个会被写坏的 `.index.json`，等于多一个会过期的权威点。每次现扫（2 张卡 64KB，毫秒级）。
- **不做语义检索**：没有 embedding、没有打分排序。这库是要「精确回到某条结论」的，
  模糊排序会让最相关的那条排第二。
- **不合规不拦**：历史卡不会因为缺一段就检索不到。缺段落是标记，不是门禁。

退出码
------
index  : 0（不论是否全部合规）
search : 0=有命中，1=没命中（可直接用于脚本判断）
"""
import argparse
import json
import os
import re
import sys

DEFAULT_KB = os.path.expanduser('~/.config/dcu-toolkit/kb')

# 沉淀五段（P1-6 口径）。值是「标题里出现任一就算这一段在」的别名表：
# 老卡写「复现脚本」和写「复现命令」是同一个概念，不该因为措辞被判不合规。
# 别名刻意保守：只收明确同义的写法。把「§ 9.4 被否掉的三条」这种也算进「已证伪」看着更绿，
# 但那正是 P1-6 要暴露的格式缺口 —— 让检查器替作者猜标题，等于把不合规藏起来。
SECTIONS = (
    ('结论', ('结论', '结论与')),
    ('前提', ('前提', '假设', '适用范围')),
    ('已证伪', ('已证伪', '证伪', '已排除', '已否')),
    ('复现命令', ('复现',)),
    ('口径', ('口径', '标定条件')),
)
REQUIRED = tuple(name for name, _ in SECTIONS)

HEADING = re.compile(r'^(#{1,6})\s+(.*)$')
# 提案骨架不是台账（它写的是「拟并入什么」，不是结论），不套五段规则
NOT_LEDGER = re.compile(r'^promotion-.*\.md$')
SKIP_SUFFIX = ('.bak', '.json')


def strip_md(text):
    """去掉代码围栏，避免把示例里的 `## 结论` 当成真标题。"""
    out, fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith('```'):
            fence = not fence
            out.append('')
            continue
        out.append('' if fence else line)
    return out


def headings(lines):
    """[(行号(1-based), 级别, 标题文本)]"""
    found = []
    for i, line in enumerate(lines, 1):
        m = HEADING.match(line)
        if m:
            found.append((i, len(m.group(1)), m.group(2).strip()))
    return found


def sections_present(lines):
    """这张卡实际有哪几段（按别名表判定）。"""
    have = set()
    for _, _, title in headings(lines):
        for name, aliases in SECTIONS:
            if any(a in title for a in aliases):
                have.add(name)
    return have


def missing_sections(lines):
    have = sections_present(lines)
    return [name for name in REQUIRED if name not in have]


def is_ledger(path):
    return not NOT_LEDGER.match(os.path.basename(path))


def read_lines(path):
    with open(path, encoding='utf-8', errors='replace') as fh:
        return strip_md(fh.read())


def walk(kb, subdir=None):
    """遍历条目文件。返回 [(abs_path, 相对 kb 的显示名)]，跳过备份与机器可读 JSON。"""
    root = os.path.join(kb, subdir) if subdir else kb
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(SKIP_SUFFIX) or not name.endswith('.md'):
                continue
            if name in ('README.md', 'INDEX.md'):
                continue
            full = os.path.join(dirpath, name)
            out.append((full, os.path.relpath(full, kb)))
    return out


def section_of(lines, lineno):
    """某一行的最近上级标题（给检索命中标注「属于哪一段」）。"""
    best = ''
    for i, _, title in headings(lines):
        if i > lineno:
            break
        best = title
    return best


def _color(s, code, enabled):
    return '\033[%sm%s\033[0m' % (code, s) if enabled else s


def cmd_index(args):
    entries = walk(args.kb, args.dir)
    if not entries:
        print('（%s 下没有任何条目）' % args.kb)
        return 0

    rows, bad = [], 0
    for full, rel in entries:
        lines = read_lines(full)
        if not is_ledger(rel):
            rows.append((rel, '提案', '—'))
            continue
        miss = missing_sections(lines)
        if miss:
            bad += 1
            rows.append((rel, '✗', ', '.join(miss)))
        else:
            rows.append((rel, '✓', ''))

    if args.json:
        print(json.dumps([{'path': r, 'ok': s == '✓', 'missing': m}
                          for r, s, m in rows], ensure_ascii=False, indent=2))
        return 0

    print('知识库：%s' % args.kb)
    print('（五段口径：%s）\n' % ' / '.join(REQUIRED))
    w = max(len(r) for r, _, _ in rows)
    for rel, flag, detail in rows:
        line = '%-*s  %s %s' % (w, rel, flag, detail)
        if flag == '✗':
            line = _color(line, '31', args.color)
        print(line)
    print('\n共 %d 条，不合规 %d 条。' % (len(rows), bad))
    return 0


def cmd_search(args):
    terms = [t.lower() for t in args.terms]
    entries = walk(args.kb, args.dir)
    hits_files = []
    total = 0

    for full, rel in entries:
        lines = read_lines(full)
        hits = []
        for i, line in enumerate(lines, 1):
            low = line.lower()
            if any(t in low for t in terms):
                if not line.strip():
                    continue
                sec = section_of(lines, i)
                hits.append((i, sec, line.strip()))
        if not hits:
            continue
        total += len(hits)
        hits_files.append((rel, lines, hits))
    if not hits_files:
        print('（没有命中：%s）' % ' '.join(args.terms))
        return 1

    if args.json:
        print(json.dumps([{
            'path': rel,
            'missing_sections': [] if not is_ledger(rel) else missing_sections(lines),
            'hits': [{'line': i, 'section': sec, 'text': text} for i, sec, text in hits],
        } for rel, lines, hits in hits_files], ensure_ascii=False, indent=2))
        return 0

    for rel, lines, hits in hits_files:
        miss = [] if not is_ledger(rel) else missing_sections(lines)
        if miss:
            # 标红：这条结论缺口径/前提等，引用前必须先补齐
            head = _color('✗ %s  [缺: %s]' % (rel, ', '.join(miss)), '31', args.color)
        else:
            head = _color('✓ %s' % rel, '32', args.color)
        print(head)
        for i, sec, text in hits:
            where = ('  § %s' % sec) if sec else ''
            print('  L%-5d%s  %s' % (i, where, text[:args.width]))
        print()

    print('共 %d 个文件、%d 处命中。' % (len(hits_files), total))
    if any(is_ledger(rel) and missing_sections(lines) for rel, lines, _ in hits_files):
        print(_color('注意：上面标 ✗ 的卡缺段，引用其结论前先补齐'
                     '（尤其「口径」：换分母/频率档结论可能反转）。', '33', args.color))
    return 0


def main(argv=None):
    # 全局参数（--kb/--dir/--json/--no-color）在**子命令前后都要能写**：
    # `kbq search 口径 --json` 是最自然的写法，argparse 默认只认前置。
    # 做法是给子解析器也挂一份，默认值设为 SUPPRESS —— 否则子解析器的默认值
    # 会把前面已经解析出来的值覆盖回去（argparse 的经典坑）。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--kb', default=argparse.SUPPRESS, help='知识库根')
    common.add_argument('--dir', default=argparse.SUPPRESS, help='只查某子目录')
    common.add_argument('--json', action='store_true', default=argparse.SUPPRESS,
                        help='机器可读输出')
    common.add_argument('--no-color', dest='color', action='store_false',
                        default=argparse.SUPPRESS, help='禁用 ANSI 颜色')

    ap = argparse.ArgumentParser(
        prog='kbq', description='用户知识库（kb/）的索引与检索', parents=[common])
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_index = sub.add_parser('index', parents=[common],
                             help='列出全部条目 + 五段合规性')
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser('search', parents=[common], help='按关键词检索')
    p_search.add_argument('terms', nargs='+', help='关键词（任一命中即列出，不区分大小写）')
    p_search.add_argument('--width', type=int, default=200, help='命中行截断宽度')
    p_search.set_defaults(func=cmd_search)

    args = ap.parse_args(argv)
    # 默认值在这里补，**不能用 ap.set_defaults(...)**：parents 是复用 action 对象，
    # set_defaults 会把那个 --json action 的 default 从 SUPPRESS 改成 False，
    # 于是子解析器新建 namespace 时又写成 False、合并回来覆盖掉前置 `--json`。
    for name, default in (('kb', DEFAULT_KB), ('dir', None),
                          ('json', False), ('color', True)):
        if not hasattr(args, name):
            setattr(args, name, default)
    # 颜色只在真终端开启：重定向到文件/管道时混进 ANSI 转义会毁掉日志
    args.color = args.color and sys.stdout.isatty()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
