#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""skill_lint.py — dcu-toolkit-skills 结构一致性门禁

只依赖标准库，Python 3.8+。

用法:
    python3 skill_lint.py --root <repo>                 # 打印摘要 + 写 lint-report.json
    python3 skill_lint.py --root <repo> --show SK006    # 只看某条规则的明细
    python3 skill_lint.py --root <repo> --baseline lint-baseline.json
                                                        # 只报「比 baseline 新增」的违规，新增则 exit 1

设计原则:
  1. 只做可机器判定的检查，不做风格偏好判定。
  2. 存量违规用 baseline 冻结，CI 只拦新增 —— 避免变成第二个空转门禁。
  3. 每条违规都带 file:line 或 file，能直接定位到待改的位置。
"""

import argparse
import json
import os
import re
import sys

KEBAB = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')
CONVENTION_FILES = {
    'SKILL.md', 'README.md', 'CHANGELOG.md', 'LICENSE', 'AGENTS.md',
    'CONTRIBUTING.md', 'Dockerfile', 'Makefile', 'NOTICE',
    'THIRD_PARTY_NOTICES.md', 'HANDOFF.md', 'package.env', 'sanitize.py',
}
PREFIX_ALLOW = ('dcu-', 'op-')
MAIN_SCHEMA = {'name', 'description', 'version', 'license', 'kind'}
LEGACY_FIELDS = {'metadata', 'shard', 'standalone', 'capabilities', 'intents',
                 'recommended_companions', 'category', 'related_skills'}
ACCEPT_PAT = re.compile(r'^#{2,3}\s*.*(验收|判定|通过标准|完成标准|验收标准|acceptance)', re.M)
NA_PAT = re.compile(r'^#{2,3}\s*.*(不适用|何时不用|不适用场景|不要用|when not)', re.M | re.I)
PITFALL_PAT = re.compile(r'^#{2,3}\s*.*(陷阱|坑|pitfall|常见错误|失败模式)', re.M | re.I)
CJK = re.compile(r'[\u4e00-\u9fff]')
LATIN = re.compile(r'[A-Za-z]{4,}')

RULES = {
    'SK001': '骨架符合声明的 kind（knowledge/tool/card/composite）',
    'SK002': '必备章节齐全（验收 / 不适用）',
    'SK003': '文件命名 kebab-case（非 .py 不用下划线，不用中文，不用大写）',
    'SK004': 'skill 目录名前缀在白名单内（dcu- / op-）',
    'SK005': 'frontmatter 只用主 schema，不用历史字段',
    'SK006': '共享资源引用写 ../_dcu-toolkit-shared/，不用裸相对路径',
    'SK007': '正文引用的脚本必须存在',
    'SK008': 'md 相对链接必须可解析',
    'SK009': 'description 语言统一（纯中文或中英混，不出现纯英文）',
    'SK010': '插件 SKILL.md 版本号与 marketplace.json 一致',
}


def read(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        return f.read()


def frontmatter(text):
    m = re.match(r'^---\n(.*?)\n---', text, re.S)
    return m.group(1) if m else ''


def collect_skills(root):
    """返回 [(skill_dir, plugin, rel_dir)]，含嵌套子 skill。"""
    out = []
    plugins_dir = os.path.join(root, 'Plugins')
    for plugin in sorted(os.listdir(plugins_dir)):
        sd = os.path.join(plugins_dir, plugin, 'skills')
        if not os.path.isdir(sd):
            continue
        for name in sorted(os.listdir(sd)):
            d = os.path.join(sd, name)
            if os.path.isfile(os.path.join(d, 'SKILL.md')):
                out.append((d, plugin, f'Plugins/{plugin}/skills/{name}'))
            nested = os.path.join(d, 'skills')
            if os.path.isdir(nested):
                for sub in sorted(os.listdir(nested)):
                    d2 = os.path.join(nested, sub)
                    if os.path.isfile(os.path.join(d2, 'SKILL.md')):
                        out.append((d2, plugin,
                                    f'Plugins/{plugin}/skills/{name}/skills/{sub}'))
    return out


_BASENAME_CACHE = {}

_caller_body = {'v': ''}


def _is_upstream_ref(token):
    """判断脚本引用是否明确指向上游仓库（而非本仓）。

    判据：同一行里出现 `/path/to/`、`<sglang>`，或 `python/sglang/` 这类
    上游目录前缀。这些不是本仓文件，SK007 不应要求它们存在。
    """
    body = _caller_body['v']
    # 整篇 skill 明确是「上游仓库操作指南」时，其中的相对路径指上游，不是本仓
    head = body[:1200].lower()
    if 'upstream' in head and ('.github/workflows' in body or 'python/sglang' in body):
        return True
    for line in body.splitlines():
        if token in line:
            if '/path/to/' in line or 'python/sglang/' in line:
                return True
    return False


def _find_by_basename(root, base):
    """全仓按 basename 找文件（跳过 .git / __pycache__），结果进程内缓存。"""
    if base not in _BASENAME_CACHE:
        hit = None
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames
                           if x not in ('.git', '__pycache__', 'node_modules')]
            if base in filenames:
                hit = os.path.join(dirpath, base)
                break
        _BASENAME_CACHE[base] = hit
    return _BASENAME_CACHE[base]


def classify_kind(d):
    """从目录形态推断 kind —— 用于 SK001 的『声明 vs 实际』对照。"""
    has_s = os.path.isdir(os.path.join(d, 'scripts'))
    has_r = os.path.isdir(os.path.join(d, 'references'))
    card = any(os.path.isfile(os.path.join(d, f))
               for f in ('contract.yaml', 'variants.yaml', 'ledger.md'))
    wf = os.path.isdir(os.path.join(d, 'workflows'))
    if card:
        return 'card'
    if wf:
        return 'composite'
    if has_s or has_r:
        return 'tool'
    return 'knowledge'


def lint(root):
    v = {k: [] for k in RULES}
    skills = collect_skills(root)
    stat = {'skills': len(skills), 'kinds': {}, 'skeletons': set()}

    # ---------- 逐 skill ----------
    for d, plugin, rel in skills:
        sk = os.path.join(d, 'SKILL.md')
        text = read(sk)
        fm = frontmatter(text)
        body = text[len(fm) + 8:] if fm else text

        # SK001 骨架
        subs = tuple(sorted(x for x in os.listdir(d)
                            if os.path.isdir(os.path.join(d, x))))
        stat['skeletons'].add(subs)
        kind = classify_kind(d)
        stat['kinds'][kind] = stat['kinds'].get(kind, 0) + 1
        declared = re.search(r'^kind\s*:\s*(\w+)', fm, re.M)
        if declared is None:
            v['SK001'].append(f'{rel}  未声明 kind（实际形态推断为 {kind}）')

        # SK002 必备章节
        if not ACCEPT_PAT.search(body):
            v['SK002'].append(f'{sk}:0  缺「验收/判定」章节')
        if not NA_PAT.search(body):
            v['SK002'].append(f'{sk}:0  缺「不适用场景」章节')

        # SK004 前缀
        name = os.path.basename(d)
        if not name.startswith(PREFIX_ALLOW):
            v['SK004'].append(f'{rel}  前缀不在白名单')

        # SK005 frontmatter
        fields = set(re.findall(r'^([A-Za-z_-]+)\s*:', fm, re.M))
        legacy = fields & LEGACY_FIELDS
        if legacy:
            v['SK005'].append(f'{sk}  含历史字段 {sorted(legacy)}')
        missing = {'name', 'description', 'version', 'license'} - fields
        if missing:
            v['SK005'].append(f'{sk}  缺必填字段 {sorted(missing)}')

        # SK006 寻址：只报「指向包根共享层却没加双态前缀」的引用。
        # skill 自己目录下的 references/scripts 是合法的，不能连带报——
        # 早期版本按裸路径计数，把 547 处合法自引用也算成了违规。
        # ops 卡豁免：卡里的 `scripts/opbench.py` 等按卡面约定指包根脚本，
        # 由 Plugins/ops/tests/test_doc_links.py 的 CARD_RELATIVE_DIRS 单独校验。
        if '/Plugins/ops/' in d.replace('\\', '/') + '/':
            pass
        elif not re.search(r'_dcu-toolkit-shared', body):
            bad = []
            for m in re.finditer(
                    r'(?<![\w/.])((?:references|scripts|data|docs)/[A-Za-z0-9_./-]+)', body):
                rel = m.group(1)
                if os.path.exists(os.path.join(d, rel)):
                    continue                      # 指向 skill 自身 → 合法
                if os.path.exists(os.path.join(root, rel)):
                    bad.append(rel)               # 指向包根却没加前缀 → 真违规
            if bad:
                v['SK006'].append(
                    f'{sk}  {len(bad)} 处指向包根却缺 ../_dcu-toolkit-shared/')

        _caller_body['v'] = body

        # SK007 脚本引用存在性。
        # 判据是「这个脚本在仓库里存在」——路径写得相对不准属于 SK006 的范畴，
        # 不在本条重复报。所以解析不到时回落到全仓按 basename 查找。
        for m in sorted(set(re.findall(
                r'((?:\.\./)*scripts/[A-Za-z0-9_./-]+\.(?:py|sh))', body))):
            base = os.path.basename(m)
            cands = [os.path.normpath(os.path.join(d, m)),
                     os.path.normpath(os.path.join(root, m)),
                     os.path.join(root, 'scripts', base)]
            if any(os.path.exists(c) for c in cands):
                continue
            if _find_by_basename(root, base) is not None:
                continue                      # 仓内存在，只是路径没写全 → 交给 SK006
            if _is_upstream_ref(m):
                continue                      # 明确指向上游仓（sglang 等）的脚本
            v['SK007'].append(f'{sk}  -> {m}（仓内不存在）')

        # SK009 description 语言
        m = re.search(r'^description\s*:\s*"?(.+)', fm, re.M)
        if m:
            desc = m.group(1)
            if LATIN.search(desc) and not CJK.search(desc):
                v['SK009'].append(f'{sk}  description 为纯英文')

    # ---------- 文件命名 SK003 ----------
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, 'Plugins')):
        dirnames[:] = [x for x in dirnames if x != '__pycache__']
        if 'kb_shard' in dirpath:
            continue
        for fn in filenames:
            if fn in CONVENTION_FILES or fn.startswith('.'):
                continue
            base, ext = os.path.splitext(fn)
            why = []
            if CJK.search(fn):
                why.append('含中文')
            if re.search(r'[A-Z]', fn):
                why.append('含大写')
            if ' ' in fn:
                why.append('含空格')
            if ext != '.py' and '_' in base:
                why.append('非py用下划线')
            if ext == '.py' and '-' in base:
                why.append('py用连字符')
            if why:
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                v['SK003'].append(f'{rel}  {"/".join(why)}')

    # ---------- SK008 md 死链 ----------
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [x for x in dirnames if x not in ('__pycache__', '.git')]
        if 'kb_shard' in dirpath:
            continue
        for fn in filenames:
            if not fn.endswith('.md'):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root)
            try:
                text = read(p)
            except OSError:
                continue
            n = 0
            for m in re.finditer(r'\]\(([^)]+)\)', text):
                tgt = m.group(1).strip().split('#')[0]
                if (not tgt or tgt.startswith(('http', 'mailto:', '/', '<'))
                        or ' ' in tgt):
                    continue
                if not os.path.exists(os.path.normpath(
                        os.path.join(dirpath, tgt))):
                    n += 1
            if n:
                v['SK008'].append(f'{rel}  {n} 条失效链接')

    # ---------- SK010 版本一致 ----------
    mp_path = os.path.join(root, '.claude-plugin', 'marketplace.json')
    if os.path.isfile(mp_path):
        try:
            mp = json.loads(read(mp_path))
            want = {p['name']: p.get('version') for p in mp.get('plugins', [])}
            for plugin, ver in sorted(want.items()):
                f = os.path.join(root, 'Plugins', plugin, 'SKILL.md')
                if not os.path.isfile(f):
                    continue
                # 只认「版本**：…」这种声明式写法；正文表格里「各版本 …（合并 0.15/…）」
                # 是产品版本不是本包版本，不能当声明抓（早先的宽匹配在这里误报过）。
                fm_v = re.search(r'^version\s*:\s*(\S+)', frontmatter(read(f)), re.M)
                body_v = re.search(r'\*\*版本\*\*[：:][^\n（(]*[（(](\d+\.[0-9.]+)', read(f))
                got = (fm_v.group(1) if fm_v else None) or \
                      (body_v.group(1) if body_v else None)
                if got is None:
                    continue
                if got != ver:
                    v['SK010'].append(
                        f'Plugins/{plugin}/SKILL.md  声明 {got}，marketplace 为 {ver}')
        except (ValueError, KeyError) as e:
            v['SK010'].append(f'.claude-plugin/marketplace.json 解析失败: {e}')

    stat['skeletons'] = len(stat['skeletons'])
    return v, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--show', default=None, help='只打印某条规则明细，如 SK006')
    ap.add_argument('--limit', type=int, default=15)
    ap.add_argument('--baseline', default=None)
    ap.add_argument('--out', default='lint-report.json')
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    v, stat = lint(root)

    print(f'== skill_lint @ {root}')
    print(f'skills={stat["skills"]}  目录骨架种类={stat["skeletons"]}  '
          f'形态分布={stat["kinds"]}')
    print()
    total = 0
    for k in sorted(RULES):
        n = len(v[k])
        total += n
        print(f'{k}  {n:5d}  {RULES[k]}')

    base = None
    if a.baseline and os.path.isfile(a.baseline):
        base = json.loads(read(a.baseline))
        new = {}
        for k in v:
            old = set(base.get(k, []))
            fresh = [x for x in v[k] if x not in old]
            if fresh:
                new[k] = fresh
        n_new = sum(len(x) for x in new.values())
        print(f'\n对照 baseline：新增违规 {n_new} 条')
        for k in sorted(new):
            print(f'  {k} +{len(new[k])}')
            for x in new[k][:a.limit]:
                print(f'    {x}')
        if n_new:
            print('\nRESULT: FAIL（新增违规）')
            return 1
        print('\nRESULT: PASS（无新增）')
        return 0

    if a.show:
        print(f'\n-- {a.show} 明细（最多 {a.limit} 条）--')
        for x in v.get(a.show, [])[:a.limit]:
            print('  ' + x)

    with open(os.path.join(root, a.out), 'w', encoding='utf-8') as f:
        json.dump(v, f, ensure_ascii=False, indent=1)
    print(f'\n总违规 {total} 条，明细写入 {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
