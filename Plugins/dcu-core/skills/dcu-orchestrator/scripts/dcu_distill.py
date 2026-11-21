#!/usr/bin/env python3
"""dcu_distill.py — 把一次算子优化 run 的产物转成用户知识库条目（确定性变换，不编数字）。

用法：
  python3 dcu_distill.py --run runs/op-optimize/validate/result.json \
      [--kb ~/.config/dcu-toolkit/kb] [--op <算子名>]

产出：
  1) <kb>/op-opt/<op>-<yyyyMMdd>.md   台账条目（数字仅来自 result.json，缺字段写 MISSING）
  2) <kb>/op-opt/promotion-<op>-<date>.md  晋升提案骨架（拟并入 skill 文件 + 不并入项）
  3) <kb>/op-opt/<op>-<yyyyMMdd>.json 机器可读证据（fingerprint 任一字段 MISSING 即 status=unverified，
     verify_evidence.py 据此拒绝放行）
  同名旧产物写入前先改名为 <file>.bak(.N) 保留，不静默覆盖。

原则：任何数字都必须能回到 result.json；本脚本不推断、不补全、不美化。
"""
import argparse
import json
import os
import re
import sys
import datetime

TEMPLATE = """# 算子优化台账：{op}

> 来源：{run}（本条目数字全部来自该产物；缺失字段标记 MISSING）
> 生成：dcu_distill.py（确定性变换，未做任何推断）
> 格式：五段骨架 `结论 / 前提 / 已证伪 / 复现命令 / 口径`（口径定义见 kbq.py）。
> 生成方只填能从产物确定性推出的部分；**推不出的段留空占位，不编造**。
> 五段是否齐全由 `scripts/kbq.py index` 检查，缺段会在检索输出里标红。

## 结论

- 本次改动：{change}
- 收益：{gain}（口径见下节；`[历史数据，随硬件/DTK/版本变化]`）
- 正确性复核：{accuracy}

## 前提

<!-- 结论在什么条件下成立。设备/版本/形状/并发任一改变都可能反转，必须写。 -->

- 设备：{device} / gfx {gfx} / DTK {dtk} / 框架 {framework}
- 形状与并发：（待补：shape-id + batch + 频率档）
- 其它前提：（待补）

## 已证伪

<!-- 本轮排除了哪些方向/写法。这是最有价值的部分：防止下一个人重走。 -->

- （待补：试过但不成立的方向 + 为什么）

## 复现命令

```bash
# 证据路径见下方「证据路径」；本条目数字全部来自该 run 产物
python3 scripts/opbench.py ...   # （待补：复现本次 A/B 的确切命令）
```

## 口径

<!-- 同一个数换个分母/频率档/标称峰值，结论可能反转。不写清楚就无法复用。 -->

| 项 | 值 |
|---|---|
| 计时方法 | （待补：如 opbench 多轮中位数 + MAD） |
| 频率档 | （待补：实测 sclk，非标称峰值） |
| 分母/读量口径 | （待补：如「模型最少字节」而非实测 DRAM） |
| 指标定义 | {metric_name} = {before} → {after} |
| 就业环境指纹 | 见下方「环境指纹」 |

## 环境指纹（source / verified_on 纪律）

| 项 | 值 |
|---|---|
| 设备型号 | {device} |
| gfx | {gfx} |
| DTK | {dtk} |
| 框架 | {framework} |
| 镜像 | {image} |
| 模型 | {model} |
| 日期 | {date} |

## 数字（同条件 A/B，历史数据，随硬件/DTK/版本变化）

| 指标 | 优化前 | 优化后 | 收益 |
|---|---|---|---|
| {metric_name} | {before} | {after} | {gain} |

- 备注：{note}

## 证据路径

- run 产物：`{run}`
- 原始 profiling / 日志：{logs}

## 复用性判定（写清楚，别含糊）

- 本次改动：{change}
- 可跨项目复用的事实（候选）：见下方「候选事实」
- 明确环境专属（不并入 skill）：{env_specific}

## 候选事实（人工复核后填 scope/status）

<!-- 每条一行：[ ] <一句话事实> | scope=<gfx号/卡型> | status=verified|unverified | 依据=<证据> -->

- [ ] （待填）

## 晋升提案

见 `{promotion}`
"""

PROPOSAL = """# 晋升提案：{op}

> 来源台账：`{ledger}`
> 目的：把其中「可跨项目复用」的事实并入厂商 skill。**需人工/厂商审核后才并入，不自动改写。**

## 拟并入

| 事实 | 目标 skill 文件 | 依据 | 状态 |
|---|---|---|---|
| （待填） | 如 `dcu-kernelwiki/wiki/<topic>.md` | {run} | 待复核 |

## 明确不并入

- 环境专属结论（设备/版本绑定）：
- 一次性参数（shape/并发/绑核等）：
- 未验证结论（status=unverified）：
- 纯性能数字（标 [历史数据]，只留台账与项目留痕）：

"""


def load_run(path):
    with open(path, encoding='utf-8-sig') as fh:  # utf-8-sig：容忍带 BOM 的 run 产物
        return json.load(fh)


def slugify_op(name):
    """op 名进文件名前消毒：剔除路径分隔/非法字符防目录逃逸，空结果回落 unnamed-op。"""
    s = re.sub(r'[^0-9A-Za-z._-]+', '_', str(name))[:80].strip('.')
    return s or 'unnamed-op'


def backup_if_exists(path):
    """同名旧产物先改名为 .bak（占用则 .bak.1、.bak.2…），对齐 install.sh 的 .local.bak 惯例。"""
    if not os.path.exists(path):
        return
    n, bak = 0, path + '.bak'
    while os.path.exists(bak):
        n += 1
        bak = '%s.%d' % (path + '.bak', n)
    os.rename(path, bak)


def pick(data, *keys, default='MISSING'):
    """在字典里按候选键名找值（兼容嵌套 fingerprint / 平铺两种写法）。"""
    pool = []
    if isinstance(data, dict):
        pool.append(data)
        for key in ('fingerprint', 'metrics', 'result', 'env', 'performance'):
            value = data.get(key)
            if isinstance(value, dict):
                pool.append(value)
    for key in keys:
        for src in pool:
            if key in src and src[key] not in (None, ''):
                return src[key]
    return default


def nested(data, *paths, default='MISSING'):
    """按嵌套路径取值，取第一个非空（如 ('baseline', 'median_ms')）。

    pick() 只翻一层，而 opverify 的 run schema 把基线放在 baseline.median_ms、
    优化后放在 metrics.median_ms —— 两个同名字段在不同父节点下，必须按路径取，
    否则会张冠李戴（before 取到 after 的值）。
    """
    for path in paths:
        cur = data
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                cur = None
                break
        if cur not in (None, ''):
            return cur
    return default


def main():
    ap = argparse.ArgumentParser(description='DCU 优化成果 → 用户知识库条目')
    ap.add_argument('--run', required=True, help='result.json 路径（唯一数字来源）')
    ap.add_argument('--kb', default=os.path.expanduser('~/.config/dcu-toolkit/kb'))
    ap.add_argument('--op', default='')
    args = ap.parse_args()

    if not os.path.isfile(args.run):
        print('[FAIL] 找不到 run 产物：%s（不猜数字，直接退出）' % args.run)
        return 2
    data = load_run(args.run)

    op = slugify_op(args.op or pick(data, 'op', 'operator', 'kernel', 'op_name', default='unnamed-op'))
    date = datetime.datetime.now().strftime('%Y%m%d')
    day = datetime.datetime.now().strftime('%Y-%m-%d')

    opt_dir = os.path.join(args.kb, 'op-opt')
    os.makedirs(opt_dir, exist_ok=True)
    ledger = os.path.join(opt_dir, '%s-%s.md' % (op, date))
    prop = os.path.join(opt_dir, 'promotion-%s-%s.md' % (op, date))

    vals = {
        'device': pick(data, 'device', 'device_model', 'card'),
        'gfx': pick(data, 'gfx', 'arch'),
        'dtk': pick(data, 'dtk', 'dtk_version'),
        'framework': pick(data, 'framework', 'engine'),
        'image': pick(data, 'image'),
        'model': pick(data, 'model'),
        'metric_name': pick(data, 'metric', default='latency_ms'),
        'before': nested(data, ('baseline', 'median_ms'), ('baseline', 'latency_ms'),
                         default=pick(data, 'baseline_metric', 'baseline_ms',
                                      'latency_before', 'before')),
        'after': nested(data, ('metrics', 'median_ms'), ('metrics', 'latency_ms'),
                        default=pick(data, 'optimized_metric', 'measured_ms',
                                     'latency_after', 'after')),
        'gain': pick(data, 'gain', 'gain_percent', 'speedup'),
        'accuracy': pick(data, 'accuracy', 'correctness'),
        'note': pick(data, 'note', 'notes', default='（待补）'),
        'logs': pick(data, 'logs', 'profile', 'trace', default='（待补：profiling/日志路径）'),
        'change': pick(data, 'change', 'change_path', 'diff_summary', default='（待补：改动文件/开关）'),
        'env_specific': pick(data, 'env_specific', default='（待补）'),
    }
    # 指纹完整性：evidence.fingerprint 任一字段 MISSING → status=unverified（联动 verify_evidence.py 拦截）
    fp_missing = [k for k in ('device', 'dtk', 'gfx', 'framework', 'gain') if vals[k] == 'MISSING']

    text = TEMPLATE.format(op=op, run=args.run, date=day, promotion=prop, **vals)
    backup_if_exists(ledger)
    with open(ledger, 'w', encoding='utf-8') as fh:
        fh.write(text)
    backup_if_exists(prop)
    with open(prop, 'w', encoding='utf-8') as fh:
        fh.write(PROPOSAL.format(op=op, ledger=ledger, run=args.run))

    # 机器可读证据（verify_evidence.py 可直接校验；跨会话/跨设备可复现）
    evidence = {
        'path': ledger,
        'fingerprint': {
                'device': vals['device'], 'dtk': vals['dtk'], 'gfx': vals['gfx'],
                'framework': vals['framework'], 'op': op, 'gain': vals['gain'],
                'source_run': args.run,
                'status': 'unverified' if fp_missing else 'verified',
        },
        'metric': {'name': vals['metric_name'], 'baseline': vals['before'],
                       'optimized': vals['after'], 'accuracy': vals['accuracy']},
    }
    ev_json = os.path.join(opt_dir, '%s-%s.json' % (op, date))
    backup_if_exists(ev_json)
    with open(ev_json, 'w', encoding='utf-8') as fh:
        json.dump(evidence, fh, ensure_ascii=False, indent=2)
    print('[OK] 证据 JSON：%s' % ev_json)

    print('[OK] 台账：%s' % ledger)
    print('[OK] 晋升提案：%s' % prop)
    missing = fp_missing + [k for k in ('before', 'after') if vals[k] == 'MISSING']
    if missing:
        print('[WARN] 这些字段在 run 产物里缺失，台账里标了 MISSING，请人工补齐或回上游修产物：%s' % ','.join(missing))

    return 0


if __name__ == '__main__':
    sys.exit(main())
