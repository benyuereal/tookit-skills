#!/usr/bin/env python3
"""算子优化循环的驱动器：队列 → 单实验 → 机器判定 → 落盘记账。

这不是一个「自动调优器」。候选假设仍然要人/模型来想；本脚本负责的是
**不让「我觉得快了」蒙混过关**，以及**不重复探索已经证伪的方向**。

    python3 loop.py -C <算子卡目录> status
    python3 loop.py -C <算子卡目录> queue
    python3 loop.py -C <算子卡目录> run maxn32 [--skv 32768 131072]
    python3 loop.py -C <算子卡目录> report
    python3 loop.py -C <算子卡目录> impl-broken <tag> --why "锚点只改到一个 body，算错"
    python3 loop.py -C <算子卡目录> direction-refuted <tag> --why "机制：合并 buffer ⇒ 生命周期重叠 ⇒ 溢出"
    python3 loop.py -C <算子卡目录> direction-refuted --direction fuse-kv-splits --why "结构上不可能"
    python3 loop.py -C <算子卡目录> verify --tag <tag>

完成判定由**外部命令**裁决，不由模型口述：
  候选的 `done_when`（一条 shell 命令）返回 0 ⇒ `verify` 才把 status 改成 adopted；
  非 0 ⇒ 保持原状态，打印输出尾部 20 行并记 `done_when_rc`。
  `adopt` 是 `verify` 的旧名（等价），保留只为兼容脚本——它同样要跑 done_when，
  不存在「不跑命令就 adopted」的路径。

失败分两类，**不要混**（混了就等于白丢一条路）：
  impl-broken        这次实现没写成 / 没跑对（编译不过、锚点没落上、算错）。
                     留在 optimize 修；**不消耗方向预算**；连挂 3 次该实现判死。
  direction-refuted  跑通了，但结论是「这条攻击面不行」，且 why 写清了会连带杀掉哪一族。
                     回 op-constrain 重定约束；同一方向证伪 2 次 ⇒ 封方向，
                     `run` 拒绝再受理该方向的任何候选。

卡片需要提供（其余由本脚本负责）：
  candidates.yaml      方向（directions）+ 候选队列（candidates）+ 已判死清单（refuted）
                       （见 op-optimize/SKILL.md §三 的 schema）
  bench/ab.py          卡**自带**的 A/B 适配器（可选）：同进程交替 A/B，支持 --json，输出见
                       ADAPTER_AB。卡片没提供时 `run` 明确拒绝（fail-closed），`status`/`queue` 照常
  bench/acc_probe.py   卡自带的精度适配器（可选；没有就跳过精度门）
  contract.yaml        精度容差（可选；没有就用 --tol 默认值）

状态全部落在 <卡片>/runs/loop/（已被仓库 .gitignore 的 `**/runs/` 覆盖），
所以上下文丢了不会丢进度：`status` 一跑就能接着干。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ADAPTER_AB = """bench/ab.py --json 必须输出：
{"shape": {...},
 "runs": [{"skv": 32768, "base_ms": 0.78, "base_mad": 0.004, "fm_ms": 0.40,
           "variants": [{"tag": "maxn32", "ms": 0.69, "mad": 0.003,
                         "delta_pct": -11.7, "maxdiff": 1.9e-3, "rel": 4e-5}]}]}
（base 必须在每轮被重测；只报 base 一次的实现会把漂移算成收益）"""

ADAPTER_ACC = """bench/acc_probe.py --json 必须输出：
{"qscale": {"1": {"ref_absmax": 3.1, "rows": [{"tag": "base", "max_abs": 1.0e-3, "nonfinite": 0},
                                             {"tag": "maxn32", "max_abs": 1.1e-3, "nonfinite": 0}]}}}
"""

OK, WARN = "\033[32m", "\033[33m"

# 两个上限是**常量不是开关**：能拿参数绕过去的规则等于没规则。
# 精度主口径的 ULP 门。默认 2：bf16 完美舍入是 1 ULP，2 留给「eps 位置不同」
# 这类合法实现差异。厂商 lightop 实测正好 2.00 ULP —— 门限贴着物理极限。
MAX_ULP = 2.0

MAX_IMPL_FAILS = 3          # 同一实现连挂 N 次 ⇒ 这个实现判死（方向预算不动）
MAX_DIRECTION_REFUTES = 2   # 同一方向**机制上**被证伪 N 次 ⇒ 封方向（stalled 不计）

# 「不赚」分两类，落点不同（2026-11-23 从 CompoundingKernel 引入）：
#   refuted  机制上不成立（例：双流叠两个 GEMM 同引擎，物理不可能）⇒ **计方向预算**，可以封
#   stalled  变慢了但**结构变了**，值得留作父代 ⇒ **不计方向预算**，不封，只是不采纳
# 依据：论文 Quant/030 的最优路径 270 步里有 76 步持平或更慢，
# 最终仍从 6.25× 涨到 27.36× ——最好的结果不是从一直变快的路径长出来的。
# 所以「变慢」不能直接读成「这条路没戏」。

# verify 的口径（同为常量，不给开关去绕）：
VERIFY_TIMEOUT_S = 3600     # done_when 是外部命令，最长给 1 小时（跑真 bench 也够）
VERIFY_TAIL_LINES = 20      # 失败时只回显输出尾部 20 行，别把整份日志灌进上下文

# 仓库根。engine/loop.py 在 Plugins/ops/engine/ 下，向上 3 层是仓库根；
# 原 skills/op-optimize/loop.py 在 Plugins/ops/skills/op-optimize/ 下，需要 4 层。
# 用「找到 .git 为止」的方式，避免这种硬编码层级再出错。
# memory 写入开关。**测试必须把它关掉** —— 否则每次跑 pytest 都会往真实
# memory/ledger.jsonl 里灌临时卡记录（实测一次灌了 147 条 tmpXXXX）。
# 这个坑很隐蔽：流水看起来"有数据"，但全是测试假数据，检索会被污染。
MEMORY_WRITE = True

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()),
                 Path(__file__).resolve().parents[3])

# engine/ 下的兄弟模块（card / floor / probe / propose / ledger）。
_ENGINE_DIR = Path(__file__).resolve().parent


def _engine(name):
    """按路径加载 engine/ 下的兄弟模块；**文件不存在时返回 None**。

    loop.py 有三种被加载方式，`__package__` 三种都不是 `engine`：
      ① `python3 -I engine/loop.py`            —— __name__ == "__main__"，无包
      ② 测试 `spec_from_file_location("op_loop_under_test", ...)` —— 合成模块名
      ③ shim `runpy.run_path(..., run_name="__main__")` —— 同 ①
    而 `-I` 又保证脚本目录不在 sys.path 上，所以裸 `import card` 同样落空。
    仓库里已有三处（_load_measure_mode / _load_forge / _load_candidates_via_loop）
    各自写了一遍这个动作 —— 那是上一轮为了不改加载方式付出的代价，
    这里不再增加第四种写法：**统一成这一个函数**，改名也只改这里。

    先挂 sys.modules 再 exec 是必须的：模块之间会互相加载（floor 要用 card，
    cli 要用全部），不先挂就会把同一个文件执行两遍，模块级缓存（例如
    `_bound_map()` 的表）跟着分裂成两份。

    返回 None 而不是抛异常：仓库里确实存在**只安装 engine/loop.py 的布局**
    （见 _FallbackCard 的说明），那种布局下兄弟模块缺失是"部署形态"，
    不是程序错误 —— 但也不能静默降级，调用方必须显式处理。
    """
    key = "_ops_engine_" + name
    if key in sys.modules:
        return sys.modules[key]
    path = _ENGINE_DIR / (name + ".py")
    if not path.is_file():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


class _FallbackCard:
    """`engine/card.py` 缺失时的最小读卡器 —— **只用于单文件部署**。

    为什么需要它（这不是"再写一份读卡实现"的借口，是一个真实的部署约束）：
    仓库里存在只安装 `engine/loop.py` 的布局 —— `scripts/opgate_checks.py` 的
    方向预算门就是 `python3 -I <装好的>/engine/loop.py -C <卡> queue`，而
    `tests/test_ops_gate.py` 的夹具按这个布局**只拷 loop.py**（它模拟的是
    插件安装后的树）。这两处都不在本轮允许改动的范围内，而让门禁崩掉等于
    让所有 commit 被拦。

    所以 loop.py 必须能单文件跑。口径上它**不是第二套判据**：读的文件、
    归一化规则（裸 list → dict、五段缺省）、报错文案都与 card.py 一致，
    少的只有 `runs/advance/overlay.yaml` 合并 —— 单文件部署里没有 advance，
    也就不存在 overlay。

    它不会悄悄生效：构造时往 stderr 打一行告警。缺少这一行，两个实现哪天
    分叉了没有任何人会发现。
    """

    #: `config` 段缺失时的空容器表（与 card.py::_SECTIONS 同值）
    SECTIONS = (("defaults", dict), ("directions", list), ("candidates", list),
                ("refuted", list), ("stalled", list))

    def __init__(self):
        print("[engine] 警告：engine/card.py 不存在，读卡降级为 loop.py 内置实现"
              "（单文件部署；overlay 不参与合并）", file=sys.stderr)

    def read_yaml(self, path, default=None):
        try:
            import yaml
            d = yaml.safe_load(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, ValueError, ImportError):
            return {} if default is None else default
        if d is None:
            return {} if default is None else default
        return d

    def load_candidates(self, card):
        p = Path(str(card)) / "candidates.yaml"
        d = self.read_yaml(p, {}) if p.exists() else {}
        if isinstance(d, list):
            d = {"candidates": d}
        if not isinstance(d, dict):
            raise SystemExit(
                "candidates.yaml 必须是 dict 或 list，实际是 %s：%s"
                % (type(d).__name__, p))
        for key, ctor in self.SECTIONS:
            d.setdefault(key, ctor())
        return d

    def load_contract_tol(self, card, default=2e-3):
        p = Path(str(card)) / "contract.yaml"
        if not p.exists():
            return default
        c = self.read_yaml(p, {})
        for path in (("tolerance", "vs_fp32_ref", "max_abs"), ("tolerance", "max_abs")):
            node = c
            for k in path:
                node = node.get(k) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, (int, float)):
                return float(node)
        return default

    def load_roofline(self, card):
        p = Path(str(card)) / "roofline.json"
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return d if isinstance(d, dict) else {}

    def load_variants(self, card):
        d = self.read_yaml(Path(str(card)) / "variants.yaml", {})
        vs = d.get("variants") if isinstance(d, dict) else d
        return [v for v in (vs or []) if isinstance(v, dict)]

    def load_directions(self, card):
        return [d for d in (self.load_candidates(card).get("directions") or [])
                if isinstance(d, dict)]


# 读卡的唯一实现搬到了 engine/card.py：`candidates.yaml` 的两种顶层格式
# （dict / 裸 list）与 runs/advance/overlay.yaml 合并都在那里。这里 re-export
# 是**对外契约**——测试、shim、别的插件都直接点名 `loop.load_candidates`。
# 别名用 `_CARD` 而不是 `card`：本文件里 `card` 是一个用了上百次的**参数名**
# （卡片目录），同名会把这个模块对象整个遮住，而这类错只在被调用到的那条分支炸。
_CARD = _engine("card")
if _CARD is None:                       # 单文件部署（见 _FallbackCard）
    _CARD = _FallbackCard()


def _load_measure_mode():
    """加载 scripts/measure_mode.py —— measure_mode 的**唯一真源**。

    用 importlib 按路径加载，而不是 `import measure_mode`：loop.py 平时以
    `python3 -I .../loop.py` 运行，-I 会把脚本目录（不是仓库根）放进 sys.path，
    直接 import 会落空。加载失败时（例如被人单独拷走）返回 None，
    调用方把「没有真源」判成不可主张，而不是放行。
    """
    import importlib.util
    p = REPO_ROOT / "scripts" / "measure_mode.py"
    try:
        spec = importlib.util.spec_from_file_location("_op_measure_mode", str(p))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_op_measure_mode"] = mod
        spec.loader.exec_module(mod)
        return mod
    except (OSError, AttributeError, ImportError):
        return None


measure_mode = _load_measure_mode()


def _c(s, color):
    return s if not sys.stdout.isatty() else color + s + "\033[0m"


# ---------------------------------------------------------------- 卡片读取

# `load_candidates` 的实现在 engine/card.py（裸 list 卡片的兼容、overlay 合并
# 都在那里，见该模块说明）。这里只保留名字：正文、shim、测试都直接 import 它。
load_candidates = _CARD.load_candidates


def direction_table(cands):
    """方向 → {refutes, status, exhausted}。计数只有这一个权威点（本段 yaml）。"""
    out = {}
    for d in cands.get("directions") or []:
        if not isinstance(d, dict) or not d.get("id"):
            continue
        refutes = int(d.get("refutes") or 0)
        status = str(d.get("status") or "open")
        out[str(d["id"])] = {
            "id": str(d["id"]),
            "statement": str(d.get("statement") or ""),
            "refutes": refutes,
            "status": status,
            "exhausted": status == "exhausted" or refutes >= MAX_DIRECTION_REFUTES,
        }
    return out


def budget_problems(cands, tags=None):
    """方向预算的前置条件。返回问题清单，空 = 预算可计、没封。

    三条其实是同一件事：**方向预算对这个候选不可计，或它属于已封的方向**
    ⇒ 2 次上限保护不到它 ⇒ 拒绝跑。宁可拦住，也不许「规则对某个候选不生效」。
    """
    table = direction_table(cands)
    probs = []
    for c in cands.get("candidates") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "?")
        if tags is not None and cid not in tags:
            continue
        if str(c.get("status") or "pending") != "pending":
            continue
        if len(c.get("fails") or []) >= MAX_IMPL_FAILS:
            probs.append("候选 %s 的实现已连挂 %d 次（判死线），它该在 refuted 里"
                         % (cid, MAX_IMPL_FAILS))
            continue
        d = c.get("direction")
        if not d:
            probs.append("候选 %s 没声明 direction：方向预算计不了（%d 次上限对它是空转）"
                         % (cid, MAX_DIRECTION_REFUTES))
            continue
        info = table.get(str(d))
        if info is None:
            probs.append("候选 %s 的 direction=%s 不在 directions 段里（拼错？预算同样计不了）"
                         % (cid, d))
        elif info["exhausted"]:
            probs.append("候选 %s 属于方向 %s，该方向已证伪 %d 次（%s）⇒ 回 op-constrain "
                         "重定约束，别再在这个方向上生成候选"
                         % (cid, d, info["refutes"], info["status"]))
    return probs


BOUND_TO_CLASS = None


def _bound_map():
    """读 bound-to-class.yaml（derive 的 bound → 可试的攻击面 class）。"""
    global BOUND_TO_CLASS
    if BOUND_TO_CLASS is not None:
        return BOUND_TO_CLASS
    # references/ 是**知识**，留在 skills/op-optimize/ 下（见 docs/ENGINE.md 模块表）；
    # engine 只读它。原先写 Path(__file__).parent 是因为实现和知识同目录，
    # 迁移后 engine 在 Plugins/ops/engine/，必须显式跨目录。
    p = Path(__file__).resolve().parents[1] / "skills" / "op-optimize" / "references" / "bound-to-class.yaml"
    # 走 card.read_yaml：读 YAML 的容错口径（缺文件 / 坏 YAML / 无 PyYAML）
    # 只该有一份。这张表是**知识**不是卡，但从「怎么读一个 yaml 文件」的角度
    # 它与读卡没有区别，所以复用同一个入口而不是再写一个 try/except。
    try:
        d = _CARD.read_yaml(p, {})
        BOUND_TO_CLASS = d.get("bound_to_class") or {}
    except Exception:                                            # noqa: BLE001
        BOUND_TO_CLASS = {}
    return BOUND_TO_CLASS


def _class_of(direction):
    """direction(surface id) → class。来自 class_members 的反查表。"""
    # references/ 是**知识**，留在 skills/op-optimize/ 下（见 docs/ENGINE.md 模块表）；
    # engine 只读它。原先写 Path(__file__).parent 是因为实现和知识同目录，
    # 迁移后 engine 在 Plugins/ops/engine/，必须显式跨目录。
    p = Path(__file__).resolve().parents[1] / "skills" / "op-optimize" / "references" / "bound-to-class.yaml"
    try:
        d = _CARD.read_yaml(p, {})
    except Exception:                                            # noqa: BLE001
        return None
    for cls, ids in (d.get("class_members") or {}).items():
        if direction in (ids or []):
            return cls
    return None


def bottleneck_problems(cands, tags=None, card=None):
    """第 3 步（定位瓶颈）的机检：没有 bottleneck 记录就不许跑候选。

    2026-11-23 实测：`grep -nE "derive|bound|bottleneck" loop.py` 零结果。
    后果是 agent 可以完全不跑 derive、不知道卡在算力还是内存，
    直接列候选改代码——正是 SKILL.md 自己批评的「凭感觉说快了」。

    两条判据：
      1. candidates.yaml 顶层必须有 bottleneck 块，且含 bound 与 derive_evidence
      2. 每个 pending 候选的 direction 必须属于 bottleneck.bound 允许的 class
    """
    probs = []
    bn = cands.get("bottleneck")
    # 真值优先级：<卡>/roofline.json（derive.py 的正式产物）> candidates.yaml 的
    # bottleneck 块（手工补充）。两份都在时以 roofline.json 为准，
    # 因为它是脚本产出的，而 bottleneck 块是人写的。
    roof = None
    p_roof = (card / "roofline.json") if card else None
    if p_roof is not None and p_roof.is_file():
        # 读 roofline.json 的口径（缺文件 / 坏 JSON ⇒ {}）只在 card 里有一份：
        # 三个调用点各自 json.loads 一遍，就会出现「这里当空、那里当异常」。
        roof = _CARD.load_roofline(card) or None
    if not isinstance(bn, dict) and not roof:
        probs.append("缺 bottleneck 记录：先做第 3 步（derive.py 出 bound + "
                     "按 surfaces 的 check 取证），再列候选")
        return probs
    if roof:
        bound = str(roof.get("bound") or "")
        ev = str(roof.get("evidence") or roof.get("derive_evidence") or
                 "roofline.json 的 bound（derive.py 产出）").strip()
    else:
        bound = str(bn.get("bound") or "")
        ev = str(bn.get("derive_evidence") or "").strip()
    # ---- P0-3：remaining 符号门 + 同口径校验
    # floor.py 有三处明写「负 remaining = 判据本身错了（上限估低或流量算多）」，
    # 而 R0 的 roofline.json 里 remaining=-0.0404 照样放行进了 EV。
    # 判据自己定义了「这是错的」，消费方却不查——那就等于没定义。
    r = roof.get("remaining") if roof else None
    # **两种负值必须分开**（2026-11-25）：
    #   A. t_actual 远低于 t_floor（|r| 大）：上限估低或流量算多，判据错了 → 拒跑
    #   B. t_actual 贴着 t_floor（|r| 很小）：地板估得略紧，**这是正常的**，
    #      说明该实现已经在地板附近。当错误处理会让「已到顶」的卡永久停摆。
    #      实测 rmsnorm：lightop 197.0 vs floor 197.3，r=-0.0015。
    #      第一个当错的实现是 R0 的 -0.0404（-4%），那属于 A。
    # 阈值 2%：归约折扣本身的观测区间就有 ±5%，2% 以内的负值不构成判据错误的证据。
    if isinstance(r, (int, float)) and r < -0.02:
        probs.append("remaining=%.4f < -2%% —— floor.py 已把负值定义为「判据本身错了」"
                     "（上限估低或流量算多），不是实现突破物理。先回去查分母，别进决策。"
                     % r)
    if roof:
        m_actual = roof.get("t_actual_mode")
        m_floor = roof.get("t_floor_mode")
        if m_actual and m_floor and m_actual != m_floor:
            probs.append("t_actual 与 t_floor 口径不同（%s vs %s）—— 两个不同口径的量"
                         "相减没有意义，remaining 也不是「剩余空间」"
                         % (m_actual, m_floor))

    if not bound:
        probs.append("bottleneck 缺 bound（应为 compute/memory/mixed，来自 derive.py）")
    if roof is None and not ev:
        probs.append("bottleneck 缺 derive_evidence —— 「卡在 X」必须有 derive 输出或 "
                     "PMC 读数做证据，不许只写结论")
    allowed = set((_bound_map().get(bound) or {}).get("classes") or [])
    if bound and not allowed:
        probs.append("bound=%r 在 bound-to-class.yaml 里没有映射" % bound)
        return probs
    for c in cands.get("candidates") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "?")
        if tags is not None and cid not in tags:
            continue
        if str(c.get("status") or "pending") != "pending":
            continue
        d = c.get("direction")
        if not d:
            continue
        cls = _class_of(str(d))
        if cls is None:
            # surfaces.yaml 是**标准攻击面库**，但真实卡可以有更具体的子方向名
            # （如 op-flash-mla 的 bypass-tilelang-codegen）。
            # 只要该卡声明了 parent_surface 就按它归 class；没有则放行但要求显式说明。
            ps = str(c.get("parent_surface") or "")
            if ps:
                cls = _class_of(ps)
            if cls is None:
                # 无 parent_surface 且不在 surfaces 里：不拦，但要求候选自己写 class
                decl = str(c.get("class") or "")
                if decl and decl in allowed:
                    continue
                if decl:
                    probs.append(
                        "候选 %s 的 class=%r 与 bottleneck.bound=%s 不匹配（允许 %s）"
                        % (cid, decl, bound, sorted(allowed)))
                else:
                    probs.append(
                        "候选 %s 的 direction=%s 不在 surfaces.yaml 里，且没写 "
                        "parent_surface 或 class —— 无法判断它属于哪类攻击面"
                        % (cid, d))
                continue
        if cls not in allowed:
            probs.append(
                "候选 %s 的 direction=%s（class=%s）与 bottleneck.bound=%s 不匹配："
                "该 bound 只允许 %s"
                % (cid, d, cls, bound, sorted(allowed)))
    return probs


def done_when_problems(cands, tags=None):
    """`status: adopted` 的候选必须带 `done_when`（一条 shell 命令）。

    理由：adopted 是「做完了」的断言，而**模型说做完不算数**——它必须能被一个
    单独的 evaluator 跑出来（`loop.py verify --tag <id>`）。没有命令的 adopted
    等于没有参照物的自我批准，所以 `run` 拒绝受理它。

    只看 adopted；pending 候选还没有完成判定，不要求（`verify` 会另拦）。
    """
    probs = []
    for c in cands.get("candidates") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "?")
        if tags is not None and cid not in tags:
            continue
        if str(c.get("status") or "pending") != "adopted":
            continue
        if not str(c.get("done_when") or "").strip():
            probs.append("候选 %s 已是 adopted 但没写 done_when：完成判定没有外部裁决点"
                         "（模型说完成不算数）⇒ 补一条命令，再用 verify 重新裁决" % cid)
    return probs


def stalled_parent_problems(cands):
    """stalled 条目必须真被当过父代 —— 否则「留作父代」只是换了个名字的判死。

    这是 CompoundingKernel 那条机制能否落地的**唯一验收点**：
    如果 stalled 里的条目从来没有后验候选引用它，
    那这套改动就只是把 refuted 改了个名，等于没做。
    判据：`candidates[].parent == <stalled 的 id>`，或记录里有 `parent=<id>`。
    """
    stalled = [r for r in (cands.get("stalled") or []) if isinstance(r, dict)]
    if not stalled:
        return []
    used = set()
    for c in cands.get("candidates") or []:
        if isinstance(c, dict) and c.get("parent"):
            used.add(str(c["parent"]))
    return ["stalled 条目 %s 从未被当过 parent（父代账本名存实亡）" % r.get("id")
            for r in stalled if str(r.get("id")) not in used]


def load_tolerance_json(card):
    """读 judge 冻结容差：<card>/judge/tolerance.json 或 $JUDGE_TOLERANCE。

    返回 dict（含 max_ulp/cosine/atol/evidence_kind/sha256）；没有就返回 {}。
    **完整性**：除 sha256 字段外规范化后重算，对不上直接拒绝 ——
    说明容差在 run 期间被改过（A4）。
    """
    import hashlib
    p = Path(str(card)) / "judge" / "tolerance.json"
    if not p.is_file():
        # P2：judge 产物落在 .artifacts/<arm>/<op>_<id>/judge/
        p = artifact_dir(Path(str(card))) / "judge" / "tolerance.json"
    if not p.is_file():
        env = os.environ.get("JUDGE_TOLERANCE")
        if env and Path(env).is_file():
            p = Path(env)
        else:
            return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(d, dict):
        return {}
    want = d.get("sha256")
    if want:
        body = {k: v for k, v in d.items() if k != "sha256"}
        blob = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        got = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        if got != want:
            raise ValueError("tolerance.json 的 sha256 对不上（期望 %s，实际 %s）"
                             "—— 容差在 run 期间被改过" % (want[:16], got[:16]))
    return d


def load_contract_tol(card, default=2e-3):
    """读 contract.yaml 的精度容差，缺就退回 default。

    实现搬到 engine/card.py（容差的键路径只该有一处知道）；名字留在这里是因为
    `cmd_run` 与测试都直接点名它。签名与缺省值必须与旧实现完全一致。
    """
    return _CARD.load_contract_tol(card, default)


def artifact_dir(card):
    """运行时数据的唯一真源：`.artifacts/<arm>/<op>_<id>/`。

    P2：账本从卡目录（在 git 里）搬到 `.artifacts`（不在 git），
    一次 run 一个目录，卡目录不再写运行时数据。

    稳定性要求：同一 run 的多次 `loop.py` 调用必须落到**同一个**目录，
    否则 status/claim/next 看不到上一条账本。所以：
      arm    = $OPS_ARM    > cwd 里 `arm-<X>` > "local"
      run id = $OPS_RUN_ID > cwd 的目录名（C 臂里就是 attempt-N）> "default"
    `OPS_ARTIFACTS_LEGACY=1` 可退回旧的 `<card>/runs/loop`（迁移期兜底）。
    """
    if os.environ.get("OPS_ARTIFACTS_LEGACY") == "1":
        return card / "runs"
    arm = os.environ.get("OPS_ARM")
    run = os.environ.get("OPS_RUN_ID")
    if not arm or not run:
        cwd = Path.cwd()
        m = re.search(r"arm-([A-Za-z0-9]+)", str(cwd))
        arm = arm or (m.group(1) if m else "local")
        run = run or (cwd.name or "default")
        # 测试/临时目录（tmpXXXX）和仓库根不配当 run id：否则每跑一次就新建
        # 一个 .artifacts/local/<随机串>_...（lead 2026-11-28 遗留项 4）。
        try:
            if cwd.resolve() == REPO_ROOT.resolve() or run.startswith("tmp"):
                run = "local"
        except OSError:
            pass
    # **出仓的卡（单测/临时卡）保持旧路径**：`.artifacts` 锚在 REPO_ROOT，
    # 只对仓库内的真实卡有意义。单测用 tmpdir 建卡，若也搬过去，30 个既有
    # 断言（账本在 <card>/runs/loop）会全红（实测 2026-11-28）。
    try:
        inside = card.resolve().is_relative_to(REPO_ROOT.resolve())
    except (AttributeError, ValueError, OSError):
        inside = str(card).startswith(str(REPO_ROOT))
    if not inside:
        return card / "runs"
    root = Path(os.environ.get("OPS_ARTIFACTS") or (REPO_ROOT / ".artifacts"))
    return root / arm / ("%s_%s" % (card.name, run))


def db_dir(card):
    d = artifact_dir(card) / "runs" / "loop"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_records(card):
    out = []
    for f in sorted(db_dir(card).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def run_repeated(cmd, ab_json, card, reps, env):
    """同一 job 跑 reps 次，返回首份 ab 并给每个变体附 repeat_ms / repeat_spread。

    R1：噪声底必须是**同一次测量**的重复 spread，不是跨轮比值的 max-min。
    失败返回 None（调用方负责退出码）。
    """
    ab_list = []
    for r in range(max(1, int(reps or 1))):
        out = ab_json if r == 0 else db_dir(card) / ("_ab_r%d.json" % r)
        cmd_r = [x if x != str(ab_json) else str(out) for x in cmd]
        rc = subprocess.call(cmd_r, env=env)
        if rc != 0 or not out.exists():
            print(_c("A/B 失败（rc=%d，第 %d/%d 次），没有记录" % (rc, r + 1, reps), WARN))
            return None
        ab_list.append(json.loads(out.read_text()))
    ab = ab_list[0]
    rep = {}
    for x in ab_list:
        for run in x.get("runs") or []:
            for v in run.get("variants") or []:
                if v.get("ms") is not None:
                    rep.setdefault(v["tag"], []).append(float(v["ms"]))
    for run in ab.get("runs") or []:
        base = run.get("base_ms")
        for v in run.get("variants") or []:
            ms = rep.get(v["tag"]) or []
            v["repeat_ms"] = ms
            v["repeat_spread"] = ((max(ms) - min(ms)) / base
                                  if len(ms) > 1 and base else None)
    return ab


def promote_base(card, best_tag, best_ms, rec):
    """把本轮最好的可接受实现晋升为下一轮基线。**必须 claim PASS**（P1）。"""
    if not best_tag:
        return
    if os.environ.get("OPS_NO_PROMOTE") == "1":
        print(_c("不晋升 %s：本轮没有 agent 改动（OPS_NO_PROMOTE=1）" % best_tag, WARN))
        return
    ok, reasons = claim_gate(rec)
    if not ok:
        print(_c("不晋升 %s：claim 未通过（%s）—— 先把这几项补齐再晋升"
                 % (best_tag, ", ".join(reasons)), WARN))
        return
    pp = db_dir(card) / "base.json"
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps({
        "base": best_tag, "ms": best_ms,
        "source_tag": best_tag,
        "evidence_kind": rec.get("evidence_kind"),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, ensure_ascii=False), encoding="utf-8")
    print("基线晋升：下一轮以 %s 为参照（%s；claim PASS）" % (best_tag, pp))


# 噪声记录（baseline MAD 超熔断线）**不是决策**，是「本轮没测出结论」。
# 实测代价（2026-11-24）：同一份代码在四张卡上，lightop 从 197µs 漂到 220~236µs，
# MAD 4.9~7.2% 全超 5% 熔断线。judge 正确地判了 INCONCLUSIVE，但记录照样进了
# 权威流水 —— D4 的轮次、calibrate 的 realized_med、recall 的检索都会把它算进去。
# 规则只有两条：**重跑，或者不写进权威流水**。这里选后者 + 隔离留证。
NOISE_DIR = "noise"


def is_noise_record(rec):
    """这条记录是不是「噪声过大、未形成决策」。"""
    if str(rec.get("decision")) != "INCONCLUSIVE":
        return False
    why = str(rec.get("why") or "")
    return "本轮不可信" in why and "噪声" not in why or "MAD/median" in why


ATLAS_CARDS = REPO_ROOT / "Plugins/ops/op-atlas/cards"
# 这些判定才允许写回 atlas 卡：都要求"这一轮测的是真实实现且口径可 claim"
_ATLAS_OK_DECISIONS = ("CALIBRATION_ONLY", "ACCEPTED_BITEXACT",
                       "ACCEPTED_NUMERICS", "ACCEPTED_NUMERICS_PENDING_E2E")


def sync_atlas_measurement(card, rec):
    """把一条实测写回 atlas 卡的 `floor_inputs`，返回 (是否写了, 说明)。

    没有这一根线，新算子就永远没有地板：`floor.py` 只要
    `bytes_mb + pattern + t_actual_us` 就能算，但 `t_source=derived` 会被它
    拒算，而"实测出来了"这件事只有这里知道。见模块说明。
    """
    if os.environ.get("OPS_NO_ATLAS_SYNC") in ("1", "true", "yes"):
        return False, "OPS_NO_ATLAS_SYNC=1"
    if "pytest" in sys.modules:
        return False, "pytest 环境不写真实卡"
    if str(rec.get("measure_mode")) != "event_isolated":
        return False, "口径 %r 不可 claim" % rec.get("measure_mode")
    if str(rec.get("decision")) not in _ATLAS_OK_DECISIONS:
        return False, "判定 %r 不写回" % rec.get("decision")

    ms = [v for v in (rec.get("base_ms"), rec.get("cand_ms"))
          if isinstance(v, (int, float)) and v > 0]
    if not ms:
        return False, "记录里没有可用的耗时"
    best_ms = min(ms)
    # 哪个实现跑出了这个数：cand 更快就是候选变体，否则是基线
    who = str(rec.get("tag"))
    if isinstance(rec.get("base_ms"), (int, float)) and rec["base_ms"] <= best_ms + 1e-12 \
            and best_ms != rec.get("cand_ms"):
        who = "base"

    cid = card.name[3:] if card.name.startswith("op-") else card.name
    f = ATLAS_CARDS / ("%s.yaml" % cid)
    if not f.is_file():
        return False, "没有 atlas 卡 %s（不新建，建卡是 op_bootstrap 的事）" % f.name

    import yaml
    d = _CARD.read_yaml(f, {})
    fi = d.get("floor_inputs")
    if not isinstance(fi, dict):
        return False, "atlas 卡缺 floor_inputs，不从这里补"
    old = fi.get("t_actual_us")
    if isinstance(old, (int, float)) and fi.get("t_source") not in ("derived", "missing") \
            and old <= best_ms * 1000.0 + 1e-9:
        return False, "已有更快且非反推的 t_actual=%.1fµs，不覆盖" % old

    try:
        rel = str(Path(rec.get("_record_path") or "").relative_to(REPO_ROOT))
    except (ValueError, TypeError):
        rel = "runs/loop/"
    fi["t_actual_us"] = round(best_ms * 1000.0, 1)
    fi["t_source"] = "results"
    fi["t_variant"] = who
    fi["t_measured_at"] = rec.get("ts")
    fi["t_measure_mode"] = "event_isolated"
    fi["evidence"] = rel
    prev = fi.get("t_previous_us")
    if isinstance(old, (int, float)) and old != fi["t_actual_us"]:
        fi["t_previous_us"] = (prev or [])[-3:] + [old]
    d["floor_inputs"] = fi
    f.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")
    return True, "atlas %s: t_actual_us=%.1f（%s，原 %s）" % (
        cid, fi["t_actual_us"], who, "无" if old is None else old)


def append_record(card, rec):
    """写流水。噪声记录进 `runs/loop/noise/`，不进权威流（`runs/loop/*.jsonl`）。

    隔离而不是丢弃：证据要留，但它不能参与 D4 轮次 / calibrate / recall 的统计。
    """
    d = db_dir(card)
    if is_noise_record(rec):
        d = d / NOISE_DIR
        d.mkdir(parents=True, exist_ok=True)
        rec = dict(rec, quarantined="baseline MAD 超熔断线，未形成决策（重跑或作废）")
    f = d / (time.strftime("%Y%m%d") + ".jsonl")
    with f.open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # P0：实测写回 atlas 卡，让 floor.py 对**任意**算子都能算出地板。
    # 放在落盘之后：先用日志留痕，再改卡；卡写失败也不能让这一轮白跑。
    try:
        _ok, _why = sync_atlas_measurement(card, dict(rec, _record_path=str(f)))
        if _ok:
            print("[atlas] " + _why, flush=True)
    except Exception as _e:                     # noqa: BLE001
        print(_c("[atlas] 写回失败（不影响本轮记录）：%s: %s"
                 % (type(_e).__name__, _e), WARN), flush=True)
    # **memory 索引**（ENGINE.md §二：memory/ 只由 engine 写）。
    # 只索引权威流水里的记录：噪声记录已被隔离，它们没形成决策，索引了会污染检索。
    # 字段 = runs/loop 那份原样 + card，两者可逐字段对账，不做二次加工。
    # 三条闸：① 全局开关（测试可关）② 在 pytest 下自动关（防漏配）
    # ③ 卡片目录必须在仓库内（临时卡绝不进跨卡索引）。
    # 实测教训：只加开关不够 —— 4 个测试文件都会调 append_record，
    # 漏一个就灌 43 条假记录进真实索引。
    _under_pytest = "pytest" in sys.modules
    _card_in_repo = REPO_ROOT in card.resolve().parents
    if MEMORY_WRITE and not _under_pytest and _card_in_repo \
            and not is_noise_record(rec):
        try:
            mf = REPO_ROOT / "memory" / "ledger.jsonl"
            mf.parent.mkdir(parents=True, exist_ok=True)
            with mf.open("a") as fh:
                fh.write(json.dumps(dict(rec, card=card.name), ensure_ascii=False) + "\n")
        except OSError:
            pass        # memory 是索引不是权威：写不进去不该让实验失败


# ---------------------------------------------------------------- 主张门（薄约束的核心）

# 口径五问的字段名以既有契约为准（op-bench/SKILL.md:65 与 bench_long.py:67
# 用的都是 "state"，不是 "thermal"）。改这里前先看那两处。
CALIBER_KEYS = ("what", "alloc", "state", "shape", "device")
# 兼容旧写法：adapter 若写 thermal，等同 state
_CALIBER_ALIASES = {"thermal": "state"}


def _caliber_get(cal, key):
    """读口径字段，容忍 thermal/state 两种写法与布尔值。"""
    if key in cal:
        return cal[key]
    for alias, canon in _CALIBER_ALIASES.items():
        if canon == key and alias in cal:
            return cal[alias]
    return None


def _record_caliber(rec):
    """把记录里的口径补齐成一个 dict。

    adapter 的写法不统一（有的写 caliber.device，多数把设备信息放在
    env_fingerprint 的 env 里）。这里做一次归一，避免因为位置不同就判
    「口径不全」——那会把合规记录误杀。
    """
    cal = dict(rec.get("caliber") or {})
    env = rec.get("env") or {}
    if _caliber_get(cal, "device") in (None, "", "unstated"):
        dev = env.get("gfx") or env.get("device")
        if dev and dev != "cpu":
            cal["device"] = dev
    if _caliber_get(cal, "shape") in (None, "", "unstated"):
        sh = rec.get("shape")
        if isinstance(sh, dict) and sh:
            cal["shape"] = ",".join("%s=%s" % (k, sh[k]) for k in sorted(sh))
        elif sh:
            cal["shape"] = str(sh)
    return cal


def claim_gate(rec):
    """这条记录能不能变成「交付结论」。返回 (claimable, [reasons])。

    **只在「主张结论」的入口调用**（report / verify / adopt / close / distill），
    run 永不拒绝。这是「薄约束」的全部含义：
        约束面 = 断言面，不是动作面。
    探索期任何实验都能跑（包括用 --no-acc 快速筛方向），
    只是它产出的记录不能升级成「快了 x%」。

    三条硬门各封一个谎言：
      caliber_incomplete  封「分母漂移」——本项目最贵的一类错
      not_significant     封「跑两次看着差不多」
      accuracy_missing    封「快了但算错了」
    """
    reasons = []
    cal = _record_caliber(rec)
    missing = [k for k in CALIBER_KEYS
               if _caliber_get(cal, k) in (None, "", "unstated")]
    if missing:
        reasons.append("caliber_incomplete")
    if not rec.get("significant"):
        reasons.append("not_significant")
    if not rec.get("acc_ok"):
        reasons.append("accuracy_missing")
    # 与 cmd_claim 的第五项同源：只有独立核验（第二实现/厂商库）才能声称
    # 数值正确；self-consistent-only 或缺证据一律不过。
    # 之前 claim_gate 只查三项，导致"evidence 不过也能晋升基线"（实测 2026-11-28）。
    if str(rec.get("evidence_kind") or "") not in ("independent_reference",
                                                   "vendor_library"):
        reasons.append("evidence_unverified")
    return (not reasons), reasons


def decide(base, cand, acc_base, acc_cand, tol, primary, stress_mult,
           record_mode=None):
    """返回 (decision, why)。

    decision ∈ ACCEPTED_* / REJECTED_* / INCONCLUSIVE。

    精度门分两层，**不能混**（这是本项目踩出来的坑）：
      - 主口径（primary qscale = 生产 logit 尺度）：硬门，max_abs 必须 ≤ contract 容差。
      - 压力档（更高 qscale，故意把 softmax 拉尖）：只判**相对退化的倍数**
        `cand_rel / base_rel`。拿绝对容差去判压力档是错的 —— base 自己在 qscale=10
        就有 1.55e-2，会把正确改动一起否掉。压力档只报警，报警就必须补端到端。
    """
    if "error" in cand:
        return "ERROR", cand["error"]

    warns = []
    if acc_cand:
        nf = sum(r.get("nonfinite", 0) for r in acc_cand.values())
        if nf:
            return "REJECTED_ACCURACY", "出现 nonfinite（%s）" % ", ".join(
                "qscale=%s:%d" % (q, r["nonfinite"]) for q, r in acc_cand.items() if r["nonfinite"])
        p = acc_cand.get(primary)
        if p:
            # **ULP 优先**：适配器报了 max_ulp 就按它判，而不是拿固定绝对容差。
            # bf16 只有 7 位尾数，ULP 随量级变（[0.5,1) 2^-8 / [1,2) 2^-7 /
            # [2,4) 2^-6）。同一个算子输出里 0.1 和 3.0 量级的元素精度差 32 倍，
            # 用统一 max_abs 必然顾此失彼 —— 实测四个变体（含厂商 lightop）
            # 在 max_abs=2e-3 下全 FAIL，而它们的真实误差是「正好 1 ULP」的完美舍入。
            if p.get("max_ulp") is not None and p["max_ulp"] > MAX_ULP:
                return "REJECTED_ACCURACY", (
                    "主口径 qscale=%s 对 fp32 参考 max_ulp=%.2f > %d ULP"
                    % (primary, p["max_ulp"], MAX_ULP))
            if p.get("max_ulp") is None and p.get("max_abs", 0.0) > tol:
                return "REJECTED_ACCURACY", (
                    "主口径 qscale=%s 对 fp32 参考 max_abs=%.2e > 容差 %.1e"
                    % (primary, p["max_abs"], tol))
        for q, r in acc_cand.items():
            if q == primary:
                continue
            b = (acc_base or {}).get(q)
            if b and b.get("rel"):
                k = r.get("rel", 0.0) / b["rel"]
                if k > stress_mult:
                    warns.append("qscale=%s 相对退化 %.1f×（%.2e vs base %.2e）"
                                 % (q, k, r.get("rel", 0.0), b["rel"]))

    b = {"median_ms": base["median_ms"], "mad_ms": base.get("mad_ms", 0.0)}
    c = {"median_ms": cand["ms"], "mad_ms": cand.get("mad", 0.0)}
    # 显著变慢必须走拒绝分支，不能进「接受」。
    # judge() 把 significant_regression 与 significant 分开返回就是为了这件事；
    # _significant 适配回 bool 时会把两者都算 True —— 所以这里必须先判回归。
    # judge(base_median, base_mad, cand_median, cand_mad) —— 参数顺序：
    # baseline 在前。第一版传反了，导致所有「变快」都被判成 significant_regression。
    # ---- P0-1 前置门：测量模式。**必须独立于 judge()**。
    # judge() 只判噪声（MAD），判不了偏差（bias）。实测：
    #   R0 背靠背   Δ/(3×MAD) = 79 倍  ← 符号是反的
    #   ledger 隔离 Δ/(3×MAD) = 1.01 倍 ← 符号是对的
    # 符号错的口径「显著性」高 78 倍，所以模式判断不能折进显著性。
    if measure_mode is not None and record_mode is not None:
        # 只有**明确声明了**模式才做这个门。record_mode=None 表示调用方
        # 没有测量模式信息（如单元测试直调 decide）——那种情况下
        # 按未知处理会把所有判定打成 INCONCLUSIVE，等于判据把功能关掉。
        # 真实路径（cmd_run）总会传 ab.get("measure_mode")，缺了就是 "unknown"。
        _m = str(record_mode)
        if not measure_mode.claimable(_m):
            return "INCONCLUSIVE", ("测量模式=%s 不可用于主张（可筛方向）—— "
                                    "缺 measure_mode 时结论的符号都不可信" % _m)
    _verdict, _why = judge(b["median_ms"], b["mad_ms"], c["median_ms"], c["mad_ms"])
    if _verdict == "significant_regression":
        return "REJECTED_NO_GAIN", _why
    if _verdict == "noise":
        return "INCONCLUSIVE", _why
    sig, ratio, why = _significant(b, c)
    tail = ("；caveat " + "；".join(warns) + " ⇒ 压力档有退化，端到端必须验") if warns else ""
    if not sig:
        if ratio is not None and ratio < 1.0:
            return "INCONCLUSIVE", why + "（方向对但没过噪声门，别当收益写）"
        return "REJECTED_NO_GAIN", why

    maxdiff = cand.get("maxdiff")
    if maxdiff is None:
        # ab.py 不产精度字段（如 rmsnorm 的 ab.py 只测时间）。
        # **但 acc_probe 可能已经测了** —— 有 acc 结果就先信它，别直接判「没数」。
        # 原实现一到这里就返回 ACCEPTED_NO_NUMERICS，于是 acc_probe 白测：
        # 精度通道接上了，判定却从不看它。这是「接了但没接上」的又一例。
        # 实测（2026-11-25）：acc.json 里明明有 max_ulp=1.0，记录仍写 acc_ok=False。
        if acc_cand:
            pr = acc_cand.get(primary) or next(iter(acc_cand.values()), {})
            ulp = pr.get("max_ulp")
            cos = pr.get("cosine")
            if ulp is not None:
                if ulp == 0.0:
                    return "ACCEPTED_BITEXACT", (
                        "显著加速 %.1f%%，acc_probe 报 0 ULP（逐位一致）%s"
                        % (-cand.get("delta_pct", 0.0), tail))
                return "ACCEPTED_NUMERICS", (
                    "显著加速 %.1f%%，acc_probe 实测最差 %.2f ULP（≤%d）、cosine=%.7f%s"
                    % (-cand.get("delta_pct", 0.0), ulp, int(MAX_ULP),
                       cos if cos is not None else 1.0, tail))
        # 真的一条精度数都没有 —— 保持原判：不许把「没测」读成「一致」。
        return "ACCEPTED_NO_NUMERICS", (
            "显著加速 %.1f%%，但**本卡未产出精度字段**（ab.py 不返回 maxdiff，"
            "acc_probe 也无结果）⇒ 不能声称逐位一致，必须另行过正确性门（done_when）%s"
            % (-cand.get("delta_pct", 0.0), tail))
    if maxdiff == 0.0:
        return "ACCEPTED_BITEXACT", "显著加速 %.1f%%，且输出逐位一致%s" % (-cand.get("delta_pct", 0.0), tail)
    return "ACCEPTED_NUMERICS", ("显著加速 %.1f%%，但**非逐位一致**（maxdiff=%.2e）⇒ 必须过端到端质量门%s"
                                 % (-cand.get("delta_pct", 0.0), maxdiff, tail))


_V2C = {}


def _check_base_role(card, defaults, calibration=False):
    """P0-2：基线必须是可作性能对比的角色。

    `correctness_only`（含 ref-heavy / torch-native 这类未优化参考实现）
    可以用于验正确性，但「打赢未优化的参考实现」没有工程意义——
    而 loop.py 原先完全不读这个字段，换张卡就可能把参考实现当基线。
    """
    # variants.yaml 的读法与 schema（`variants:` 列表、元素是 dict）收在 card 里：
    # 这里原先自己 safe_load + `d.get("variants") or []`，于是同一份文件在
    # loop / explore / rank_impls 三处各有一套「不是 dict 怎么办」的分支。
    roles = {}
    for v in _CARD.load_variants(card):
        if isinstance(v, dict):
            tag = v.get("tag") or v.get("name") or v.get("id")
            if tag:
                roles[str(tag)] = str(v.get("role") or "")
    # 基线名可能来自 defaults.base，也可能写在 bench_args 里
    base = str(defaults.get("base") or "")
    if not base:
        ba = defaults.get("bench_args") or []
        for i, a in enumerate(ba):
            if a == "--base" and i + 1 < len(ba):
                base = str(ba[i + 1])
                break
    if base and roles.get(base) == "correctness_only":
        # 新算子还没有 competitor 时，参考实现是**唯一**可用基线；
        # 这时允许跑，但记录里必须标明"对比对象是参考实现"，不能当成
        # "打赢了现成库"。ForgeKernel 的 target 就是"比参考快 x 倍"。
        has_competitor = any(r in ("competitor", "primary", "fallback")
                             for r in roles.values())
        if not has_competitor:
            print(_c("caveat：本卡没有 competitor 变体 —— 以参考实现作基线，"
                     "结论只代表「比参考快」，不代表打赢现成库。", WARN))
            return
        if calibration:
            # 新算子第一次跑：还没有 competitor 基线，先用参考实现**校准**，
            # 只取 t_actual，不声称收益（判定写 CALIBRATION_ONLY）。
            print(_c("caveat：基线 %s 是 correctness_only 参考实现 —— "
                     "本轮只作校准（CALIBRATION_ONLY），不计收益。"
                     % base, WARN))
            return
        print(_c("拒绝跑：基线 %s 的 role=correctness_only —— 它是未优化的参考实现，"
                 "打赢它没有工程意义。换一个 role=competitor 的实现作基线。"
                 % base, WARN))
        raise SystemExit(2)


def _cid_of(variant):
    """变体名 → 候选 id 的反查。没有映射就返回 None。"""
    return _V2C.get(variant)


def _bn_signal(card):
    """从 roofline.json 读 bottleneck signal（M2.5 diagnose.py 的产出位置）。

    没有就返回 None —— 不许编一个。signal 是 diagnose.py 从 PMC 算出来的，
    人写的不算。
    """
    p = card / "roofline.json"
    if not p.is_file():
        return None
    d = _CARD.load_roofline(card)
    sig = d.get("bottleneck_signal")
    return sig if isinstance(sig, str) and sig else None


def _falsified_of(decision):
    """decision → falsified 三态的**唯一映射**（v4 螺栓 B1）。

    true   —— 这条方向被证伪，EV 衰减因子要吃它
    false  —— 方向有效
    null   —— 本轮不可信（noise / 没测出结论），**不计入证伪次数**

    为什么 noise 不能计入证伪：噪声过大时判「没提升」本身就是错的，
    把它算成证伪会让 EV 的 (1 − 0.5×已证伪次数) 因子错误衰减，
    好方向被过早判死。这是 loop.py 早先 impl-broken / direction-refuted
    两分法的延伸：实现级失败（没做成）≠ 方向级失败（做成了但没用）。
    """
    if decision in ("REJECTED_NO_GAIN", "REJECTED_ACCURACY"):
        # 显著变慢、或精度不过 —— 都是「这条路不通」的硬证据
        return True
    if decision in ("INCONCLUSIVE", "ERROR"):
        return None
    if decision in ("ACCEPTED_BITEXACT", "ACCEPTED_NUMERICS",
                    "ACCEPTED_NUMERICS_PENDING_E2E", "ACCEPTED_NO_NUMERICS"):
        return False
    return None


def judge(base_median, base_mad, cand_median, cand_mad,
          noise_cutoff=0.05, k=3.0):
    """**唯一的显著性与噪声判定实现**。文档和代码都指向这一个函数。

    返回 (verdict, reason)。verdict ∈
        "noise"                 噪声过大，本轮不可信（不决策，重跑）
        "significant"           候选显著更快
        "significant_regression"候选显著更慢 —— **比无效更强的证伪信号**
        "insignificant"         差异在噪声内

    三条设计决定，每条都有代价：
      ① 噪声尺度用 **baseline 的 MAD**。用候选自身 MAD 会「越快越容易判显著」：
         优化后耗时降、MAD 可能也降，3×MAD 门槛跟着变小——那是自欺。
      ② `noise_cutoff`：baseline 的 MAD/median > 5% 直接判 noise。
         此时任何 delta 都不可信，判了也是在噪声上做决策。
      ③ `significant_regression` 单列出来，不许并进 significant。
         一个让算子变慢的改动，和让算子变快的改动，在 EV 衰减里必须是两种处理。
    """
    if not base_median or not cand_median:
        return "noise", "缺少数值（base=%r cand=%r）" % (base_median, cand_median)
    base_mad = base_mad or 0.0
    cand_mad = cand_mad or 0.0
    rel = base_mad / base_median
    if rel > noise_cutoff:
        return "noise", ("baseline MAD/median=%.1f%% > %.0f%%，本轮不可信（重跑而不是决策）"
                         % (rel * 100, noise_cutoff * 100))
    delta = abs(cand_median - base_median)
    thr = k * max(base_mad, cand_mad)
    if delta <= thr:
        return "insignificant", ("|Δ|=%.3g ≤ %.0f×max(MAD)=%.3g（噪声内）"
                                 % (delta, k, thr))
    if cand_median < base_median:
        return "significant", ("显著加速 %.1f%%（|Δ|=%.3g > %.0f×MAD=%.3g）"
                               % ((1 - cand_median / base_median) * 100, delta, k, thr))
    return "significant_regression", ("**显著变慢 %.1f%%**（|Δ|=%.3g > %.0f×MAD=%.3g）"
                                      "—— 该方向不仅无效且有害"
                                      % ((cand_median / base_median - 1) * 100, delta, k, thr))


def _significant(base, cand, k=3.0):
    """(significant, ratio, why) —— **委托给 judge()，不再自己算一套**。

    原先这里内联了一份 3×MAD 逻辑，与 decide() 里的那份是两份独立实现。
    两份实现意味着「文档说显著、代码说不显著」这种情况迟早发生（评审指出）。
    现在唯一实现是 judge()，本函数只是把它适配成 (bool, ratio, why) 的旧签名。
    """
    b, bmad = base["median_ms"], base.get("mad_ms", 0.0) or 0.0
    c, cmad = cand["median_ms"], cand.get("mad_ms", 0.0) or 0.0
    if not b or not c:
        return False, None, "缺少数值"
    verdict, why = judge(b, bmad, c, cmad, k=k)
    return verdict in ("significant", "significant_regression"), (c / b), why


# ---------------------------------------------------------------- 子命令

# references/ 是**知识**，留在 skills/op-optimize/ 下（见 docs/ENGINE.md 模块表）。
# 实现迁到 engine/ 后，这里必须显式跨目录 —— 原先是同目录，迁移时漏了这一处。
_REFS = Path(__file__).resolve().parents[1] / "skills" / "op-optimize" / "references"
SURFACES_PATH = _REFS / "surfaces.yaml"
SCENARIOS_PATH = _REFS / "scenarios.yaml"


def load_surfaces():
    """读攻击面表。缺文件返回空表（不报错——老卡片环境可能没有）。"""
    d = _CARD.read_yaml(SURFACES_PATH, [])
    return [s for s in d if isinstance(s, dict) and s.get("id")]


def load_scenarios():
    """读场景映射。缺文件返回空列表（老环境兼容）。"""
    d = _CARD.read_yaml(SCENARIOS_PATH, {})
    return [s for s in (d.get("scenarios") or []) if isinstance(s, dict) and s.get("id")]


def cmd_enumerate(a):
    """列出可用攻击面，并标出「已被方向预算封掉」的那些。

    这是给人和 agent 看的候选菜单：direction 只能从这张表里挑，
    不再允许凭空发明。已封的方向直接标出来，避免白跑。

    --new 只打印可以填进 candidates.yaml 的 YAML 片段。
    --scenario <id> 按场景给**优先序**（这是「该先试哪几条」的答案）。
    --list-scenarios 列出所有场景。
    """
    if getattr(a, "list_scenarios", False):
        scs = load_scenarios()
        if not scs:
            print("没有 scenarios.yaml")
            return 2
        for s in scs:
            print("%-22s %s" % (s["id"], s.get("statement", "")))
        return 0

    if getattr(a, "scenario", None):
        return _enumerate_by_scenario(a)

    surfaces = load_surfaces()
    if not surfaces:
        print(_c("找不到 %s" % SURFACES_PATH, WARN), file=sys.stderr)
        return 2

    cands = load_candidates(a.card)
    table = direction_table(cands)
    pending_dirs = {str(c.get("direction")) for c in (cands.get("candidates") or [])
                    if isinstance(c, dict) and str(c.get("status") or "pending") == "pending"}

    rows = []
    for s in surfaces:
        sid = str(s["id"])
        # ---- P0-4：dead_end 不进候选推荐
        # surfaces.yaml 里 `verdict: dead_end` 的两条（slc-policy / prefetch-depth-isa）
        # 是**已实测证伪的死路**。bound-to-class.yaml 的注释写着
        # 「留在映射里是为了查得到、别重试，不是推荐去试」，但代码不读 verdict，
        # 于是它们照样出现在候选菜单里。
        if str(s.get("verdict") or "") == "dead_end":
            rows.append(("DEAD_END", sid, str(s.get("class") or ""),
                         str(s.get("cost") or ""), str(s.get("law") or ""),
                         str(s.get("statement") or ""),
                         "已实测证伪，只进 recall 不进推荐"))
            continue
        info = table.get(sid)
        if info and info["exhausted"]:
            state, note = "SEALED", "已证伪 %d 次，回 L1 重定约束" % info["refutes"]
        elif sid in pending_dirs:
            state, note = "QUEUED", "已有 pending 候选挂在这个方向"
        elif info and info["refutes"]:
            state, note = "OPEN*", "已证伪 %d 次（再 1 次封方向）" % info["refutes"]
        else:
            state, note = "OPEN", ""
        rows.append((state, sid, s.get("class", ""), s.get("cost", ""),
                     s.get("law", ""), s.get("statement", ""), note))

    if a.new:
        for state, sid, cls, cost, law, stmt, _ in rows:
            if state.startswith("OPEN"):
                print("- id: %s" % sid)
                print("  direction: %s" % sid)
                print("  status: pending")
                print("  # %s（%s/%s  %s）" % (stmt, cls, cost, law))
        return 0

    print("%-7s %-18s %-12s %-4s %-4s %s" %
          ("状态", "surface", "class", "cost", "law", "说明"))
    for state, sid, cls, cost, law, stmt, note in rows:
        color = {"OPEN": "", "OPEN*": WARN, "QUEUED": "", "SEALED": WARN}.get(state, "")
        line = "%-7s %-18s %-12s %-4s %-4s %s" % (state, sid, cls, cost, law, stmt)
        if note:
            line += "  ← " + note
        print(_c(line, color) if color else line)

    open_n = sum(1 for r in rows if r[0].startswith("OPEN"))
    print("\n可挑 %d 条 / 共 %d 条；已封 %d 条。"
          % (open_n, len(rows), sum(1 for r in rows if r[0] == "SEALED")))
    print("下一步：把选中的 id 填进 candidates.yaml 的 direction（可加 --new 直接出片段）")
    return 0


def cmd_resume(a):
    """一句话回答「上下文丢了以后，下一个该跑谁」。

    这是「状态机 ≥ 上下文」的最小实现：所有判断都从落盘状态算出来，
    不依赖任何模型记忆。上下文清空、换人接手、隔天重开，跑这一条就够。

    输出四段：
      next     可以立刻跑的（方向 open、无 claim、未超预算）
      blocked  不该跑的（方向已封 / 被 claim 占着）
      held     跑过但不达主张门槛的（要补口径/轮数/精度，不是重跑）
      counts   总量

    --json 给机器读；默认给人读。
    """
    card = a.card
    cands = load_candidates(card)
    table = direction_table(cands)
    pending = [c for c in cands["candidates"] if isinstance(c, dict)
               and str(c.get("status") or "pending") == "pending"]

    # 每条 tag 的最新记录：用来判 held（最后跑过的那次是否达门槛）
    latest = {}
    for r in read_records(card):
        if r.get("tag") and "cand_ms" in r:
            latest[r["tag"]] = r

    claims = {}
    cdir = card / "runs" / "loop" / "claims"
    if cdir.is_dir():
        for p in cdir.glob("*.claim"):
            try:
                claims[p.stem] = json.loads(p.read_text())
            except (ValueError, OSError):
                claims[p.stem] = {"worker": "?", "pid": 0}

    nxt, blocked, held = [], [], []
    for c in pending:
        tag = str(c.get("id") or "?")
        d = str(c.get("direction") or "")
        rec = latest.get(tag)

        if rec is not None:
            ok, reasons = claim_gate(rec)
            if not ok:
                held.append({"tag": tag, "surface": d, "reasons": reasons})
                continue

        info = table.get(d)
        if info and info["exhausted"]:
            blocked.append({"tag": tag, "reason": "direction exhausted (%s %d/%d)"
                            % (d, info["refutes"], MAX_DIRECTION_REFUTES)})
            continue
        if tag in claims:
            blocked.append({"tag": tag, "reason": "claim held by %s"
                            % claims[tag].get("worker", "?")})
            continue

        surf = {s["id"]: s for s in load_surfaces()}.get(d, {})
        nxt.append({"tag": tag, "surface": d, "law": surf.get("law"),
                    # 方向不在 surfaces.yaml 里 = 本算子特有的方向，没挂定律。
                    # 不隐藏这一点：判死时就没法追溯到机制。
                    "in_table": bool(surf),
                    "priority": c.get("priority"), "status": "open"})

    nxt.sort(key=lambda x: (x.get("priority") is None, x.get("priority") or 0, x["tag"]))

    # 前置检查：run 需要卡内 A/B 适配器。缺了就别建议一条注定失败的命令——
    # 冷启动最伤人的体验就是「照着提示跑，立刻被拒」。
    defaults = cands.get("defaults") or {}
    bench_rel = defaults.get("bench", "bench/ab.py")
    bench_ok = (card / bench_rel).exists()
    if not bench_ok:
        blocked.append({"tag": "(全部)", "reason": "缺 A/B 适配器 %s —— 先补齐再跑"
                        % bench_rel})

    counts = {"pending": len(pending), "refuted": len(cands["refuted"]),
              "next": len(nxt), "blocked": len(blocked), "held": len(held)}

    if a.json:
        print(json.dumps({"next": nxt, "blocked": blocked, "held": held,
                          "bench_ready": bench_ok,
                          "counts": counts}, ensure_ascii=False, indent=2))
        return 0

    print("下一个该跑谁（全部从落盘状态算出，不依赖记忆）\n")
    if nxt:
        print("可跑 %d 条%s：" % (len(nxt), "" if bench_ok else "（但当前跑不了，见下）"))
        for x in nxt:
            law = x.get("law") or ("(算子特有方向，无定律)" if not x["in_table"] else "-")
            print("  %-22s direction=%-24s law=%s"
                  % (x["tag"], x["surface"] or "(未声明)", law))
        if bench_ok:
            print("\n  → python3 loop.py -C %s run %s" % (a.card.name, nxt[0]["tag"]))
        else:
            print("\n  ✗ 先补 %s 才能跑（当前卡片没有 A/B 适配器）" % bench_rel)
    else:
        print(_c("没有可跑的候选。", WARN))
        if not pending:
            print("  candidates.yaml 里没有 pending 候选 —— 用 enumerate --new 生成方向片段。")
        print("  → python3 loop.py -C %s enumerate --new" % a.card.name)

    real_blocked = [x for x in blocked if x["tag"] != "(全部)"]
    if real_blocked:
        print("\n不该跑 %d 条：" % len(real_blocked))
        for x in real_blocked:
            print("  %-22s %s" % (x["tag"], x["reason"]))
    if held:
        print("\n跑过但不够格 %d 条（补证据，不是重跑）：" % len(held))
        for x in held:
            print("  %-22s %s" % (x["tag"], ", ".join(x["reasons"])))
        print("  → python3 loop.py -C %s report --all 看逐条补齐方法" % a.card.name)

    print("\n合计：pending %(pending)d / 可跑 %(next)d / 阻塞 %(blocked)d / 待补 %(held)d"
          " / 已判死 %(refuted)d" % counts)
    return 0


def _enumerate_by_scenario(a):
    """按场景列出攻击面优先序——回答「这个场景该先试哪几条」。"""
    scs = {s["id"]: s for s in load_scenarios()}
    s = scs.get(a.scenario)
    if s is None:
        print(_c("未知场景 %r；可用：%s" % (a.scenario, sorted(scs)), WARN))
        return 2

    cands = load_candidates(a.card)
    table = direction_table(cands)

    print("场景 %s —— %s" % (s["id"], s.get("statement", "")))
    print("判据   %s" % s.get("detect", ""))
    print("瓶颈   %s\n" % s.get("bound", ""))
    print("%-4s %-18s %-6s %s" % ("序", "surface", "状态", "为什么在这个场景优先"))
    for i, item in enumerate(s.get("surfaces") or [], 1):
        sid = item.get("id")
        info = table.get(sid)
        state = "SEALED" if (info and info["exhausted"]) else "OPEN"
        line = "%-4d %-18s %-6s %s" % (i, sid, state, item.get("why", ""))
        print(_c(line, WARN) if state == "SEALED" else line)
        if state == "SEALED":
            print("     └ 已封（证伪 %d 次）——本场景跳过它" % info["refutes"])
    er = s.get("engine_ref")
    if er:
        # engine 层结论：pipeline 级决策，不是单算子结论。
        # 指向 op-atlas/cards/<card>.yaml，那里有完整 scope。
        print("\n适用本场景的 engine 层结论（layer: engine，非单算子）：")
        print("  卡       %s（op-atlas/cards/%s.yaml）" % (er.get("card"), er.get("card")))
        if er.get("applies_when"):
            print("  适用条件 %s" % er["applies_when"])
        print("  决策     %s" % er.get("decision", ""))
        if er.get("why"):
            print("  理由     %s" % er["why"])
        print("  注意     engine 级数字带 scope，不要当单算子结论引用")

    if s.get("dead_ends"):
        print("\n本场景的已知死路（先排掉，别试）：")
        for d in s["dead_ends"]:
            print("  ✗ %s" % d)
    print("\n下一步：把选中的 surface id 填进 candidates.yaml 的 direction")
    return 0


def round_stats(recs):
    """**轮次 ≠ 重跑次数**。

    一条 round = (card, shape, direction, tag) 的**首次出现**；
    同 key 之后的行是 rerun（重复测同一个东西，不产生新信息）。

    为什么必须分开：R0 的 5 条记录是同一个 tag 重跑 5 次，不是 5 轮实验。
    如果 DS 的 D4（≥3 轮）按 jsonl 行数算，重跑 3 次就"达标"了——
    那是刷分，不是迭代。所以 D4 只许数 round，且每轮 verdict 必须
    ∈ {adopt, refute, stalled} 之一。
    """
    seen = {}
    rounds, reruns = [], []
    for r in recs:
        key = (r.get("card"), json.dumps(r.get("shape") or {}, sort_keys=True),
               r.get("direction"), r.get("tag"))
        if key in seen:
            reruns.append(r)
        else:
            seen[key] = r
            rounds.append(r)
    return rounds, reruns


def next_step_hint(cands, runnable, card_name=""):
    """没有可跑候选时给下一步。返回 (文案, 是否需要补线)。

    为什么要单独成一个函数：这里是「引擎与使用者的唯一接口」，而它此前在
    「待做 0」时什么都不说。agent 唯一一次执行 loop.py 就是在这一步走掉的。
    """
    pend = [c for c in cands.get("candidates") or []
            if isinstance(c, dict) and c.get("status", "pending") == "pending"]
    prop = [c for c in cands.get("candidates") or []
            if isinstance(c, dict) and c.get("status") == "proposed"]
    cmd = "python3 -I scripts/explore.py --card %s --apply" % (card_name or "<卡>")
    if runnable:
        return "", False
    if pend:
        return ("\n下一步：待做的 %d 条都被方向预算或登记门拦住了（见上）。"
                % len(pend)), False
    if prop:
        lines = ["\n下一步：待做 0 条，已有 %d 条**拟做**（status=proposed）。"
                 "它们还没有实现变体，所以 run 会（且应该）拒绝：" % len(prop)]
        for c in sorted(prop, key=lambda x: x.get("priority", 99))[:3]:
            lines.append("  [%s] %s — %s" % (c.get("priority", "-"), c.get("id"),
                                              str(c.get("hypothesis", ""))[:70]))
        lines.append("  推进：实现变体 → 把 status 改成 pending 并补 variant/done_when "
                     "→ loop.py run <id>")
        return "\n".join(lines), False
    return ("\n下一步：待做 0 条、拟做 0 条 —— 这张卡上没有候选了。\n"
            "  从通用信号（roofline / 带宽 / 占用率 / LDS / 串行比）生成拟做候选：\n"
            "    %s\n"
            "  只看不写：去掉 --apply。生成的是**拟做**不是待做：变体实现出来之前\n"
            "  run 仍然会拒，这是设计如此，不是故障。" % cmd), True


def _load_floor():
    """延迟加载 engine/floor.py（拿 Floor 对象给报告用）。"""
    import importlib.util
    key = "_engine_floor"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, str(Path(__file__).parent / "floor.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def _nextstep():
    """延迟加载 engine/nextstep.py（本仓没有 packaging，按路径加载）。"""
    import importlib.util
    key = "_engine_nextstep"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, str(Path(__file__).parent / "nextstep.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def cmd_next(a):
    """打印「下一步做什么」—— 给 agent 和人都只用读这几行。"""
    ns = _nextstep()
    plan = ns.plan_next(a.card)
    if getattr(a, "json", False):
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        print(ns.render(plan))
    # 退出码：能推进=0，已收敛/需收尾=1。脚本化调用靠它判断是否继续。
    return 0 if plan.get("action") in ("run", "baseline", "implement", "explore") else 1


def cmd_status(a):
    card, cands, recs = a.card, load_candidates(a.card), read_records(a.card)
    pend = [c for c in cands["candidates"]
            if isinstance(c, dict) and c.get("status", "pending") == "pending"]
    print("卡片      %s" % card)
    # **第一行就说下一步**。实测（2026-11-28）：C 臂唯一一次执行 loop.py 就是
    # `status`，读到「待做 0」就走了。它不会再读第二屏 —— 所以下一步必须在这儿。
    try:
        _ns = _nextstep()
        _plan = _ns.plan_next(card)
        _lbl = {"run": "跑实验", "implement": "实现变体", "explore": "出题",
                "baseline": "先测基线", "stop": "停止"}.get(_plan.get("action"),
                                                           _plan.get("action"))
        print("下一步    %s —— %s" % (_lbl, str(_plan.get("why", ""))[:100]))
        print("命令      %s" % _plan.get("cmd", ""))
    except Exception as _e:                      # noqa: BLE001
        print(_c("下一步    算不出来（%s: %s）" % (type(_e).__name__, _e), WARN))
    _rounds, _reruns = round_stats(recs)
    prop = [c for c in cands["candidates"]
            if isinstance(c, dict) and c.get("status") == "proposed"]
    print("候选      待做 %d / 拟做 %d / 已判死 %d / 停滞可作父代 %d / 共记录 %d 条"
          % (len(pend), len(prop), len(cands["refuted"]),
             len(cands.get("stalled") or []), len(recs)))
    # 轮次与重跑分开报：同一 tag 反复测不算多轮（P0-6）
    print("轮次      独立实验 %d 轮 / 重跑 %d 次（D4 只数前者）"
          % (len(_rounds), len(_reruns)))
    table = direction_table(cands)
    if table:
        print("方向      " + " · ".join(
            "%s %d/%d%s" % (k, v["refutes"], MAX_DIRECTION_REFUTES,
                            "（已封）" if v["exhausted"] else "")
            for k, v in sorted(table.items())))
    for prob in budget_problems(cands):
        print(_c("caveat " + prob, WARN))
    for prob in stalled_parent_problems(cands):
        print(_c("caveat " + prob, WARN))
    if recs:
        last = recs[-1]
        print("最近一次  %s %s → %s" % (last.get("ts"), last.get("tag"), last.get("decision")))
    best = {}
    for r in recs:
        if r.get("decision", "").startswith("ACCEPTED") and r.get("decision") != "ACCEPTED_NUMERICS_PENDING_E2E":
            key = r["skv"]
            if key not in best or r["cand_ms"] < best[key]["cand_ms"]:
                best[key] = r
    for skv in sorted(best):
        r = best[skv]
        print("最强      SKV=%-7d %-14s %.4f（base %.4f，%+.1f%%）"
              % (skv, r["tag"], r["cand_ms"], r["base_ms"], r["delta_pct"]))
    runnable = [c for c in pend if not budget_problems(cands, tags={c.get("id")})]
    if runnable:
        print("\n下一步：")
        for c in sorted(runnable, key=lambda x: x.get("priority", 99))[:3]:
            print("  [%s] %s — %s" % (c.get("priority", "-"), c.get("id"), c.get("hypothesis", "")))
    else:
        hint, _needs_forge = next_step_hint(cands, runnable, card.name)
        if hint:
            print(hint)


def cmd_queue(a):
    d = load_candidates(a.card)
    done = {r["tag"] for r in read_records(a.card)}
    table = direction_table(d)
    if table:
        print("# 方向预算（证伪 %d 次封方向 ⇒ 该方向的候选一律不再受理）" % MAX_DIRECTION_REFUTES)
        for k, v in sorted(table.items()):
            print("%-26s %d/%d  %-9s %s" % (k, v["refutes"], MAX_DIRECTION_REFUTES,
                                           v["status"], v["statement"]))
        print()
    print("# 待做（按 priority）")
    for c in sorted([c for c in d["candidates"]
                     if isinstance(c, dict) and c.get("status", "pending") == "pending"],
                    key=lambda x: x.get("priority", 99)):
        mark = "已跑过" if c["id"] in done else ""
        print("%-3s %-22s [%s] %s" % (c.get("priority", "-"), c["id"],
                                      c.get("direction", "无方向"), c.get("hypothesis", "")))
        for k in ("expect", "why", "evidence"):
            if c.get(k):
                print("      %-8s %s" % (k, c[k]))
        fails = c.get("fails") or []
        if fails:
            print("      %-8s %d/%d 次实现失败；最近：%s"
                  % ("fails", len(fails), MAX_IMPL_FAILS, fails[-1]))
        if mark:
            print("      %-8s %s" % ("note", mark))
    if d.get("stalled"):
        print("\n# 停滞（变慢但结构变了 —— 可以当父代，方向未封）")
        for r in d["stalled"]:
            print("%-22s %-28s %s" % (r["id"], r.get("what_changed", ""), r.get("why", "")))
    if d["refuted"]:
        print("\n# 已判死（别重开；机制比结论重要）")
        for r in d["refuted"]:
            print("%-22s %-10s %s" % (r["id"], r.get("kind", "impl"), r.get("why", "")))
    dw = done_when_problems(d)
    if dw:
        print()
        for prob in dw:
            print(_c("caveat " + prob, WARN))
    probs = budget_problems(d)
    if probs:
        print()
        for prob in probs:
            print(_c("caveat " + prob, WARN))
        return 2
    return 0


# 天花板 → 允许的动作类。这是 op-constrain 的替代：约束来自 roofline 的 verdict，
# 不是来自一个不存在的 skill。判据：天花板决定了「哪一类改动可能有效」。
# 单一真源（与 atlas 的 ceiling.bound 词表对齐）：
#   atlas 卡里 26 张写 bandwidth、13 张写 mixed、13 张写 compute、1 张写 latency。
#   loop.py 原先只认 memory/compute/latency —— 39 张卡的词不在表里，
#   而 load_allowed_actions 对未知 bound **静默 return None（= 不拦）**，
#   等于天花板约束在多数卡上失效。这是「薄约束在最后一步漏了」的重演。
BOUND_ALIASES = {
    "bandwidth": "memory",   # atlas 的写法
    "memory": "memory",
    "compute": "compute",
    "latency": "latency",
    "mixed": "mixed",
    "launch": "launch",
}

BOUND_ACTIONS = {
    # 访存受限：只有减字节的改动有意义；调 tile/提高发射率都是白费
    "memory": {"reduce-bytes", "improve-locality", "reuse-across-iter"},
    # 混合受限：两类都可能有意义，放行全集（由 plan.py 的 EV 排序决定）
    "mixed": {"reduce-bytes", "improve-locality", "reuse-across-iter",
              "reduce-flops", "raise-issue-rate", "precision-drop",
              "pipeline-depth", "raise-concurrency"},
    # launch 受限：融合是唯一有效手段
    "launch": {"reduce-bytes", "raise-concurrency"},
    # 算力受限：只有减 FLOP 或提高发射率有意义；减字节没用（已经不是在等数据）
    "compute": {"reduce-flops", "raise-issue-rate", "precision-drop"},
    # 依赖链/并发受限：改流水线与驻留，不是改 tile 大小
    "latency": {"pipeline-depth", "raise-concurrency", "persistent-dispatch"},
}


def load_allowed_actions(card):
    """读卡片的天花板判定，映射成允许的动作类集合。

    产物来源：`<卡>/roofline.json` 里的 `bound` 字段（由 op-roofline 的
    derive.py 产出）。卡里没有这个文件时返回 None —— **不校验**，
    因为不是每张卡都做过 roofline 分析；但有就必须遵守。
    """
    p = card / "roofline.json"
    if not p.is_file():
        return None, None
    d = _CARD.load_roofline(card)
    bound = d.get("bound")
    if not bound:
        return None, "roofline.json 缺 bound 字段"
    canon = BOUND_ALIASES.get(str(bound))
    if canon is None:
        # **fail-closed**：未知 bound 不许静默放行。
        # 第一版 return None 让「不拦」和「拿不到」无法区分，
        # 39 张卡的 bandwidth/mixed 因此全部绕过了天花板约束。
        return None, "未知 bound=%r（只允许 %s）" % (bound, sorted(BOUND_ALIASES))
    allowed = BOUND_ACTIONS.get(canon)
    if allowed is None:
        return None, "bound=%r 已归一为 %r 但 BOUND_ACTIONS 里没有" % (bound, canon)
    return set(allowed), canon


BENCH_STD_NOTE = (
    "本文件的数字来自 loop.py run 调起的 bench/ab.py（GPU event、event_isolated 口径）。"
    "**由引擎生成** —— 手写的同形状文件不满足这条链路的可复现契约。")


def write_bench_std(out_path, card, ab, variants, denominator=None):
    """把一次 `loop.py run` 的实测写成 opbench 口径的 `results/bench_std.json`。

    为什么要由引擎写而不是让 agent 自己拼：评分只读这个文件，而 `loop.py run`
    原先只写 `runs/loop/*.jsonl` —— **交付物与引擎产出是两套**，agent 自然
    先手搓一份能得分的，再把 loop.py 当打卡。见模块说明。

    形状取自卡的 `shapes.yaml`（ab 里带 params 就用它），不猜 uuid：
    opbench 的 `axes` 必须与题目逐项相同，多/少一个键都判无效，所以这里
    只搬运卡里声明的参数，不添油加醋。
    """
    import json as _json
    shape = (ab or {}).get("shape") or {}
    params = dict(shape.get("params") or {})
    rows = []
    iters = int((ab or {}).get("rounds") or 0) or 1
    for run in (ab or {}).get("runs") or []:
        for v in run.get("variants") or []:
            if v.get("tag") not in variants:
                continue
            ms = v.get("ms")
            if not isinstance(ms, (int, float)):
                continue
            rows.append({
                "uuid": str(shape.get("shape") or shape.get("_name") or "default"),
                "axes": params,
                "median_ms": float(ms),
                # ab.py 目前只报中位数与 MAD。**不假装采过 min**：写明来源。
                "min_ms": float(ms),
                "iters": iters,
                "warmup": 10,
                "l2_flush_bytes": int((ab or {}).get("l2_flush_bytes") or 33554432),
            })
            break
    if not rows:
        return None
    doc = {
        "task": card.name,
        "shapes": rows,
        "impl": "%s/bench/ab.py:_measure  —  变体 %s（由 loop.py run 调起，"
                "同一进程内 event 计时）" % (card.name, ",".join(sorted(variants))),
        "detail": {
            "generated_by": "Plugins/ops/engine/loop.py::write_bench_std",
            "card": card.name,
            "measure_mode": (ab or {}).get("measure_mode"),
            "min_ms_source": "ab.py 未单独采集 min，此处与 median 相同",
            "note": BENCH_STD_NOTE,
        },
    }
    if denominator:
        doc["denominator"] = denominator
    p = Path(out_path)
    if not p.is_absolute():
        # 相对路径按**调用者的 cwd** 解析：agent 在题目工作目录里跑 loop.py，
        # `--bench-std results/bench_std.json` 就该落到题目的 results/ 下。
        p = Path.cwd() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# 报告模板：文本（标题 / 表格 / 说明）全在 `instrument/prompts/report.md`，
# 本文件只把实测数字和行数据填进去。按 `<!-- name -->` 分片段；非最后一段去掉
# 与下一标记之间的分隔换行，最后一段（wr_main）原样读到文件末尾。
_REPORT_TPL = (Path(__file__).resolve().parent
               / "instrument" / "prompts" / "report.md")
_REPORT_MARK = re.compile(r"^<!-- (\w+) -->\n", re.M)
_REPORT_PH = re.compile(r"\{\{(\w+)\}\}")


def _report_fragments():
    """`report.md` 的命名片段（`<!-- name -->` 分隔）。"""
    text = _REPORT_TPL.read_text(encoding="utf-8")
    marks = list(_REPORT_MARK.finditer(text))
    frags = {}
    for i, m in enumerate(marks):
        start = m.end()
        if i + 1 < len(marks):
            body = text[start:marks[i + 1].start()]
            if body.endswith("\n"):          # 片段之间的分隔换行
                body = body[:-1]
        else:
            body = text[start:]              # 最后一段：保留文件末尾的原样
        frags[m.group(1)] = body
    return frags


def _report_fill(tpl, **ctx):
    return _REPORT_PH.sub(lambda m: str(ctx.get(m.group(1), "")), tpl)


def write_report(out_path, card, rec, plan=None, floor=None, cmdline=""):
    """按 opbench 题面的六问，用**实测数据**填一份 report.md 骨架。

    为什么要引擎来写：题面要报告，`next` 驱动实验 —— 形状不同，agent 就会
    判定"跑实验答不了题"而去做调研（实测 mla 会话 24 条 Bash 里 0 条执行 run）。
    把报告和实测一起产出，走引擎才不比直接干活贵。

    **文本在 `instrument/prompts/report.md`**（本条注释以上）：这里只负责把
    记录里的数字格式化成模板占位符的值，以及把逐条数据拼成 block。
    """
    d = rec or {}
    nxt = (plan or {}).get("action")
    frags = _report_fragments()
    claim_note = _report_fill(
        frags["wr_claimable"] if d.get("measure_mode") == "event_isolated"
        else frags["wr_unclaimable"])
    sig_note = _report_fill(
        frags["wr_significant"] if d.get("significant")
        else frags["wr_insignificant"])
    floor_section = ""
    if floor is not None and getattr(floor, "ok", False):
        floor_section = _report_fill(
            frags["wr_floor"],
            floor_us="%.1f" % (getattr(floor, "t_floor_us", 0) or 0),
            floor_source=getattr(floor, "source", "?"),
            actual_us="%.1f" % (getattr(floor, "t_actual_us", 0) or 0),
            eff_pct="%.1f" % ((getattr(floor, "efficiency", 0) or 0) * 100),
            gate=getattr(floor, "gate", "?"),
            basis=(getattr(floor, "basis", "") or "（见 atlas 卡）"))
    acc = d.get("acc") or {}
    if acc:
        lines = []
        for qs, row in acc.items():
            lines.append(_report_fill(
                frags["wr_acc_item"], qscale=qs,
                max_abs="%.3e" % row.get("max_abs", 0),
                max_ulp="%.2f" % row.get("max_ulp", 0),
                cosine="%.7f" % row.get("cosine", 0),
                nonfinite=row.get("nonfinite", 0)))
        acc_block = "\n".join(lines) + "\n" + _report_fill(
            frags["wr_acc_footer"], tolerance=d.get("tolerance"),
            acc_ok=d.get("acc_ok"))
    else:
        acc_block = _report_fill(frags["wr_acc_missing"])
    next_section = ""
    if nxt:
        next_section = _report_fill(
            frags["wr_next"], next_action=nxt,
            next_why=(plan or {}).get("why", ""),
            next_cmd=(plan or {}).get("cmd", ""))
    text = _report_fill(
        frags["wr_main"],
        card_name=card.name,
        tag=d.get("tag"),
        measure_mode=d.get("measure_mode"),
        claim_note=claim_note,
        base_ms="%.4f" % (d.get("base_ms") or 0),
        base_mad="%.4f" % (d.get("base_mad") or 0),
        cand_ms="%.4f" % (d.get("cand_ms") or 0),
        mad="%.4f" % (d.get("mad") or 0),
        delta="%+.2f%%" % (d.get("delta_pct") or 0),
        sig_note=sig_note,
        floor_section=floor_section,
        acc_block=acc_block,
        decision=d.get("decision"),
        why=d.get("why"),
        cmdline=(cmdline or ("python3 -I Plugins/ops/engine/loop.py -C %s run %s"
                             % (card, d.get("tag")))),
        next_section=next_section)
    p = Path(out_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def cmd_run(a):
    card = a.card
    cands = load_candidates(card)
    refuted = {r["id"]: r for r in cands["refuted"]}
    declared = {c.get("id"): c for c in cands["candidates"] if isinstance(c, dict)}
    forced = set()
    for tag in a.tags:
        if tag in refuted:
            kind = refuted[tag].get("kind", "impl")
            label = {"direction": "方向已证伪", "impl": "实现已判死"}.get(kind, "已判死")
            if not a.force:
                print(_c("拒绝重跑 %s：%s —— %s" % (tag, label, refuted[tag].get("why", "")), WARN))
                print("（确实有理由就加 --force，并在记录里写清楚为什么这次不一样）")
                return 2
            # --force 是**显式否决**：这个 tag 已不在队列里，挂不上方向，就别拿方向预算
            # 去拦它；记录里 forced=true 留痕，事后能查出是谁把死马又跑了一遍
            forced.add(tag)
        elif tag not in declared:
            print(_c("拒绝跑 %s：不在 candidates.yaml 的队列里。先把它写进队列"
                     "（带 direction + expect），再跑 —— 没登记的实验记账无从谈起。" % tag, WARN))
            return 2
    # 第 3 步（定位瓶颈）没做 ⇒ 拒跑。这是 6 步循环里唯一原本零机检的一步，
    # 不拦的话 agent 可以直接列候选改代码，跳过「卡在算力还是内存」的判断。
    for prob in bottleneck_problems(cands, tags=set(a.tags), card=card):
        print(_c("拒绝跑：" + prob, WARN))
        return 2
    # 已 adopted 但没写 done_when ⇒ 它的「完成」没有任何外部裁决点，不许再跑
    for prob in done_when_problems(cands, tags=set(a.tags)):
        print(_c("拒绝跑：" + prob, WARN))
        return 2
    # 方向预算不可计 / 已封 ⇒ 这个 tag 的「2 次封方向」约束保护不到它，fail-closed
    for prob in budget_problems(cands, tags=set(a.tags) - forced):
        print(_c("拒绝跑：" + prob, WARN))
        return 2
    if forced:
        print(_c("caveat --force 重跑已判死项：%s（这次不一样的理由必须写进记录）"
                 % ", ".join(sorted(forced)), WARN))

    # 天花板约束：卡里有 roofline 判定时，候选的动作类必须落在允许集合内。
    # 这一步此前是断的——allowed_actions 只有提示语，没有读取和拦截，
    # 于是越界的改法照样能跑，薄约束在最后一步漏了。
    allowed, bound = load_allowed_actions(card)
    if allowed is not None:
        offenders = []
        for tag in a.tags:
            cand = declared.get(tag) or {}
            cls = cand.get("action_class")
            if cls is None:
                continue                    # 未声明动作类：不拦（向后兼容），但提示
            if cls not in allowed:
                offenders.append((tag, cls))
            elif cls:
                pass
        if offenders:
            for tag, cls in offenders:
                print(_c("拒绝跑 %s：动作类 %r 不在允许集合 %s 内"
                         % (tag, cls, sorted(allowed)), WARN))
            print("原因：本卡 roofline 判定天花板为 %r ⇒ 只有 %s 类改动可能有效。"
                  % (bound, sorted(allowed)))
            print("要越界请先改 roofline.json 的 bound（需附新的推导），"
                  "或在候选里把 action_class 改成允许集合内的值。")
            return 2
        undeclared = [t for t in a.tags
                      if not (declared.get(t) or {}).get("action_class")]
        if undeclared:
            # **拦，不是警告**。不声明 action_class 就等于绕过了天花板约束，
            # 而"绕过"和"没约束"在结果上无法区分——那条约束就等于不存在。
            # 原实现只打 caveat 然后放行（测试名也写着 warns_but_does_not_block），
            # 那是把"没填"当成"不需要填"。
            print(_c("拒绝跑 %s：未声明 action_class，天花板约束无法生效。"
                     "在候选里补 `action_class: <允许集合之一>`（bound=%r 允许 %s）"
                     % (", ".join(undeclared), bound, sorted(allowed)), WARN))
            return 2

    # ---- P0-2：基线角色校验
    # variants.yaml 里 correctness_only 的变体写着「不得用于性能对比：
    # 它是未优化的参考实现，打赢它没有工程意义」，而 loop.py 一个字不读。
    # R0 是靠 candidates.yaml 手写 --base lightop-fused 才躲过这条。
    d = dict(cands["defaults"])
    # ---- 基线晋升（自进化链）----
    # ForgeKernel 的核心：被接受的 kernel 成为下一轮的参照。
    # 我们把它落在 runs/loop/base.json（引擎状态，不改进用户文件），
    # 下一轮 run 自动以它为 base —— 否则每轮都在跟最开始的基线比，
    # 链条永远原地踏步。
    _promo = db_dir(card) / "base.json"
    if _promo.is_file():
        try:
            _pb = json.loads(_promo.read_text(encoding="utf-8")).get("base")
        except Exception:                                    # noqa: BLE001
            _pb = None
        if _pb and _pb != d.get("base"):
            d["base"] = _pb
            _ba = list(d.get("bench_args") or [])
            for _i, _x in enumerate(_ba):
                if _x == "--base" and _i + 1 < len(_ba):
                    _ba[_i + 1] = _pb
            d["bench_args"] = _ba
            print(_c("基线晋升：本轮以 %s 为参照（runs/loop/base.json）" % _pb, WARN))
    _calib = bool(a.tags) and all(
        (declared.get(t) or {}).get("purpose") == "calibration" for t in a.tags)
    _check_base_role(card, d, calibration=_calib)

    bench = card / d.get("bench", "bench/ab.py")
    if not bench.exists():
        print(_c("拒绝跑：本卡没有 A/B 适配器 %s（卡片需自带 `bench/ab.py`，契约见本文件 ADAPTER_AB）。"
                 "改用卡片自己的 bench/opcheck 命令跑，再用 `impl-broken` / `direction-refuted` 记账。"
                 % bench, WARN))
        return 2
    # skv 是 sparse-mla 的扫描维度，不是所有卡的必需项。
    # 没声明 sweep 的卡（rmsnorm 等）用 [None]，适配器自己输出 skv: 0 占位。
    skv = a.skv or d.get("skv") or ([] if d.get("no_sweep") else [32768]) or [None]
    ab_json = db_dir(card) / "_ab.json"
    # bench 参数必须由卡片自己声明，不能硬编码某一个卡的形状。
    # 原先这里写死了 --s/--topk/--nvalid/--heads/--skv（sparse-mla 的形状），
    # 结果 rmsnorm 的 ab.py 直接报 `ambiguous option: --s`——每张卡都被
    # 按 sparse-mla 的口径调用。loop.py 是从单卡演化来的，这层没抽象出来。
    bench_args = d.get("bench_args")
    if bench_args is None:
        # 没声明就退回旧行为（sparse-mla 兼容），但显式打印出来
        bench_args = ["--s", str(a.s or d.get("s", 128)),
                      "--topk", str(a.topk or d.get("topk", 2048)),
                      "--nvalid", str(a.nvalid or d.get("nvalid", 2000)),
                      "--heads", str(a.heads or d.get("heads", 64)),
                      "--skv", *[str(x) for x in skv]]
        print("note: candidates.yaml 未声明 bench_args，退回 sparse-mla 默认参数"
              "（其他卡应在 defaults.bench_args 里声明自己的）", flush=True)
    else:
        bench_args = [str(x) for x in bench_args]
        if a.s is not None:
            bench_args += ["--s", str(a.s)]
    # tag（候选 id）≠ variant（已实现的变体名）。原先直接把 tag 当位置参数传给
    # ab.py，而 ab.py 只认 --variants。候选应声明它对应哪个变体。
    vmap = {}
    for c in cands["candidates"]:
        if isinstance(c, dict) and c.get("variant"):
            vmap[c.get("id")] = str(c["variant"])
    tags = a.tags
    unmapped = [t for t in tags if t not in vmap]
    if unmapped:
        print(_c("拒绝跑：候选 %s 没声明 variant（候选 id ≠ 变体名，"
                 "在 candidates.yaml 里加 `variant: <已实现的变体名>`）"
                 % ",".join(unmapped), WARN))
        return 2
    variants_arg = ",".join(vmap[t] for t in tags)
    # 变体名 → 候选 id。**只登记本次要跑的 tag**：同一变体可能被多个候选复用
    # （fuse-residual 与 inplace-residual 都指向 fused-3pass-triton），
    # 若按整个 vmap 反查，字典后写者覆盖前者，跑 A 会记成 B 的预期收益。
    _v2c_run = {}
    for _t in tags:
        _prev = _v2c_run.get(vmap[_t])
        if _prev is not None and _prev != _t:
            print(_c("警告：变体 %s 同时被候选 %s 与 %s 指向，本次只跑 %s，"
                     "record 归因按后者，语义可疑——建议一次只跑一个共享变体的候选。"
                     % (vmap[_t], _prev, _t, _t), WARN), flush=True)
        _v2c_run[vmap[_t]] = _t
    globals()["_V2C"] = _v2c_run
    cmd = [sys.executable, str(bench),
           "--json", str(ab_json), "--rounds", str(a.rounds), *bench_args]
    if "--variants" not in bench_args:
        cmd += ["--variants", variants_arg]
    # 选一张空闲卡：不设 HIP_VISIBLE_DEVICES 会默认落到卡 0，
    # 而卡 0 常常被别人的大任务占满（实测 "0 bytes free"）。
    # **优先级：OPS_GPU 环境变量 > 卡的 defaults.gpu > 7**。
    # 环境变量优先是刻意的：评测/多任务并行时由外层调度器分配卡
    # （实测 2026-11-27：卡里写死 gpu: 7，会把并行实验全赶到卡 7，
    #  外层怎么绑卡都没用，跨臂对照直接混进卡差）。
    _req = str(os.environ.get("OPS_GPU") or d.get("gpu", "7"))
    # 逻辑索引 → 物理卡：外层若已用 HIP_VISIBLE_DEVICES 把进程限制到一张卡，
    # 模型看到的设备号是 0。直接把 0 写回 HIP_VISIBLE_DEVICES 会选中**物理卡 0**
    # （实测 2026-11-27：C 臂自己加 OPS_GPU=0，而它的进程可见的是物理卡 1，
    #  一跑就会落到 A 臂的卡上，跨臂污染）。
    _vis = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if _vis and _req.isdigit():
        _ids = [x.strip() for x in _vis.split(",") if x.strip()]
        if int(_req) < len(_ids):
            _req = _ids[int(_req)]
    gpu = _req
    env = dict(os.environ, HIP_VISIBLE_DEVICES=gpu)
    print("$ HIP_VISIBLE_DEVICES=%s %s" % (gpu, " ".join(cmd)), flush=True)
    # ---- 同一 job 重复测量（R1）----
    # 噪声底必须是**同一次测量的重复 spread**，不是跨轮比值的 max-min：
    # 后者在出现一次大改善后会把噪声底抬到几十个百分点，真实收益被 NEUTRAL。
    reps = max(1, int(getattr(a, "repeat", 1) or 1))
    ab = run_repeated(cmd, ab_json, card, reps, env)
    if ab is None:
        return 1

    # ---- 交付物落盘（--bench-std）----
    # 放在判定之前：一个变体被拒不代表这次测量无效，已测到的数仍应成为交付物。
    # 这是「引擎在主路上」的关键一环：评分只读 results/bench_std.json，
    # 而 loop.py 原先只写 runs/loop/*.jsonl，两者是两套 —— agent 自然先手搓
    # 一份能得分的，再把 loop.py 当打卡（2026-11-28 读 C 臂对话实证）。
    if getattr(a, "bench_std", None):
        _den = None
        try:
            _roof = card / "roofline.json"
            if _roof.is_file():
                _r = json.loads(_roof.read_text())
                _pb = (_r.get("ridge") or {}).get("peak_bw") or {}
                if _pb.get("id") and _pb.get("value") is not None:
                    _den = {"peak_id": _pb["id"], "value_gbs": _pb["value"]}
        except (OSError, json.JSONDecodeError, TypeError):
            _den = None
        # rounds 是本次实际测量轮数，交给交付物当 iters；
        # 不传的话 write_bench_std 只能写 1，读的人会以为是单次测的。
        ab["rounds"] = int(getattr(a, "rounds", 0) or 1)
        _p = write_bench_std(a.bench_std, card, ab,
                             {vmap[t] for t in tags}, denominator=_den)
        if _p:
            print("[bench_std] 交付物已生成：%s（%d 个形状档）"
                  % (_p, len(json.loads(_p.read_text()).get("shapes") or [])), flush=True)
        else:
            print(_c("[bench_std] 没写出来：ab 结果里没有可用的实测耗时"
                     "（所有变体都被跳过？）—— 交付物缺失，本轮不算完成", WARN), flush=True)

    acc = {}
    # 精度门是**可选适配器**。它的参数同样由卡片声明（acc_args），
    # 不能把 sparse-mla 的 --qscale/--skv 写死——那张卡的 qscale 是 logit 尺度，
    # 别的算子没有这个概念。原实现让 rmsnorm 这类卡要么报参数错、要么跑到错的档位。
    acc_adapter = d.get("acc", "bench/acc_probe.py")
    if not a.no_acc and (card / acc_adapter).exists():
        acc_json = db_dir(card) / "_acc.json"
        acc_args = d.get("acc_args")
        if acc_args is None:
            acc_args = ["--qscale", *[str(x) for x in (a.qscale or d.get("acc_qscale") or [1, 10])],
                        "--skv", str(skv[0])]
            print("note: 未声明 defaults.acc_args，退回 sparse-mla 默认（--qscale/--skv）",
                  flush=True)
        else:
            acc_args = [str(x) for x in acc_args]
        # **传变体名，不传候选 id**（2026-11-25 修）。记录里 acc 的键是
        # v["tag"]（变体名），而 a.tags 是候选 id —— 两者不同（候选 id ≠ 变体名，
        # 这个区别本文件 1340 行已经写过一次）。原来这里直接传 a.tags，
        # 于是 acc_probe 报的 tag 都对不上，acc 永远为空，acc_ok 恒 False。
        # 这和「全仓 0 条 acc_ok=true」是同一个根因：不是没测，是测了没接上。
        acc_tags = [vmap.get(t, t) for t in a.tags]
        cmd = [sys.executable, str(card / acc_adapter), *acc_tags,
               "--json", str(acc_json), *acc_args]
        print("$ " + " ".join(cmd), flush=True)
        if subprocess.call(cmd) == 0 and acc_json.exists():
            raw = json.loads(acc_json.read_text())
            for qs, node in (raw.get("qscale") or {}).items():
                for row in node.get("rows", []):
                    acc.setdefault(row["tag"], {})[qs] = row

    _tol_meta = load_tolerance_json(card)
    tol = a.tol or _tol_meta.get("atol") or load_contract_tol(card)
    primary = "%g" % float(a.primary_qscale or d.get("acc_primary_qscale") or 1.0)
    stress_mult = float(a.stress_mult or d.get("acc_stress_mult") or 1.5)
    # 校准记录不参与合计（否则全是 CALIBRATION_ONLY 时会显示初始值
    # ACCEPTED_BITEXACT —— 实测 2026-11-28 新卡第一条记录就是这样，误导性极强）。
    worst = None
    # 判定的"好坏"排序。ACCEPTED_NO_NUMERICS 必须排在 ACCEPTED_NUMERICS 之前
    # （没测精度 ≠ 精度合格，更不能和逐位一致同级）。
    order = ["ERROR", "REJECTED_ACCURACY", "REJECTED_NO_GAIN", "INCONCLUSIVE",
             "ACCEPTED_NO_NUMERICS", "ACCEPTED_NUMERICS_PENDING_E2E",
             "ACCEPTED_BITEXACT"]

    print("\n%-14s %-9s %-8s %s" % ("tag", "skv", "Δ%", "判定"))
    _best_tag, _best_ms = None, None
    _recs_by_tag = {}
    for run in ab["runs"]:
        for v in run["variants"]:
            decision, why = decide(
                {"median_ms": run["base_ms"], "mad_ms": run.get("base_mad", 0.0)},
                v, acc.get("base"), acc.get(v["tag"]), tol, primary, stress_mult,
                record_mode=ab.get("measure_mode"))
            # 显著性：与 decide 内部同一判据（3×max(MAD)），只是单独取出来给主张门用
            _sig, _ratio, _why = _significant(
                {"median_ms": run["base_ms"], "mad_ms": run.get("base_mad", 0.0)},
                {"median_ms": v.get("ms"), "mad_ms": v.get("mad", 0.0)})
            sig = bool(_sig)
            # shape_key 原先写死 "s%dk%d"（sparse-mla 的 S 与 KV）。
            # 改成：优先用适配器自报的 shape_key，否则用 sweep 值，都没有就用 "default"。
            shape_key = (ab.get("shape", {}).get("shape_key")
                         or ("s%dk%d" % (ab.get("shape", {}).get("S") or a.s or 0, run.get("skv", 0))
                             if run.get("skv") is not None else "default"))
            if decision == "ACCEPTED_NUMERICS":
                decision = "ACCEPTED_NUMERICS_PENDING_E2E"
            _cal = (declared.get(_cid_of(v["tag"])) or {}).get("purpose") == "calibration"
            if _cal:
                why = ("校准口径：base=correctness_only 参考实现，不声称收益"
                       "（原判定 %s，Δ=%.1f%%）"
                       % (decision, v.get("delta_pct") or 0.0))
                decision = "CALIBRATION_ONLY"
            if not _cal and (worst is None or order.index(decision) < order.index(worst)):
                worst = decision
            # 记录本轮最好的可接受实现，跑完后晋升为下一轮基线
            if (not _cal and str(decision).startswith("ACCEPTED")
                    and decision != "ACCEPTED_NO_NUMERICS"):
                if _best_ms is None or (v.get("ms") or 1e9) < _best_ms:
                    _best_tag, _best_ms = v["tag"], v.get("ms")
            _skv_disp = run.get("skv")
            print("%-14s %-9s %+8.1f %s  %s" % (v["tag"],
                  _skv_disp if _skv_disp is not None else "-", v.get("delta_pct", 0.0),
                                                decision, why))
            _rec = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card.name,
                "tag": v["tag"], "skv": run.get("skv"),
                "direction": (declared.get(_cid_of(v["tag"])) or {}).get("direction"),
                "shape": ab.get("shape", {}),
                "base_ms": run["base_ms"], "base_mad": run.get("base_mad", 0.0),
                "fm_ms": run.get("fm_ms"), "fm_ratio": run.get("fm_ratio"),
                "cand_ms": v.get("ms"), "mad": v.get("mad"),
                "delta_pct": v.get("delta_pct"), "ratio_vs_base": v.get("ratio_vs_base"),
                "maxdiff": v.get("maxdiff"), "rel": v.get("rel"),
                "repeat_ms": v.get("repeat_ms"),
                "repeat_spread": v.get("repeat_spread"),
                "acc": acc.get(v["tag"], {}), "tolerance": tol,
                "evidence_kind": _tol_meta.get("evidence_kind"),
                "tolerance_sha256": _tol_meta.get("sha256"),
                "decision": decision, "why": why,
                "base_role": ("correctness_only" if _cal else None),
                "forced": bool(a.force and v["tag"] in refuted),
                # 完成判定的裁决码：run 不跑 done_when（那是 verify 的事），记 None 让
                # 流水里字段恒在，消费方不必按记录类型分叉。
                "done_when_rc": None,
                # P0-1：测量模式必填。ab.py 的值原样搬运，缺失写 "unknown"。
                # 缺这个字段，结论的符号都不可信（R0 就是这么翻车的）。
                "measure_mode": str(ab.get("measure_mode") or "unknown"),
                # ---- 迭代链的 4 个字段（v4 §3）。少了它们，下一轮无法回读：
                # 不知道该方向预期多少（expected_effect）、为什么试（root_cause_signal）、
                # 按什么键检索（trigger_pattern）、算不算证伪（falsified）。
                # declared 的键是「候选 id」，而这里 v["tag"] 是「变体名」。
                # 所以先按 vmap 反查候选 id，否则 direction/expect 永远是 None
                # （第一版就是这样，字段进了 record 但全是空——比没进更糟，
                #  因为它看起来像"记录了"）。
                **({"expected_effect": (declared.get(_cid_of(v["tag"])) or {}).get("expect"),
                    "root_cause_signal": _bn_signal(card),
                    "trigger_pattern": {
                        "bound": bound,
                        "signal": _bn_signal(card),
                        "direction": (declared.get(_cid_of(v["tag"])) or {}).get("direction"),
                    }}),
                "falsified": _falsified_of(decision),
                # ---- 主张门的三项输入（run 只记录，不拒绝）
                "caliber": {
                    # adapter（bench/ab.py）的输出优先；它没写的才标 unstated。
                    # 字段名沿用 op-bench 契约：what / alloc / state / shape / device
                    "what": _caliber_get(ab.get("caliber") or {}, "what") or "unstated",
                    "alloc": _caliber_get(ab.get("caliber") or {}, "alloc"),
                    "state": _caliber_get(ab.get("caliber") or {}, "state") or "unstated",
                    "shape": shape_key,
                    "device": (ab.get("caliber") or {}).get("device") or "unstated",
                },
                "significant": bool(sig),
                # --no-acc 时 acc_ok 恒为 False：比较可以跑，主张不许下
                "acc_ok": bool(acc.get(v["tag"])) and not a.no_acc,
                # ---- 五项审计需要的三个字段（claim 会用）----
                # 实测 2026-11-28：`claim` 一上来就判引擎自己的记录 FAIL ——
                # 缺复跑命令 / 实现定位 / 环境指纹。我们要求 agent 留痕，
                # 自己的账本先得可审计。
                "impl": "%s/bench/ab.py:_measure  —  变体 %s"
                        % (card.name, v["tag"]),
                "command": " ".join(cmd),
                # adapter 的环境指纹原样保留：claim_gate 在 caliber 缺 device 时
                # 从这里回落取 gfx/device，避免位置不同就误判口径不全。
                "env": {**(ab.get("env") or {}), **_env_fingerprint()},
            }
            append_record(card, _rec)
            _recs_by_tag[v["tag"]] = _rec
    print("\n合计判定：%s" % (worst or "CALIBRATION_ONLY（只有校准记录，没有可声称的收益）"))
    promote_base(card, _best_tag, _best_ms, _recs_by_tag.get(_best_tag) or {})
    # ---- 报告（--report）：用最后一条记录 + 地板 + 下一步填成题面六问 ----
    if getattr(a, "report", None):
        _last = None
        try:
            _recs = [r for r in read_records(card) if "cand_ms" in r]
            _last = _recs[-1] if _recs else None
        except Exception:                       # noqa: BLE001
            _last = None
        _fl = None
        try:
            _fl = _nextstep().floor_state(card)  # 只为拿 floor 对象
        except Exception:                       # noqa: BLE001
            _fl = None
        if _last is not None:
            _plan = None
            try:
                _plan = _nextstep().plan_next(card)
            except Exception:                   # noqa: BLE001
                _plan = None
            try:
                _floor_obj = None
                try:
                    _floor_obj = _load_floor().resolve(card)
                except Exception:               # noqa: BLE001
                    _floor_obj = None
                _cmd = "python3 -I " + " ".join(
                    [p for p in sys.argv if p != "-I"])
                _p = write_report(a.report, card, _last, plan=_plan,
                                  floor=_floor_obj, cmdline=_cmd)
                print("[report] 已生成 %s" % _p, flush=True)
            except Exception as _e:              # noqa: BLE001
                print(_c("[report] 写失败：%s: %s" % (type(_e).__name__, _e), WARN),
                      flush=True)
    if worst is None:
        # 全是校准记录：没有可声称的收益，下一步是补一条真实候选
        print("下一步：这只是校准。按瓶颈与方向补一条**真实候选**"
              "（期望收益 > 噪声），再 loop.py run。")
    elif worst.startswith("ACCEPTED_NUMERICS"):
        print("下一步：这个改动不是逐位一致的，必须补端到端质量门（GPQA/HumanEval 一档）才能进生产。")
    elif worst == "ACCEPTED_BITEXACT":
        print("下一步：先给候选补 done_when，再 loop.py verify --tag <tag>（rc=0 才算完成），"
              "然后把它落进生产文件并复测。")
    return 0


def _env_fingerprint():
    """环境指纹：复现的前提。取不到就写 unknown，不编。"""
    import json as _json
    import platform
    fp = {"python": platform.python_version(), "host": platform.node()}
    try:
        import torch
        fp["torch"] = torch.__version__
        fp["hip"] = getattr(torch.version, "hip", None)
        fp["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:                            # noqa: BLE001
        fp["torch"] = "unavailable"
    try:
        import subprocess as _sp
        r = _sp.run(["hy-smi"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            fp["hy_smi_head"] = " ".join(r.stdout.split())[:200]
    except Exception:                            # noqa: BLE001
        pass
    return fp


def cmd_record(a):
    """把**agent 自己做的**实验记进账本。不判定快慢，只保证可审计。

    这是"自由迭代 + 强制留痕"：agent 用自己喜欢的方式测，
    但要想让结论成立，必须能通过 `claim` 的五项审计。
    """
    card = a.card
    if not a.command:
        print(_c("拒绝记录：缺 --command（复跑命令）。没有复跑方式的结果不可复现，"
                 "不能进账本。", WARN))
        return 2
    if not a.impl:
        print(_c("拒绝记录：缺 --impl（`<文件>:<入口>`）。复测方靠它找到实现。", WARN))
        return 2
    if a.mode not in ("event_isolated", "end_to_end", "back_to_back"):
        print(_c("拒绝记录：--mode 必须是 event_isolated / end_to_end / back_to_back。", WARN))
        return 2
    if a.ms is None or a.ms <= 0:
        print(_c("拒绝记录：缺 --ms（实测中位耗时，毫秒）。", WARN))
        return 2
    delta = None
    if a.base_ms:
        delta = (a.ms - a.base_ms) / a.base_ms * 100.0
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "card": card.name,
        "tag": a.tag,
        "skv": a.skv or 0,
        "direction": a.direction,
        "shape": {"shape": a.shape} if a.shape else {},
        "base_ms": a.base_ms, "base_mad": 0.0,
        "cand_ms": a.ms, "mad": a.mad or 0.0,
        "delta_pct": delta,
        "decision": "RECORDED_EXTERNAL",
        "why": "由 agent 自行测量后经 `record` 记账；快慢与合理性由 `claim` 审计",
        "measure_mode": a.mode,
        "impl": a.impl,
        "command": a.command,
        "env": _env_fingerprint(),
        "acc_ok": bool(a.max_ulp is not None and a.max_ulp <= (a.tol_ulp or 2)),
        "acc": ({"1": {"tag": a.tag, "max_ulp": a.max_ulp,
                       "cosine": a.cosine}} if a.max_ulp is not None else {}),
        "notes": a.notes,
        "recorded_by": "agent",
    }
    append_record(card, rec)
    print("已记账：%s / %s  %.4f ms（口径 %s）" % (card.name, a.tag, a.ms, a.mode))
    print("下一步：`loop.py -C %s claim --tag %s` 审计五项性质" % (card, a.tag))
    return 0


# claim 审计的五项（业主点名的五件事）
# 每项检查的**机器 key**：--json 输出与会话控制流只用它，不再解析中文文案
CLAIM_CHECK_KEYS = {
    "口径可 claim": "caliber",
    "可复现": "reproducible",
    "实现可定位": "impl",
    "环境已冻结": "env",
    "有精度证据": "evidence",
}

CLAIM_CHECKS = (
    ("口径可 claim", lambda r: str(r.get("measure_mode")) == "event_isolated",
     "measure_mode=event_isolated（back_to_back 永远不可 claim）"),
    ("可复现", lambda r: bool(r.get("command")),
     "记录里有完整复跑命令（--command）"),
    ("实现可定位", lambda r: bool(r.get("impl")),
     "记录里有 `<文件>:<入口>`（--impl），复测方不必猜"),
    ("环境已冻结", lambda r: isinstance(r.get("env"), dict) and bool(r.get("env"))
     and r["env"].get("torch") not in (None, "unavailable"),
     "记录里有环境指纹（torch/hip/device）"),
    ("有精度证据", lambda r: str(r.get("evidence_kind") or "") in
     ("independent_reference", "vendor_library"),
     "证据种类必须是 independent_reference / vendor_library；"
     "self-consistent-only（只有模型自写 judge）判 FAIL：无独立核验，不能声称数值正确"),
)


def cmd_claim(a):
    """审计一条记录是否满足五项性质。**任一项不过就 FAIL。**

    `--json` 输出 `{tag, found, verdict, exit_code, checks:[{key,name,passed,why}],
    reasons:[key...]}` —— 会话控制流（driver/iterate）只许读它，不再解析中文文案。
    """
    want_json = bool(getattr(a, "json", False))
    recs = [r for r in read_records(a.card) if r.get("tag") == a.tag]
    if not recs:
        if want_json:
            print(json.dumps({"tag": a.tag, "found": False, "verdict": "NO_RECORD",
                              "exit_code": 1, "checks": [], "reasons": ["no_record"]},
                             ensure_ascii=False))
        else:
            print(_c("没有 tag=%r 的记录 —— 先 run 或 record" % a.tag, WARN))
        return 1
    rec = recs[-1]
    checks, reasons, ok = [], [], True
    for name, fn, why in CLAIM_CHECKS:
        try:
            passed = bool(fn(rec))
        except Exception:                        # noqa: BLE001
            passed = False
        key = CLAIM_CHECK_KEYS.get(name, name)
        checks.append({"key": key, "name": name, "passed": passed, "why": why})
        if not passed:
            reasons.append(key)
            ok = False
    if want_json:
        print(json.dumps({"tag": a.tag, "found": True,
                          "verdict": "PASS" if ok else "FAIL",
                          "exit_code": 0 if ok else 1,
                          "checks": checks, "reasons": reasons},
                         ensure_ascii=False))
        return 0 if ok else 1
    print("审计 %s / %s（记录时间 %s）" % (a.card.name, a.tag, rec.get("ts")))
    print("")
    for c in checks:
        print("  %s %-12s %s" % ("✓" if c["passed"] else "✗", c["name"], c["why"]))
    print("")
    if ok:
        print("结论：**PASS** —— 这条结论可以进报告（五项性质齐全）")
        return 0
    print(_c("结论：**FAIL** —— 缺项不能靠文字补，补完重新 record", WARN))
    return 1


def cmd_report(a):
    # 只渲染性能实验记录：verify 记录没有 skv/cand_ms，混进来会把交付表打崩
    recs = [r for r in read_records(a.card) if "skv" in r and "cand_ms" in r]
    frags = _report_fragments()
    if not recs:
        print(_report_fill(frags["cr_empty"]))
        return 1
    best = {}
    for r in recs:
        key = (r["skv"], r["tag"])
        if key not in best or r["cand_ms"] < best[key]["cand_ms"]:
            best[key] = r
    rows, held = [], []
    for (skv, tag), r in sorted(best.items()):
        ok, reasons = claim_gate(r)
        (rows if ok or getattr(a, "all", False) else held).append((skv, tag, r, reasons))

    if rows:
        print(_report_fill(frags["cr_table"]))
        for skv, tag, r, reasons in rows:
            tail = r["decision"] if not reasons else "%s (held: %s)" % (
                r["decision"], ",".join(reasons))
            print(_report_fill(
                frags["cr_row"],
                skv=skv, tag=tag,
                base="%.4f" % r["base_ms"], cand="%.4f" % r["cand_ms"],
                delta="%+.1f%%" % r["delta_pct"],
                fm="%.2fx" % r["fm_ratio"] if r.get("fm_ratio") else "-",
                maxdiff="%.2e" % (r.get("maxdiff") or 0.0),
                tail=tail))

    if held:
        print(_report_fill(frags["cr_held"], n=len(held)))
        for skv, tag, r, reasons in held:
            print(_report_fill(frags["cr_held_item"],
                               tag=tag, skv=skv, reasons=", ".join(reasons)))
        print(_report_fill(frags["cr_help"]))
    return 0


def _yaml_scalar(s):
    return json.dumps(str(s), ensure_ascii=False)


def _section_span(text, key):
    """顶格段 `key:` 的 [start, end)。段尾 = 下一个顶格非空行。"""
    m = re.search(r"(?m)^%s:" % re.escape(key), text)
    if m is None:
        return None
    nxt = re.search(r"(?m)^\S", text[m.end():])
    return m.start(), (m.end() + nxt.start()) if nxt else len(text)


def _item_block(text, section, item_id):
    """段内 `  - id: <item_id>` 那一块的绝对 [start, end)。

    边界：下一个 `  - id: ` 或任何顶格非空行。**只在指定段里找**，
    避免 directions / candidates / refuted 三处同名 id 互相误伤。
    """
    sp = _section_span(text, section)
    if sp is None:
        return None
    seg = text[sp[0]:sp[1]]
    m = re.search(r"(?m)^  - id: %s[ \t]*$" % re.escape(item_id), seg)
    if m is None:
        return None
    rest = seg[m.end():]
    nxt = re.search(r"(?m)^(?:  - id: |\S)", rest)
    end = m.end() + (nxt.start() if nxt else len(rest))
    return sp[0] + m.start(), sp[0] + end


def _cand_block(text, tag):
    return _item_block(text, "candidates", tag)


def _write_candidates(p, text):
    """先校验再落盘：写坏了立刻炸，且不覆盖原文件。"""
    import yaml
    try:
        yaml.safe_load(text)
    except Exception as e:
        raise SystemExit("拒绝写入：生成的 YAML 无法解析（%s）。原文件未改动。" % e)
    p.write_text(text)


def _drop_refuted(text, tag):
    """删掉 refuted 段里同 id 的旧条目（保证判死是幂等的）。"""
    blk = _item_block(text, "refuted", tag)
    if blk is None:
        return text
    return text[:blk[0]] + text[blk[1]:]


def _scalar(v):
    """简单标识符不加引号（和手写风格一致），其余走 JSON 字符串。"""
    if isinstance(v, bool) or isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str) and re.match(r"^[A-Za-z_][\w.-]*$", v):
        return v
    return _yaml_scalar(v)


def _render_field(indent, key, value):
    pad = " " * indent
    if isinstance(value, (list, tuple)):
        out = "%s%s:\n" % (pad, key)
        for v in value:
            out += "%s  - %s\n" % (pad, _scalar(v))
        return out
    return "%s%s: %s\n" % (pad, key, _scalar(value))


def _field_span(seg, indent, key):
    """seg 内 `    key:` 这一段的 [start, end)；多行列表值（更深缩进的 `- ...`）一并算进来。"""
    lines = seg.splitlines(keepends=True)
    off = 0
    for i, ln in enumerate(lines):
        if re.match(r"^%s%s:" % (" " * indent, re.escape(key)), ln):
            j = i + 1
            while j < len(lines) and re.match(r"^ {%d,}\S" % (indent + 2), lines[j]):
                j += 1
            return off, off + sum(len(x) for x in lines[i:j])
        off += len(ln)
    return None


def _set_field(text, block, indent, key, value):
    """在 block 指向的条目里设置字段：有就整段替换（含多行值），没有就追加到最后。"""
    if block is None:
        return text
    s, e = block
    seg = text[s:e]
    if not seg.endswith("\n"):
        seg += "\n"
    rendered = _render_field(indent, key, value)
    span = _field_span(seg, indent, key)
    if span:
        seg = seg[:span[0]] + rendered + seg[span[1]:]
    else:
        seg = seg.rstrip("\n") + "\n" + rendered
    return text[:s] + seg + text[e:]


def _to_refuted(text, tag, why, kind, direction=None):
    """把一个条目从 candidates/directions 挪进 refuted —— **已判死只有这一个落点**。"""
    blk = _cand_block(text, tag)
    if blk:
        text = text[:blk[0]] + text[blk[1]:]
    text = _drop_refuted(text, tag)
    # 同一 id 只能在一个落点：从 stalled 转 refuted 时要把 stalled 的残影清掉，
    # 否则两处各说各话（一处说「结构变了值得留」，一处说「机制不成立」）。
    text = _drop_item(text, "stalled", tag)
    body = "  - id: %s\n    kind: %s\n" % (_scalar(tag), _scalar(kind))
    if direction:
        body += "    direction: %s\n" % _scalar(direction)
    body += "    why: %s\n    on: %s\n" % (_yaml_scalar(why), time.strftime("%Y-%m-%d"))
    m = re.search(r"(?m)^refuted:[ \t]*$\n?", text)
    if not m:
        text = text.rstrip("\n") + "\nrefuted:\n" + body
    else:
        text = text[:m.end()] + body + text[m.end():]
    return text.replace("refuted: []\n", "refuted:\n")


def _to_stalled(text, tag, why, what_changed, direction=None, ratio=None):
    """把条目挪进 stalled —— **变慢但结构变了，留作父代**。

    与 _to_refuted 的区别只有一条：stalled **不计方向预算**，所以方向不会被封。
    结构上必须记 `what_changed`（结构上变了什么），否则「变慢」就退化成
    「试过，慢」——那是本仓库明令禁止的写法，也没法当父代复用。
    """
    blk = _cand_block(text, tag)
    if blk:
        text = text[:blk[0]] + text[blk[1]:]
    text = _drop_item(text, "stalled", tag)
    text = _drop_item(text, "refuted", tag)
    body = "  - id: %s\n    kind: stalled\n" % _scalar(tag)
    if direction:
        body += "    direction: %s\n" % _scalar(direction)
    if ratio is not None:
        body += "    ratio_vs_parent: %s\n" % _yaml_scalar(str(ratio))
    body += "    what_changed: %s\n" % _yaml_scalar(what_changed)
    body += "    why: %s\n    on: %s\n" % (_yaml_scalar(why), time.strftime("%Y-%m-%d"))
    m = re.search(r"(?m)^stalled:[ \t]*$\n?", text)
    if not m:
        # stalled 段放在 refuted 之前（保持「越靠上越可用」的阅读顺序）
        anchor = re.search(r"(?m)^refuted:[ \t]*$", text)
        if anchor:
            text = text[:anchor.start()] + "stalled:\n" + body + text[anchor.start():]
        else:
            text = text.rstrip("\n") + "\nstalled:\n" + body
    else:
        text = text[:m.end()] + body + text[m.end():]
    return text.replace("stalled: []\n", "stalled:\n")


def _drop_item(text, section, tag):
    """删掉某段里同 id 的旧条目（保证落点切换是幂等的）。"""
    blk = _item_block(text, section, tag)
    if blk is None:
        return text
    return text[:blk[0]] + text[blk[1]:]


def _add_direction(text, dir_id):
    """direction-refuted --direction X 遇到未声明的方向：建条目（证伪事件本身就是事实）。"""
    entry = ("  - id: %s\n    statement: ''\n    status: open\n    refutes: 0\n"
             % _scalar(dir_id))
    sp = _section_span(text, "directions")
    if sp is not None:
        m = re.search(r"(?m)^directions:[ \t]*$", text)
        return text[:m.end() + 1] + entry + text[m.end() + 1:]
    anchor = _section_span(text, "candidates") or _section_span(text, "refuted")
    if anchor is None:
        return text.rstrip("\n") + "\ndirections:\n" + entry
    return text[:anchor[0]] + "directions:\n" + entry + "\n" + text[anchor[0]:]


def cmd_impl_broken(a):
    """实现级失败：留在 optimize 修，**不消耗方向预算**；连挂 N 次该实现判死。"""
    card = a.card
    p = card / "candidates.yaml"
    cands = load_candidates(card)
    cand = next((c for c in cands["candidates"]
                 if isinstance(c, dict) and c.get("id") == a.tag), None)
    if cand is None:
        print(_c("拒绝：%s 不在 candidates 段里（先把它写进队列，再谈失败次数）" % a.tag, WARN))
        return 2
    fails = list(cand.get("fails") or [])
    fails.append("%s %s" % (time.strftime("%Y-%m-%d"), a.why))
    n = len(fails)
    direction = cand.get("direction")
    text = p.read_text()
    text = _set_field(text, _cand_block(text, a.tag), 4, "fails", fails)
    dead = n >= MAX_IMPL_FAILS
    if dead:
        text = _to_refuted(text, a.tag, "连挂 %d 次：%s" % (n, "；".join(fails)), "impl", direction)
    _write_candidates(p, text)
    append_record(card, {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card.name, "tag": a.tag,
        "direction": direction, "decision": "IMPL_BROKEN", "n": n,
        "exhausted": dead, "why": a.why,
    })
    if dead:
        print("%s 连挂 %d 次 ⇒ 这个实现判死（已进 refuted，kind=impl）。" % (a.tag, n))
        if direction:
            print("方向 %s 的预算**没有**被消耗 —— 换个实现重来，别换方向。" % direction)
    else:
        print("%s 记一次实现失败（%d/%d）：修实现；方向预算没动。"
              % (a.tag, n, MAX_IMPL_FAILS))
        if n + 1 >= MAX_IMPL_FAILS:
            print(_c("下一次再挂就判死这个实现。", WARN))
    return 0


def cmd_direction_refuted(a):
    """方向级失败：回 op-constrain 重定约束；证伪 N 次封方向，拒绝再受理同类候选。"""
    card = a.card
    p = card / "candidates.yaml"
    cands = load_candidates(card)
    cand = None
    if a.tag:
        cand = next((c for c in cands["candidates"]
                     if isinstance(c, dict) and c.get("id") == a.tag), None)
    dir_id = a.direction or (cand or {}).get("direction")
    if not dir_id:
        if a.tag and cand is None:
            print(_c("拒绝：%s 不在 candidates 段里，且没给 --direction" % a.tag, WARN))
        else:
            print(_c("拒绝：这个候选没声明 direction，方向预算无从计。"
                     "补 direction: 或直接给 --direction <id>", WARN))
        return 2

    table = direction_table(cands)
    text = p.read_text()
    info = table.get(str(dir_id))
    if info is None:
        text = _add_direction(text, dir_id)
        refutes, status = 0, "open"
        print("directions 段里没有 %s，已建条目（statement 待补）" % dir_id)
    else:
        refutes, status = info["refutes"], info["status"]
    refutes += 1
    if refutes >= MAX_DIRECTION_REFUTES:
        status = "exhausted"
    text = _set_field(text, _item_block(text, "directions", str(dir_id)), 4, "refutes", refutes)
    text = _set_field(text, _item_block(text, "directions", str(dir_id)), 4, "status", status)
    if cand is not None:
        text = _to_refuted(text, a.tag, a.why, "direction", str(dir_id))
    _write_candidates(p, text)
    append_record(card, {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card.name, "tag": a.tag,
        "direction": str(dir_id), "decision": "DIRECTION_REFUTED",
        "refutes": refutes, "exhausted": refutes >= MAX_DIRECTION_REFUTES, "why": a.why,
    })
    print("方向 %s 已证伪 %d/%d —— %s" % (dir_id, refutes, MAX_DIRECTION_REFUTES, a.why))
    if refutes >= MAX_DIRECTION_REFUTES:
        print(_c("方向 %s 已封：loop.py 不再受理 direction=%s 的任何候选。"
                 "下一个候选必须来自 op-constrain 重定后的约束。" % (dir_id, dir_id), WARN))
    else:
        print("回 op-constrain：这条约束不成立，重定一个再开候选。")
    return 0


def cmd_stalled(a):
    """候选变慢但结构变了：留作父代，**不耗方向预算、不封方向**。

    要求 `--what-changed` 写清结构上变了什么。没有它，「变慢」就退化成
    「试过，慢」——既违反本仓库的结论纪律，也没法被后续候选当父代复用。
    """
    card = a.card
    p = card / "candidates.yaml"
    cands = load_candidates(card)
    what_changed = str(getattr(a, "what_changed", "") or "").strip()
    if not what_changed:
        print(_c("拒绝：stalled 必须带 --what-changed（结构上变了什么）。"
                 "只写「变慢」等于「试过，慢」，既不能当父代也没法迁移。", WARN))
        return 2
    cand = None
    if a.tag:
        cand = next((c for c in cands["candidates"]
                     if isinstance(c, dict) and c.get("id") == a.tag), None)
    direction = a.direction or (cand or {}).get("direction")

    text = p.read_text()
    text = _to_stalled(text, a.tag, a.why, what_changed, direction,
                       ratio=getattr(a, "ratio", None))
    _write_candidates(p, text)
    append_record(card, {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card.name, "tag": a.tag,
        "direction": direction, "decision": "STALLED",
        "what_changed": what_changed, "why": a.why,
        "ratio_vs_parent": getattr(a, "ratio", None),
    })
    print("候选 %s 已记入 stalled（结构变了：%s）" % (a.tag, what_changed))
    print("  → 不耗方向预算。它可以被当成后续候选的 parent："
          "loop.py 不限制从 stalled 节点再出发。")
    if direction:
        table = direction_table(load_candidates(card))
        info = table.get(str(direction)) or {}
        print("  → 方向 %s 的计数不变（仍是 %s/%d）"
              % (direction, info.get("refutes", "?"), MAX_DIRECTION_REFUTES))
    return 0


def cmd_refute(a):
    print(_c("refute 已拆成两条（P0-3）：", WARN))
    print("  impl-broken <tag>        这次实现没写成/没跑对，留在 optimize 修，不耗方向预算")
    print("  direction-refuted <tag>  跑通了但这套攻击面不行，回 constrain 重定约束")
    return 2


def adopted_text(text, tag):
    """把候选条目的 status 改成 adopted（纯文本改写，落盘前由 _write_candidates 校验）。"""
    blk = _cand_block(text, tag)
    if blk:
        seg = text[blk[0]:blk[1]]
        if re.search(r"^    status: .*$", seg, re.M):
            seg = re.sub(r"^    status: .*$", "    status: adopted", seg, count=1, flags=re.M)
        else:
            seg = seg.rstrip("\n") + "\n    status: adopted"
        seg = re.sub(r"^    adopted_on: .*\n", "", seg, flags=re.M)
        seg = seg.rstrip("\n") + "\n    adopted_on: %s\n" % time.strftime("%Y-%m-%d")
        return text[:blk[0]] + seg + text[blk[1]:]
    m = re.search(r"^candidates:[ \t]*$\n?", text, re.M)
    entry = ("  - id: %s\n    status: adopted\n    adopted_on: %s\n"
             % (_scalar(tag), time.strftime("%Y-%m-%d")))
    if not m:
        return text.rstrip("\n") + "\ncandidates:\n" + entry
    return text[:m.end()] + entry + text[m.end():]


def run_done_when(cmd, card):
    """执行候选的 done_when。返回 (rc, 合并后的 stdout+stderr)。

    cwd = 仓库根（`scripts/opcheck.py` 这类共享脚本按仓库根相对路径写才对）；
    卡片目录用环境变量 `OPS_CARD` 暴露给命令，仓库根是 `OPS_REPO_ROOT`。
    超时按失败算（rc=124）——「跑不完」不能读成「通过」。
    """
    env = dict(os.environ)
    env["OPS_CARD"] = str(card)
    env["OPS_REPO_ROOT"] = str(REPO_ROOT)
    try:
        cp = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT), env=env,
                            capture_output=True, text=True, timeout=VERIFY_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        def _s(x):
            if isinstance(x, bytes):
                return x.decode("utf-8", "replace")
            return x or ""
        return 124, ("done_when 超过 %ss 未结束：%s\n%s"
                     % (VERIFY_TIMEOUT_S, cmd, _s(e.stdout) + _s(e.stderr)))
    return cp.returncode, (cp.stdout or "") + (cp.stderr or "")


def cmd_verify(a):
    """完成判定：**只有 done_when 返回 0 才允许 adopted**。

    失败时保持原状态（不改 YAML），只回显尾部 20 行 + 记 `done_when_rc`——
    「我看日志觉得它没问题」不是通过条件。
    """
    card = a.card
    cands = load_candidates(card)
    cand = next((c for c in cands["candidates"]
                 if isinstance(c, dict) and c.get("id") == a.tag), None)
    if cand is None:
        print(_c("拒绝：%s 不在 candidates 段里（没有候选就没有完成判定可谈）" % a.tag, WARN))
        return 2
    cmd = str(cand.get("done_when") or "").strip()
    if not cmd:
        print(_c("拒绝：%s 没有 done_when —— 完成判定不能由模型口述。先在 candidates.yaml "
                 "给它写一条命令（写法见 op-optimize/SKILL.md §七）" % a.tag, WARN))
        return 2

    # 主张门：adopted 是在主张「这个改动可交付」，必须有够格的记录支撑
    recs = [r for r in read_records(card) if r.get("tag") == a.tag and "cand_ms" in r]
    if recs:
        ok, reasons = claim_gate(recs[-1])
        if not ok:
            print(_c("拒绝 adopted：%s 的最新记录未达主张门槛（%s）。"
                     "用 report --all 看补齐方法；补齐后重跑 verify。"
                     % (a.tag, ", ".join(reasons)), WARN))
            return 2
    elif not getattr(a, "allow_no_record", False):
        print(_c("拒绝 adopted：%s 没有任何 run 记录 —— adopted 是「已测且够格」，"
                 "不是「我觉得行」。先 loop.py run %s。" % (a.tag, a.tag), WARN))
        return 2

    print("$ " + cmd, flush=True)
    rc, out = run_done_when(cmd, card)
    tail = "\n".join(out.splitlines()[-VERIFY_TAIL_LINES:])
    if rc != 0:
        if tail:
            print(tail)
        print(_c("verify 失败（done_when rc=%d）：%s **保持原状态**（%s）。"
                 "完成判定没通过，不许写 adopted。"
                 % (rc, a.tag, cand.get("status") or "pending"), WARN))
        if str(cand.get("status") or "") == "adopted":
            print(_c("caveat 它现在是 adopted 但自己的 done_when 不过 —— 这条 adopted 不成立，"
                     "该降级并查为什么之前判过了。", WARN))
        append_record(card, {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card.name, "tag": a.tag,
            "direction": cand.get("direction"), "decision": "VERIFY_FAILED",
            "done_when": cmd, "done_when_rc": rc,
            "status": str(cand.get("status") or "pending"),
        })
        return 1

    cur = str(cand.get("status") or "pending")
    if cur == "verified":
        # verified 是另一个终结态（知识卡/事实条目），复验通过只留记录，不改状态词
        print("%s 已是 verified；done_when 返回 0（复验通过），状态保持 verified" % a.tag)
    else:
        p = card / "candidates.yaml"
        _write_candidates(p, adopted_text(p.read_text(), a.tag))
        cur = "adopted"
        print("%s 的 done_when 返回 0 ⇒ status=adopted（完成判定由命令裁决）" % a.tag)
    append_record(card, {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card.name, "tag": a.tag,
        "direction": cand.get("direction"), "decision": "VERIFIED",
        "done_when": cmd, "done_when_rc": 0, "status": cur,
    })
    return 0


def cmd_adopt(a):
    """旧名，等价于 verify：不跑命令就 adopted 的路已经关掉。"""
    print(_c("adopt 已并入 verify（同一个动作）：完成判定由 done_when 裁决，不由模型口述。", WARN))
    return cmd_verify(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-C", "--card", required=True, help="算子卡目录（含 candidates.yaml）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="当前最强 + 待做 + 记录数")
    sub.add_parser("queue", help="打印候选队列与已证伪清单")
    rp = sub.add_parser("report", help="从记录渲染交付表（默认只出 claimable）")
    rp.add_argument("--all", action="store_true", help="也显示未达主张门槛的记录")

    e = sub.add_parser("enumerate", help="列出攻击面（direction 只能从这挑），标出已封的")
    e.add_argument("--new", action="store_true", help="只打印可用的 candidates.yaml 片段")
    e.add_argument("--scenario", default=None, help="按场景给攻击面优先序")
    e.add_argument("--list-scenarios", action="store_true", help="列出所有场景")

    rs = sub.add_parser("resume", help="上下文丢了以后：下一个该跑谁")
    rs.add_argument("--json", action="store_true", help="机器可读输出")

    rd = sub.add_parser("record", help="把 agent 自己测的实验记进账本（带口径/命令/环境）")
    rd.add_argument("--tag", required=True)
    rd.add_argument("--impl", required=True, help="`<文件>:<入口>`")
    rd.add_argument("--ms", type=float, required=True, help="实测中位耗时（毫秒）")
    rd.add_argument("--base-ms", type=float, default=None, dest="base_ms")
    rd.add_argument("--mad", type=float, default=None)
    rd.add_argument("--mode", default="event_isolated",
                    choices=["event_isolated", "end_to_end", "back_to_back"])
    rd.add_argument("--command", required=True, help="完整复跑命令")
    rd.add_argument("--max-ulp", type=float, default=None, dest="max_ulp")
    rd.add_argument("--tol-ulp", type=float, default=2.0, dest="tol_ulp")
    rd.add_argument("--cosine", type=float, default=None)
    rd.add_argument("--shape", default=None)
    rd.add_argument("--skv", type=int, default=None)
    rd.add_argument("--direction", default=None)
    rd.add_argument("--notes", default=None)

    cl = sub.add_parser("claim", help="审计一条记录是否满足五项性质")
    cl.add_argument("--tag", required=True)
    cl.add_argument("--json", action="store_true",
                    help="机器可读输出（{tag,found,verdict,exit_code,checks,reasons}）")

    nx = sub.add_parser("next", help="下一步做什么（给 agent 的决策参考，不是命令）")
    nx.add_argument("--json", action="store_true", help="机器可读输出")

    r = sub.add_parser("run", help="跑一个实验并判定")
    r.add_argument("tags", nargs="+")
    r.add_argument("--skv", type=int, nargs="+")
    r.add_argument("--s", type=int)
    r.add_argument("--topk", type=int)
    r.add_argument("--nvalid", type=int)
    r.add_argument("--heads", type=int)
    r.add_argument("--qscale", type=float, nargs="+")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--repeat", type=int, default=1,
                   help="同一 job 重复测量几次；噪声底取这几次的 spread（R1）")
    r.add_argument("--tol", type=float, help="对 fp32 参考的 max_abs 容差（默认读 contract.yaml）")
    r.add_argument("--primary-qscale", type=float, help="主口径 logit 尺度（默认 1；压力档只报退化倍数）")
    r.add_argument("--stress-mult", type=float, help="压力档允许的相对退化倍数（默认 1.5）")
    r.add_argument("--no-acc", action="store_true", help="跳过精度探针（只在你确定逐位一致时用）")
    r.add_argument("--report", default=None, metavar="PATH",
                   help="按题面六问生成实测报告 report.md（用实测数据填，不编）")
    r.add_argument("--bench-std", default=None, metavar="PATH",
                   help="把本次实测写成 opbench 口径的 results/bench_std.json。"
                        "**这是交付物的唯一合法来源**：手写的同形状文件不满足"
                        "「由引擎驱动」的可复现契约")
    r.add_argument("--force", action="store_true", help="允许重跑已证伪的变体")

    i = sub.add_parser("impl-broken", help="实现级失败：修实现，不耗方向预算")
    i.add_argument("tag")
    i.add_argument("--why", required=True)

    d = sub.add_parser("direction-refuted", help="方向级失败：回 constrain 重定约束")
    d.add_argument("tag", nargs="?")
    d.add_argument("--direction", help="直接点名方向（候选已不在队列里时用）")
    d.add_argument("--why", required=True)

    st = sub.add_parser("stalled",
                        help="变慢但结构变了：留作父代，不计方向预算、不封方向")
    st.add_argument("tag")
    st.add_argument("--why", required=True, help="为什么这次更慢（机制）")
    st.add_argument("--what-changed", required=True, dest="what_changed",
                    help="结构上变了什么（父代复用要看这个）")
    st.add_argument("--ratio", help="相对 parent 的耗时比（>1 表示更慢），有就写")
    st.add_argument("--direction", help="候选已不在队列里时直接点名方向")

    f = sub.add_parser("refute", help=argparse.SUPPRESS)   # 已拆分，保留只为给出迁移提示
    f.add_argument("tag", nargs="?")
    f.add_argument("--why")

    v = sub.add_parser("verify", help="跑候选的 done_when；rc=0 才允许 adopted")
    v.add_argument("--tag", required=True)
    v.add_argument("--allow-no-record", action="store_true",
                   help="允许没有 run 记录也 verify（仅用于纯构建/契约类候选）")

    a2 = sub.add_parser("adopt", help="旧名，等价于 verify（完成判定由命令裁决）")
    a2.add_argument("tag")

    a = ap.parse_args()
    a.card = Path(a.card).resolve()
    return {"status": cmd_status, "next": cmd_next, "record": cmd_record,
            "claim": cmd_claim, "queue": cmd_queue, "run": cmd_run,
            "report": cmd_report, "enumerate": cmd_enumerate,
            "resume": cmd_resume,
            "impl-broken": cmd_impl_broken,
            "direction-refuted": cmd_direction_refuted,
            "stalled": cmd_stalled,
            "refute": cmd_refute, "verify": cmd_verify,
            "adopt": cmd_adopt}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
