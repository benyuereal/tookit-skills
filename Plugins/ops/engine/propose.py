# -*- coding: utf-8 -*-
"""propose —— 信号 → 候选（「拟做」，不是「待做」）。

## 这里的规则为什么长这样

每条规则都必须能回答三问：**看到什么、指向哪个攻击面、怎么验收**。
答不出来的方向，提出来只会浪费一次方向预算（同方向证伪 2 次就封）。

## 它不绕过任何一道门

* 生成的候选 `status: proposed`，**不是** `pending`；
* `loop.py run` 仍然拒绝它（变体确实还不存在，冒名跑等于造假账）；
* `expect` 一律 `0%` + `expect_basis`，不许编一个百分比去搅乱 EV 排序；
* 方向**只允许**来自 `references/surfaces.yaml`，不许发明（forge_next 同规矩）。

## 为什么和 probe 分家

probe 取事实、propose 消费事实。混在一起时，「这张卡为什么没出候选」要读
600 行；分开之后，要改规则只动这个文件，要加信号只动 probe。
"""
from __future__ import annotations

# 通用信号的达成率阈值。**这些是经验值，写在这里是为了可被审阅和改写**，
# 不是物理常数——所以每条规则输出里都带上用的哪个阈值。
BW_LOW = 0.60        # 低于这个：多半是多趟 HBM / 向量宽度不够
BW_MID = 0.85        # 高于这个：已经贴近地板，别再折腾 kernel 内部
SERIAL_RATIO = 1.30  # 串行地板 / 重叠地板 超过这个 = 并行度没吃到


def make(sig, n, direction, action_class, hypothesis, mechanism,
         expect_basis, todo, extra=None):
    """造一条**拟做**候选。status 一律 proposed —— 见模块说明。

    `n` 只用于 id 序号（`auto-<direction>-<n>`），不参与排序；排序看 priority。
    """
    d = {
        "id": "auto-%s-%d" % (direction, n),
        "direction": direction,
        "status": "proposed",
        "priority": n,
        "action_class": action_class,
        "hypothesis": hypothesis,
        "expect": "0%",
        "expect_basis": expect_basis,
        "expect_confidence": "low",
        "note": mechanism,
        "todo": todo,
        "signal": {k: sig.get(k) for k in
                   ("bound", "efficiency", "cta_per_cu", "rounds", "lds_ratio",
                    "serial_ratio", "achieved_us", "ceiling_us")},
    }
    if extra:
        d.update(extra)
    return d


def propose(sig):
    """信号 → 候选。**只出方向，不出变体**；变体由人/agent 实现后改成 pending。

    返回 (proposals, skipped)。skipped 是「为什么没出题」的逐条原因 ——
    它和 proposals 一样重要：**没有候选时，用户唯一需要知道的就是为什么**。
    """
    out, skipped = [], []
    surfs = set(sig.get("surfaces") or [])
    budget = sig.get("budget") or {}

    def allow(direction):
        """方向必须合法且还有预算。这两条任一不满足就不出——不是"少一条"，是"别提"。"""
        if direction not in surfs:
            skipped.append("%s：不在 surfaces.yaml 白名单" % direction)
            return False
        if budget.get(direction) is not None and budget[direction] <= 0:
            skipped.append("%s：方向预算已用完（2 次封方向）" % direction)
            return False
        return True

    n = 0

    def add(*a, **kw):
        nonlocal n
        n += 1
        out.append(make(sig, n, *a, **kw))

    # ── 规则 0：没有可 claim 的实测 ⇒ 第一条候选就是"把基线量一次" ──────────
    # 这不是凑数。空白卡没候选⇒没跑过⇒没达成率⇒本该出的候选全出不来，是自锁。
    # 破它的正道不是猜一个方向，而是先测。引擎已经支持 purpose=calibration
    # （loop.py 会判 CALIBRATION_ONLY，不声称收益），这里只是把这条用起来。
    if not sig.get("achieved_us"):
        vs = sig.get("variants")
        if vs:
            add("occupancy", "measure-baseline",
                "这张卡在当前档位还没有可 claim 的实测（runs/loop 为空），"
                "所有基于达成率的判断都无从谈起；先把现成实现量一次档位基线",
                "用已有实现（%s）跑一次 event_isolated，拿到 achieved 与达成率，"
                "再让后续规则有数可用" % " / ".join(vs[:3]),
                "基线缺失，先校准；变体来自 %s，未做任何改动" % sig.get("variants_src"),
                ["把 status 改成 pending 并补 variant: %s" % vs[0],
                 "补 done_when（opcheck correctness）"],
                extra={"purpose": "calibration"})
        else:
            skipped.append("没有实测、也没有已实现变体可作基线——%s"
                           % (sig.get("variants_src") or ""))

    if not sig.get("roof") and not sig.get("floor_src"):
        skipped.append("既没有 derive 模型也没有 floor.py 地板，除校准外的规则不启用：%s"
                       % "; ".join(sig.get("missing") or []))
        return out, skipped

    # floor.py 说已经到了该停的地方，就别再出候选 —— 这是它比 derive 多的信息
    if sig.get("gate") and sig.get("gate") != "headroom":
        skipped.append("floor.py 判定 gate=%s（%s）：%s"
                       % (sig.get("gate"), sig.get("remaining"),
                          sig.get("floor_advice") or "不再出候选"))
        return out, skipped

    # 几何类规则（1-4）只有 derive 模型才给得出；没有就明说，而不是静默跳过
    if not sig.get("roof") and sig.get("floor_src") == "floor.py":
        skipped.append("地板来自 floor.py（按访问模式算），没有 cta/cu、rounds、lds、"
                       "串行比这些几何量 —— 几何类规则（prefetch/occupancy/grid-tail/"
                       "lds）本轮不启用，只按达成率出题")

    eff = sig.get("efficiency")
    bound = sig.get("bound")

    # ── 规则 1：串行地板远高于重叠地板 ⇒ 计算和数据搬运没重叠起来 ──────────
    sr = sig.get("serial_ratio")
    if sr and sr >= SERIAL_RATIO and allow("prefetch-depth"):
        add("prefetch-depth", "improve-overlap",
            "serial/overlap = %.2f ≥ %.2f，说明数据搬运与计算是串行的，"
            "吃到的是 serial 地板（%.1fµs）而不是 overlap 地板（%.1fµs）"
            % (sr, SERIAL_RATIO, sig["floors_us"].get("serial", 0),
               sig["floors_us"].get("overlap", 0)),
            "把下一块的 gather/load 提前到当前块的计算之前；"
            "hw_kb 的 reg_resident_vs_lds 说明驻留方式决定能不能重叠",
            "由 derive 的 serial/overlap=%.2f 推出，尚未实测" % sr,
            ["实现一个 prefetch 变体", "补 done_when（opcheck correctness）"])

    # ── 规则 2：每个 CU 摊不到 1 个 CTA ⇒ 占用率不足 ───────────────────────
    cpu = sig.get("cta_per_cu")
    if isinstance(cpu, (int, float)) and cpu < 1 and allow("occupancy"):
        add("occupancy", "raise-parallelism",
            "cta_per_cu = %.2f < 1：CTA 总数 %s 填不满 CU，SM 有闲的"
            % (cpu, sig.get("ctas")),
            "把块切小 / 增加并行维度让 CTA 数 ≥ CU 数；warp_reduce_shfl 可用来说明"
            "跨 lane 归约不该再切块",
            "由 derive 的 geometry.cta_per_cu=%.2f 推出，尚未实测" % cpu,
            ["实现一个更细的切块变体", "补 done_when"])

    # ── 规则 3：grid 有多轮 ⇒ 尾轮不满 ─────────────────────────────────────
    r = sig.get("rounds")
    if isinstance(r, (int, float)) and r > 1 and allow("grid-tail"):
        add("grid-tail", "reduce-tail",
            "geometry.rounds = %s > 1：grid 要多轮才铺完，尾轮有 CTA 空转" % r,
            "让 grid 规模整除 CU 数，或把尾块合并进上一轮",
            "由 derive 的 geometry.rounds 推出，尚未实测",
            ["实现一个整除 CU 的切块", "补 done_when"])

    # ── 规则 4：LDS 预算吃紧 ⇒ num_stages 上不去 ───────────────────────────
    lr, lok = sig.get("lds_ratio"), sig.get("lds_ok")
    if ((isinstance(lr, (int, float)) and lr >= 0.80) or lok is False) and allow("lds-read-form"):
        add("lds-read-form", "reduce-lds",
            "LDS 占用比例 %s（lds_ok=%s）：shared memory 预算吃掉了流水深度"
            % (lr, lok),
            "改读形式省 LDS（例如直接读 VMEM 或复用寄存器）；"
            "reg_resident_vs_lds 里记着两条路线的取舍",
            "由 derive 的 geometry.lds_ratio=%s / lds_ok=%s 推出，尚未实测"
            % (lr, lok),
            ["实现一个省 LDS 的变体", "补 done_when"])

    # ── 规则 5/6/7：按达成率分档（memory bound 才有意义）───────────────────
    if bound == "memory" and isinstance(eff, (int, float)):
        if eff >= BW_MID:
            skipped.append("达成率 %.0f%% ≥ %.0f%%：已贴地板，不再出候选"
                           "（ceiling %.1fµs vs 实测 %.1fµs）"
                           % (eff * 100, BW_MID * 100,
                              sig.get("ceiling_us") or 0, sig.get("achieved_us") or 0))
        elif eff >= BW_LOW:
            for dr in ("slc-policy", "prefetch-depth"):
                if allow(dr):
                    add(dr, "improve-locality",
                        "memory bound，达成率只有 %.0f%%（%.1fµs vs 地板 %.1fµs），"
                        "差额多半在 L2/SLC 命中与预取深度上"
                        % (eff * 100, sig.get("achieved_us") or 0,
                           sig.get("ceiling_us") or 0),
                        "slc_cache_hint 与 vec16_uint4 说明了缓存驻留提示与向量宽度"
                        "各自能省多少往返",
                        "由达成率 %.0f%% 落在 [%.0f%%, %.0f%%) 推出，尚未实测"
                        % (eff * 100, BW_LOW * 100, BW_MID * 100),
                        ["实现一个 cache hint / 预取变体", "补 done_when"])
                    break
        else:
            for dr in ("vmem-vector-width", "layout-pack"):
                if allow(dr):
                    add(dr, "reduce-bytes",
                        "memory bound 但达成率只有 %.0f%%（%.1fµs vs 地板 %.1fµs）："
                        "典型原因是多趟 HBM 或向量宽度不够"
                        % (eff * 100, sig.get("achieved_us") or 0,
                           sig.get("ceiling_us") or 0),
                        "vec16_uint4 / store_width 说明了向量宽度对带宽的直接影响",
                        "由达成率 %.0f%% < %.0f%% 推出，尚未实测"
                        % (eff * 100, BW_LOW * 100),
                        ["实现一个向量化更宽的变体", "补 done_when"])
                    break
    elif bound == "memory":
        skipped.append("memory bound 但拿不到实测耗时（runs/loop 里没有可 claim 的记录），"
                       "算不出达成率 —— 先跑一次基线，别凭猜出候选")

    if not out:
        skipped.append("信号不足以支撑任何一条候选。这不是失败：说不出为什么的方向，"
                       "提出来只会浪费一次方向预算")
    return out, skipped


def render(sig, props, skipped):
    """人读输出。**这是对外文本契约**：字段顺序与措辞被测试和文档引用，别顺手改。"""
    L = []
    L.append("=" * 78)
    L.append("explore — 从通用信号出题")
    L.append("=" * 78)
    roof = sig.get("roof") or {}
    L.append("卡片 %s   档位 %s" % (sig.get("card"), sig.get("shape")))
    if roof:
        fl = sig.get("floors_us") or {}
        L.append("地板  内存 %.1fµs / 算力 %.1fµs / 重叠 %.1fµs / 串行 %.1fµs  支配项=%s"
                 % (fl.get("memory", 0), fl.get("compute", 0),
                    fl.get("overlap", 0), fl.get("serial", 0), sig.get("bound")))
        L.append("几何  CTAs=%s  cta/cu=%s  rounds=%s  lds=%.2f  serial_chain=%s"
                 % (sig.get("ctas"), sig.get("cta_per_cu"), sig.get("rounds"),
                    sig.get("lds_ratio") or 0, sig.get("serial_chain")))
        L.append("达成  实测 %s  天花板 %.1fµs  达成率 %s"
                 % ("%.1fµs" % sig["achieved_us"] if sig.get("achieved_us") else "（无）",
                    sig.get("ceiling_us") or 0,
                    "%.0f%%" % (sig["efficiency"] * 100) if sig.get("efficiency") else "（算不出）"))
    if sig.get("floor_src") == "floor.py":
        L.append("地板来源 floor.py：%s" % (sig.get("floor_basis") or ""))
    for m in sig.get("missing") or []:
        L.append("缺  %s" % m)
    L.append("")
    if props:
        L.append("拟做 %d 条（**不是**待做：变体还没实现，loop.py run 会拒绝，这是对的）："
                 % len(props))
        for p in props:
            L.append("  [%d] %-24s direction=%-20s action=%s"
                     % (p["priority"], p["id"], p["direction"], p["action_class"]))
            L.append("      %s" % p["hypothesis"])
            L.append("      basis: %s" % p["expect_basis"])
            L.append("      todo : %s" % "；".join(p.get("todo") or []))
    else:
        L.append("拟做 0 条。")
    if skipped:
        L.append("")
        L.append("没出的原因：")
        for s in skipped:
            L.append("  - %s" % s)
    return "\n".join(L)
