#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ingest_check.py 与 schema.yaml 的测试：守卫本身必须被守卫。

test_ingest.py 守提取器（抽得对、不编造、人类字段不被机器填）；本文件守
**校验器**和**schema 契约**，因为一个只放行的校验器等于没有校验器。

旧 verification.json 的 7 条自检一条不能少；新 records.jsonl 的每条判据都
造一份该被拒的输入，断言它真被拒、理由指向对应字段。
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CHECK = REPO / "scripts/ingest_check.py"
SCHEMA = REPO / "Plugins/ops/ingest/schema.yaml"
RECORDS = REPO / "Plugins/ops/ingest/mhc/records.jsonl"
OLD_JSON = REPO / "Plugins/ops/ingest/mhc/verification.json"


def _load():
    spec = importlib.util.spec_from_file_location("_ingest_check_t", str(CHECK))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


chk = _load()
BODY = "l1\nl2\n上界是 **1870 GB/s**，别写错行号。\n"


class TestSchemaContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        cls.doc = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))

    def test_every_field_has_four_keys(self):
        for f in self.doc["fields"]:
            for k in ("name", "type", "required", "extract", "from", "if_missing"):
                self.assertIn(k, f, "%s 缺 %s" % (f.get("name"), k))
            self.assertIn(f["extract"], ("machine", "human"))

    def test_human_fields_declared_with_reason(self):
        """业主反复强调：人的判断不许伪装成机器产出。"""
        human = [f for f in self.doc["fields"] if f["extract"] == "human"]
        names = {f["name"] for f in human}
        for must in ("architecture_applicable", "absorb", "our_measured_value"):
            self.assertIn(must, names)
        for f in human:
            self.assertIn("why_human", f, "%s 必须解释为何不能机器化" % f["name"])

    def test_mapping_files_exist(self):
        for k in ("loop_jsonl", "card_yaml"):
            self.assertTrue((REPO / self.doc["field_mapping"][k]["file"]).is_file())

    def test_schema_and_checker_agree_on_human(self):
        names = {f["name"] for f in self.doc["fields"] if f["extract"] == "human"}
        self.assertEqual(names, set(chk.HUMAN),
                         "schema 的 human 字段与校验器的 HUMAN 集合不一致")

    def test_schema_and_checker_agree_on_kinds(self):
        self.assertEqual(set(self.doc["record_contract"]["required_by_kind"]),
                         set(chk.KIND_REQUIRED))


class TestOldFormat(unittest.TestCase):
    """旧 verification.json 的 7 条自检原样保留。"""

    def test_selftest_passes(self):
        self.assertEqual(chk.selftest(), 0)

    def test_old_json_still_valid(self):
        doc = json.loads(OLD_JSON.read_text(encoding="utf-8"))
        self.assertEqual(chk.check_verification(doc), [])

    def test_old_bad_verdict_rejected(self):
        doc = json.loads(OLD_JSON.read_text(encoding="utf-8"))
        doc["records"][0]["claim_verdict"] = "大概对"
        self.assertTrue(any("不在允许集合" in p
                            for p in chk.check_verification(doc)))


class TestNewFormat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp())
        (cls.d / "R.md").write_text(BODY, encoding="utf-8")
        cls.man = {"kind": "manifest", "schema": "mhc-ingest/1",
                   "source_root": str(cls.d), "counts": {"claim": 1}}
        cls.rec = {
            "schema": "mhc-ingest/1", "id": "a", "kind": "claim",
            "stage": "total", "metric": "speedup_x", "claim_text": "q",
            "claim_unit": "x", "claimed_value": 1.89, "baseline_desc": "b",
            "claim_verdict": "not_reverified",
            "provenance": {"file": "R.md", "line": 3,
                           "quote": "上界是 **1870 GB/s**，别写错行号。"},
            "human_pending": ["our_measured_value", "verdict_note",
                              "our_significance", "mechanism", "preconditions",
                              "architecture_applicable", "absorb"],
        }

    def _mut(self, fn):
        r = json.loads(json.dumps(self.rec))
        fn(r)
        return chk.check_records([self.man, r])

    def test_good_passes_and_lists_pending(self):
        probs, pend = self._mut(lambda r: None)
        self.assertEqual(probs, [])
        self.assertIn("a:architecture_applicable", pend)

    def test_human_pending_is_not_an_error(self):
        probs, pend = self._mut(lambda r: None)
        self.assertEqual(pend, ["a:%s" % h for h in self.rec["human_pending"]])
        self.assertEqual(probs, [])

    # ── 防编造：核心判据 ──────────────────────────────────────────────────
    def test_fabricated_quote_rejected(self):
        ps, _ = self._mut(lambda r: r["provenance"].update(quote="文档没这句 9.9x"))
        self.assertTrue(any("grep 不到" in p for p in ps), ps)

    def test_line_drift_rejected(self):
        ps, _ = self._mut(lambda r: r["provenance"].update(line=1))
        self.assertTrue(any("行号漂移" in p for p in ps), ps)

    def test_unreadable_file_rejected(self):
        ps, _ = self._mut(lambda r: r["provenance"].update(file="no.md"))
        self.assertTrue(any("不存在或不可读" in p for p in ps), ps)

    def test_path_escape_rejected(self):
        ps, _ = self._mut(lambda r: r["provenance"].update(file="../../etc/passwd"))
        self.assertTrue(any("不得含 '..'" in p for p in ps), ps)

    def test_stub_quote_rejected(self):
        ps, _ = self._mut(lambda r: r["provenance"].update(quote="x"))
        self.assertTrue(any("太短" in p for p in ps), ps)

    # ── 人类判断与等级 ────────────────────────────────────────────────────
    def test_conforms_without_our_run_rejected(self):
        ps, _ = self._mut(lambda r: r.update(claim_verdict="conforms"))
        self.assertTrue(any("our_measured_value" in p for p in ps), ps)

    def test_missing_required_human_rejected(self):
        ps, _ = self._mut(lambda r: r["human_pending"].remove("absorb"))
        self.assertTrue(any("未列 human_pending" in p for p in ps), ps)

    def test_absorb_true_without_target_rejected(self):
        ps, _ = self._mut(lambda r: r.update(absorb=True))
        self.assertTrue(any("absorb_target" in p for p in ps), ps)

    def test_human_in_missing_rejected(self):
        ps, _ = self._mut(lambda r: r.update(missing={"absorb": "不判"}))
        self.assertTrue(any("人没判不能算机器缺失" in p for p in ps), ps)

    def test_unknown_pending_rejected(self):
        ps, _ = self._mut(lambda r: r["human_pending"].append("vibes"))
        self.assertTrue(any("未知字段" in p for p in ps), ps)

    def test_bandwidth_without_denominator_rejected(self):
        ps, _ = self._mut(lambda r: r.update(metric="bandwidth_gbs"))
        self.assertTrue(any("denominator_gbs" in p for p in ps), ps)

    def test_significance_inconsistent_rejected(self):
        ps, _ = self._mut(lambda r: r.update(significance="none"))
        self.assertTrue(any("significance" in p for p in ps), ps)

    def test_duplicate_id_rejected(self):
        ps, _ = chk.check_records([self.man, self.rec, dict(self.rec)])
        self.assertTrue(any("id 重复" in p for p in ps))

    def test_manifest_count_mismatch_rejected(self):
        ps, _ = chk.check_records([dict(self.man, counts={"claim": 9}), self.rec])
        self.assertTrue(any("counts" in p for p in ps))


class TestRepoRecords(unittest.TestCase):
    """仓库里的 records.jsonl 必须干净且可被 CLI 接受。"""

    def test_records_clean(self):
        recs = [json.loads(x) for x in
                RECORDS.read_text(encoding="utf-8").splitlines() if x.strip()]
        probs, pend = chk.check_records(recs, RECORDS.parent)
        self.assertEqual(probs, [])
        self.assertTrue(pend)

    def test_cli_exit_zero(self):
        r = subprocess.run([sys.executable, "-I", str(CHECK), str(RECORDS)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_cli_rejects_tampered_quote(self):
        """对真数据改一个字，CLI 必须非 0 —— 防「校验器对真数据放水」。"""
        recs = [json.loads(x) for x in
                RECORDS.read_text(encoding="utf-8").splitlines() if x.strip()]
        recs[1]["provenance"]["quote"] += " 编的"
        p = Path(tempfile.mkdtemp()) / "x.jsonl"
        p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in recs),
                     encoding="utf-8")
        r = subprocess.run([sys.executable, "-I", str(CHECK), str(p)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("grep 不到", r.stdout)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------- absorption 数组

def _base_doc():
    return {
        "source": "/x", "ingested_at": "t", "card": "c", "shape": "s",
        "caliber": {"measure": "m", "baseline": "b", "denominator_gbs": 1870},
        "records": [{"stage": "a", "claim_verdict": "not_reverified", "note": "n"}],
        "summary": {"claims_verified": 0},
    }


def test_absorption_true_requires_target_and_action():
    """**反向**：absorb=true 必须说清吸收到哪、怎么吸收。

    实测背景：mqa_opt 的 absorption 原先用 where/action，与 schema 约定的
    absorb_target/absorb_action 不一致，而旧的校验器**只查记录内的 absorb 字段**，
    顶层 absorption 数组整个漏检 —— rc=0 放过了不一致。
    """
    doc = _base_doc()
    doc["absorption"] = [{"item": "某结论", "absorb": True, "why": "有道理"}]
    probs = chk.check_verification(doc)
    assert any("缺 absorb_target" in p for p in probs), probs
    assert any("缺 absorb_action" in p for p in probs), probs


def test_absorption_false_requires_reason():
    """**反向**：absorb=false 必须写 why —— 不吸收的理由也要留痕。"""
    doc = _base_doc()
    doc["absorption"] = [{"item": "某结论", "absorb": False}]
    probs = chk.check_verification(doc)
    assert any("没说 why" in p for p in probs), probs


def test_absorption_missing_flag_rejected():
    """**反向**：不带 absorb 字段不许默认吸收。"""
    doc = _base_doc()
    doc["absorption"] = [{"item": "某结论", "why": "看起来不错"}]
    probs = chk.check_verification(doc)
    assert any("缺 absorb" in p for p in probs), probs


def test_absorption_complete_passes():
    """正向：字段齐全时不该报（否则上面三条是假绿）。"""
    doc = _base_doc()
    doc["absorption"] = [
        {"item": "A", "absorb": True, "absorb_target": "surfaces.yaml",
         "absorb_action": "补 evidence"},
        {"item": "B", "absorb": False, "why": "别人的成果，不进卡"},
    ]
    probs = chk.check_verification(doc)
    assert probs == [], probs
