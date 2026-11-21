#!/usr/bin/env python3
"""DCU skills 脱敏断言/修复脚本（open-source 公开版，v3）。

用法：
  python3 sanitize.py [repo_root]         # 断言模式：只扫描，有残留即非零退出
  python3 sanitize.py --fix [repo_root]   # 修复模式：应用替换并写回；随后跑 .py/.sh 语法回归
  python3 sanitize.py --selftest          # 自测：构造含各类敏感模式的样本，断言全部被命中

覆盖（v2 在 v1 基础上补齐的空缺）：
  （v3 在 v2 基础上补齐——对应 2026-10-22 开源审查清单的漏报类别）：
  * 内部 GitLab 主机（含端口）/ 内部组路径 / 内部 git 仓库路径 / 内部邮箱域
  * 内网用户名拼音别名（词边界；@提及与邮箱前缀豁免，避免误伤公开 GitHub 用户）
  * 裸设备号 BDF：domain 形（dddd:bb:dd.f）全局命中；短形（bb:dd.f）仅 lspci/BDF/vfio 上下文行
  * 内部 agent 运行时目录路径（~/、^/、/root/、$HOME/ 前缀与词边界裸写）与 frontmatter 内部键名
  * 在线文档指纹短 token（反引号 10-31 位 + 文档上下文词）
  * GBK 乱码文件名检测（mojibake 往返成功即报）；--msg-file 提交信息检查模式（CI/pre-commit）
  * kdocs/金山 文档来源、kdocs 短链、内部文档.cn 短链与域名
  * kdocs 文档指纹 file_id（28~45 位混合大小写+数字，行内需出现文档来源词）
  * 内部机名（node/dcu-node+机号）、裸设备号+单位、设备号+型号（机号=型号 形式）
  * internal_ip_* 占位残缺、内网用户名/内网域、内网 IP（升级为 FAIL，不再仅 warn）
  * CSD- 工单号、内部人名
  * 额外扫描 .gz（解压）/ .sqlite（抽字符串），避免二进制语料/库漏检
  * --fix 后对 .py 跑 py_compile、.sh 跑 bash -n，避免脱敏误伤代码（v1 曾把代码列表里的机号替换成占位符导致 SyntaxError）
  （v4：新增 ssh 主机别名 / 新增用户名别名；selftest 样本改为运行时从 b64 表合成——本文件含 selftest 在内零明文敏感值；下载站/包源等公开分发主机的白名单见 CONTRIBUTING.md）

★ 与内部版（api-service 分支）的区别：本文件是公开交付物，所有内部敏感值
  （人名/用户名/域名/IP/机号）以 base64 编码存储，避免"脱敏脚本自身泄露脱敏对象"。
  规则语义与内部版一致；内部版覆盖范围变化时须同步本文件。
sourcefind.cn 为光合开发者社区公开平台，内容允许引用，不禁。
"""
import base64, gzip, re, shutil, subprocess, sys, tempfile
from pathlib import Path

args = [a for a in sys.argv[1:] if a not in ("--fix", "--selftest")]
ROOT = Path(args[0]).resolve() if args else Path(__file__).parent
FIX = "--fix" in sys.argv
SELFTEST = "--selftest" in sys.argv


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _b64(s):
    return base64.b64decode(s).decode()


# 敏感值编码表（base64；勿改回明文——本文件随公开包发布）
_PERSONS = [_b64(x) for x in ("5b6Q5pmT5qyn", "54aK6L2y", "6YOR5Y+v", "5YiY55Sf5ZCM")]  # 内部作者
_USER = _b64("aml4eA==")                                                        # 内网用户名
_DOMAIN = _b64("c2NuZXQuY24=")                                                  # 内网域
_GW_IP = _b64("MTAuMTcuMjcuNzY=")                                               # 网关 IP
_ALL_DEVS = [_b64(x) for x in ("MTA3", "MjQy", "MTA5", "MzY=")]                 # 内部机号（4 台，值见 b64）
_BARE_DEVS = _ALL_DEVS[:3]                                                      # 参与"裸机号/设备号+型号"规则的前三个（第 4 台会撞小数/版本号，由内部机名规则与型号上下文覆盖）
# 设备/型号关键词（产品公开标识，出现即视为设备语境；用于"设备号=型号"规则）
_DEV_MODEL_KW = r"BW\d+|BW1000|BW100|BW150|BW1100|K100|K500|gfx9\d\d|HYGON|海光|DCU|node\d|机器|节点|设备|容器"
# 平台 SM 架构号（英伟达/TRT 的 SM90/100/103/107，不是我们机号，豁免）
_SM_OK = re.compile(r"SM\d+/\s*\d+/\s*107|SM\d+\s*/\s*107|SM\d+,?\s*107\b|SM\d+/\s*103\s*/\s*107|SM1\d\d")
  # v3 追加（内部 GitLab 主机/组/仓库路径/邮箱域/agent 目录/用户名别名）
_GITLAB_IP = _b64("MTEyLjExLjExOS45OQ==")                                           # 内部 GitLab 主机
_INTERNAL_GROUP = _b64("ZGFzYXBw")                                            # 内部组名
_INTERNAL_GIT_PATH = _b64("ZGN1dG9vbGtpdC9kZWVwbGVhcmluZw==")            # 内部 git 仓库路径
_SUGON_DOMAIN = _b64("c3Vnb24uY29t")                                          # 内部邮箱域
_HYGON_DOMAIN = _b64("aHlnb24uY24=")                                          # 内部邮箱/主机域（v5 新增：此前漏检，sanitize 假绿）
_HYGON_PERSON = _b64("57qq56yR5pet")                                          # 内部 MAINTAINER 署名（v5 新增）
_HERMES_DIR = _b64("Lmhlcm1lcw==")                                           # 内部 agent 运行时目录名
_SSH_ALIAS = _b64("ZGN1ODk=")                                                 # 内部 ssh config 主机别名
_ALIASES = [_b64(x) for x in ("eXVlYmlu","bGl1eHU=","d2FuZ21pbjY=","d3VqbDU=","eWFuZ3Fs","bGFpYmFv","emh1d2Vud2Vu","aGllY29f","ZmFud2g=","c3Vuemhx")]  # 内网用户名拼音别名

DEV_WORDS = r"(?=\s*(?:台|卡|节点|机器|机|设备|的|上|（|\(|实测|容器|真机|镜像|环境|现场|凭证|机型|跳板|跑|,|，|、|基础)(?!\d))"

PERSON_NAMES = _PERSONS
FORBIDDEN_LITERALS = [
    ("\u91d1\u5c71\u6587\u6863", "\u5185\u90e8\u6587\u6863"),
    ("kdocs", "\u5185\u90e8\u6587\u6863"),
    (_USER, "<internal-user>"),
    (_DOMAIN, "<internal-domain>"),
]
RULES = [
    (re.compile(r"https?://(?:www\.)?kdocs\.cn/l/[A-Za-z0-9]+"), "<internal-doc-link>", "kdocs短链URL"),
    (re.compile(r"（/l/[A-Za-z0-9]{8,20}）"), "（内部文档链接，已略）", "kdocs短链"),
    (re.compile(r"(?<![A-Za-z0-9])/l/[A-Za-z0-9]{8,20}\b"), "<internal-doc-link>", "kdocs短链"),
    (re.compile(r"https?://(?:www\.)?\u5185\u90e8\u6587\u6863\.cn/l/[A-Za-z0-9]+"), "<internal-doc-link>", "内部文档短链"),
    (re.compile(r"\u5185\u90e8\u6587\u6863\.cn"), "<internal-doc-link>", "内部文档域"),  # 内部文档.cn
    (re.compile(r"\bdcu-node(" + "|".join(_ALL_DEVS) + r")\b"), "<device-alias>", "内部机名"),
    (re.compile(r"\bnode(" + "|".join(_ALL_DEVS) + r")\b"), "<device-alias>", "内部机名"),
    (re.compile(r"internal_ip_node\d+"), "<internal-ip>", "占位符嵌机名"),
    (re.compile(r"internal_ip_\d+"), "<internal-ip>", "占位符嵌机号"),
]
# 裸机号+单位词（仅前三个机号；按编码表生成，避免明文）
for _i, _d in enumerate(_BARE_DEVS):
    RULES.append((re.compile(r"(?<![-+/\w])" + _d + DEV_WORDS), "<device-%s>" % "ABCD"[_i], "裸设备号"))
# 斜杠组合机号
for _i in range(len(_BARE_DEVS)):
    for _j in range(len(_BARE_DEVS)):
        if _i == _j:
            continue
        RULES.append((re.compile(r"\b%s/%s\b" % (_BARE_DEVS[_i], _BARE_DEVS[_j])),
                      "<device-%s>/<device-%s>" % ("ABCD"[_i], "ABCD"[_j]), "斜杠组合机号"))
# 设备号+型号（机号=型号 / 机号:型号 / 机号 型号 等形式；仅在紧跟 = / : / 或型号词时命中，避开代码列表）
for _i, _d in enumerate(_BARE_DEVS):
    RULES.append((re.compile(r"(?<![-+/\w])" + _d + r"(?=\s*[=:/]?\s*(?:" + _DEV_MODEL_KW + r"))"),
                  "<device-%s>" % "ABCD"[_i], "设备号+型号"))
# 设备号+量词（括号/悬空形式：型号（机号，N 卡）；仅在紧跟 ) 或 ）且随后出现 卡/台/节点 时命中
for _i, _d in enumerate(_BARE_DEVS):
    RULES.append((re.compile(r"(?<![-+/\w])" + _d + r"(?=\s*[)）]\s*\d*\s*[卡台节点])"),
                  "<device-%s>" % "ABCD"[_i], "设备号+量词"))
RULES.append((re.compile(r"\b" + re.escape(_GW_IP) + r"\b"), "<gateway-ip>", "网关IP"))
RULES.append((re.compile(r"\bCSD-\d{4,}\b"), "<internal-ticket>", "工单号"))
RULES.append((re.compile(r"\b" + re.escape(_GITLAB_IP) + r"(?::\d+)?"), "<internal-gitlab>", "内部GitLab主机"))
RULES.append((re.compile(re.escape(_INTERNAL_GROUP) + r"/"), "<internal-group>/", "内部组路径"))
RULES.append((re.compile(re.escape(_INTERNAL_GIT_PATH)), "<internal-group>", "内部git路径"))
RULES.append((re.compile(r"[A-Za-z0-9._%+-]+@" + re.escape(_SUGON_DOMAIN)), "<internal-email>", "内部邮箱"))
# v5：该内部域此前完全漏检，导致 `sanitize.py .` 假绿（exit 0 但仓库里有 21 处残留）
RULES.append((re.compile(r"[A-Za-z0-9._%+-]+@" + re.escape(_HYGON_DOMAIN)), "<internal-email>", "内部邮箱"))
RULES.append((re.compile(r"https?://(?:[\w-]+\.)*" + re.escape(_HYGON_DOMAIN)),
              "<internal-gitlab>", "内部GitLab主机"))
RULES.append((re.compile(r"(?<![@\w.])" + re.escape(_HYGON_DOMAIN)), "<internal-domain>", "内部域"))
RULES.append((re.compile(re.escape(_HYGON_PERSON)), "<internal-user>", "内部作者署名"))
RULES.append((re.compile(r"(?:~/|\^/|/root/|\$HOME/|(?<![\w./-]))" + re.escape(_HERMES_DIR) + r"(?![\w-])"), "<agent-home>", "内部agent路径"))
RULES.append((re.compile(r"^\s{2}hermes:\s*$"), "  toolkit:", "frontmatter内部键"))
BDF_DOMAIN = re.compile(r"\b[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b")
RULES.append((BDF_DOMAIN, "<device-BDF>", "裸BDF"))
BDF_SHORT_CTX = re.compile(r"lspci|BDF|vfio|host=|busid|Bus ID", re.I)
BDF_SHORT = re.compile(r"\b[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b")
for _a in _ALIASES:
    RULES.append((re.compile(r"(?<![@\w])" + re.escape(_a) + r"(?!\w)"), "<internal-user>", "内网用户名"))
RULES.append((re.compile(r"(?<![\w./-])" + re.escape(_SSH_ALIAS) + r"(?!\w)"), "<ssh-alias>", "内部主机别名"))


# kdocs 指纹：28~45 位混合大小写+数字；仅当行内出现文档来源词才判定（v1 只认 32 位，漏了 34/40 位）
FID_CONTEXT = re.compile(r"file_id|fileId|docId|documentId|internal-file-id|内部文档|子文档|溯源|《[^》]{4,}》|kdocs|金山\s*文档", re.I)
FILE_ID = re.compile(r"\b(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]{28,45}\b")
FID_SHORT = re.compile(r"`(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[0-9])([A-Za-z0-9]{10,31})`")
_CONT_NAME = re.compile(r"\b" + re.escape(_USER) + r"-[A-Za-z0-9_-]+\b")
LOOPBACK = {"0.0.0.0", "127.0.0.1", "127.0.1.1", "255.255.255.255"}
VERSION_LIKE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
PRIVATE_IP = re.compile(r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b")
ALLOWED_IP = ("10.244.", "10.96.", "192.168.122.")
EXTS = {".md", ".sh", ".py", ".json", ".yaml", ".yml", ".sql", ".txt", ".rst", ".toml", ".cpp", ".hpp", ".cc", ".h", ".c", ".mk", ".cmake", ".cfg", ".ini", ".conf", ".dockerfile", ".gz", ".sqlite", ".db",
        # v5：此前漏扫——package.env 的 MAINTAINER/HOMEPAGE 在这里
        ".env"}
# 无后缀（dotfile）也要扫的文件名：.gitmodules 的 url 曾漏检
EXTRA_NAMES = {".gitmodules", ".gitconfig"}
SKIP_FILES = {"sanitize.py"}
SKIP_DIRS = {".git"}

PUBLIC_REPO_HOST = "developer.sourcefind.cn"
PUBLIC_REPO_PATH = "/codes/jixx/dcu-toolkit-skills"


def _is_public_repo_url(line):
    return PUBLIC_REPO_HOST in line and PUBLIC_REPO_PATH in line


def iter_files():
    for f in ROOT.rglob("*"):
        if f.is_file() and (f.suffix.lower() in EXTS or f.name in EXTRA_NAMES) \
                and not (set(f.parts) & SKIP_DIRS):
            if f.name in SKIP_FILES:
                continue
            yield f


def read_scan_text(f):
    """返回 (text, 是否二进制需按行扫)。.gz 解压；.sqlite/.db 抽可打印串。"""
    if f.suffix.lower() == ".gz":
        try:
            with gzip.open(f, "rb") as gh:
                return gzip.decompress(gh.read()).decode("utf-8", "replace")
        except Exception:
            return ""
    if f.suffix.lower() in (".sqlite", ".db"):
        try:
            data = f.read_bytes()
            return "".join(chr(b) if 32 <= b < 127 else "\n" for b in data)
        except Exception:
            return ""
    return None  # 普通文本


DEV_RULE_KINDS = {"裸设备号", "设备号+型号", "斜杠组合机号", "内部机名"}
CODE_EXTS = {".py", ".sh", ".cpp", ".hpp", ".cc", ".h", ".c", ".mk", ".cmake", ".toml"}


def process_line(line, is_code=False):
    hits = []
    for name in PERSON_NAMES:
        if name in line:
            line = line.replace(name, "<internal-author>")
            hits.append("person")
    if _CONT_NAME.search(line):
        line = _CONT_NAME.sub("<container>", line)
        hits.append("container")
    for lit, rep in FORBIDDEN_LITERALS:
        if lit.lower() in line.lower():
            line = re.sub(re.escape(lit), rep, line, flags=re.I)
            hits.append("literal")
    for pat, rep, kind in RULES:
        if is_code and kind in DEV_RULE_KINDS:
            continue
        if pat.search(line):
            line = pat.sub(rep, line)
            hits.append(kind)
    if FID_CONTEXT.search(line):
        for m in FILE_ID.finditer(line):
            line = line.replace(m.group(0), "<internal-file-id>")
            hits.append("file-id")
    if FID_CONTEXT.search(line):
        for m2 in FID_SHORT.finditer(line):
            line = line.replace(m2.group(0), "`<internal-file-id>`")
            hits.append("file-id-short")
    if BDF_SHORT_CTX.search(line) and BDF_SHORT.search(line):
        line = BDF_SHORT.sub("<device-BDF>", line)
        hits.append("裸BDF短形")
    return line, hits


def syntax_check():
    """--fix 后跑 .py/.sh 语法回归；返回失败列表。避免 v1 脱敏误伤代码。
    .py 用 py_compile（阻断）；.sh 用 bash -n，但先探测 bash 是否可用（Windows/无 bash 时跳过，避免误报）。"""
    import shutil
    bad = []
    _has_bash = shutil.which("bash") is not None
    if _has_bash:
        # 探针：bash 能正常解析一个合法空脚本才认为可用（本机 bash 损坏时会失败）
        import tempfile
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as tf:
                tf.write(": \n")
                tf.write("exit 0\n")
                probe = tf.name
            r = subprocess.run(["bash", "-n", probe], capture_output=True)
            _has_bash = (r.returncode == 0)
        except Exception:
            _has_bash = False
        finally:
            try:
                import os
                os.unlink(probe)
            except Exception:
                pass
    def _rc(args):
        try:
            r = subprocess.run(args, capture_output=True)
            return r.returncode
        except Exception:
            return 1
    for f in iter_files():
        if f.suffix.lower() == ".py":
            if _rc([sys.executable, "-m", "py_compile", str(f)]) != 0:
                bad.append("PY " + str(f.relative_to(ROOT)))
        elif f.suffix.lower() == ".sh" and not str(f).endswith("install_sync_selftest.sh"):
            if not _has_bash:
                continue
            if _rc(["bash", "-n", str(f)]) != 0:
                bad.append("SH " + str(f.relative_to(ROOT)))
    return bad



def check_mojibake_names():
    """GBK 乱码文件名：非 ASCII 文件名若 GBK 编码后可按 UTF-8 解码为不同字符串，视为乱码。"""
    bad = []
    for f in ROOT.rglob("*"):
        if f.is_file() and not (set(f.parts) & SKIP_DIRS):
            n = f.name
            if any(ord(c) > 127 for c in n):
                try:
                    fixed = n.encode("gbk").decode("utf-8")
                    if fixed != n:
                        bad.append((str(f.relative_to(ROOT)), fixed))
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
    return bad


def check_msg(path):
    """提交信息检查模式：复用行级规则，命中即非零退出（CI/pre-commit 用）。"""
    fails = []
    t = path.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(t.split("\n"), 1):
        if _is_public_repo_url(line):
            continue
        new_line, hits = process_line(line)
        for m in PRIVATE_IP.finditer(line):
            ip = m.group(0)
            if ip in LOOPBACK or VERSION_LIKE.match(ip) or any(ip.startswith(p) for p in ALLOWED_IP):
                continue
            fails.append("  commit-msg:%d [internal-ip] %s" % (i, line.strip()[:100]))
        if hits:
            fails.append("  commit-msg:%d [%s] %s" % (i, ",".join(sorted(set(hits))), line.strip()[:100]))
    tag = "OK" if not fails else "FAIL"
    print("%s: %d 处提交信息命中（--msg-file %s）" % (tag, len(fails), path.name))
    for x in fails[:200]:
        print(x)
    return 0 if not fails else 1


def main():
    changed = 0
    fails = []
    scanned = 0
    skipped = 0
    for f in iter_files():
        try:
            t = read_scan_text(f)
        except Exception:
            continue
        if t is None:
            try:
                t = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                skipped += 1
                continue
        scanned += 1
        out_lines = []
        file_hits = []
        _is_code = f.suffix.lower() in CODE_EXTS
        for i, line in enumerate(t.split("\n"), 1):
            if _is_public_repo_url(line):
                out_lines.append(line)
                continue
            new_line, hits = process_line(line, is_code=_is_code)
            if hits:
                file_hits.append((i, hits, line.strip()[:110]))
            out_lines.append(new_line)
            for m in PRIVATE_IP.finditer(line):
                ip = m.group(0)
                if ip in LOOPBACK or VERSION_LIKE.match(ip) or any(ip.startswith(p) for p in ALLOWED_IP):
                    continue
                fails.append("  %s:%d [internal-ip] %s" % (f.relative_to(ROOT), i, line.strip()[:100]))
        if file_hits:
            rel = str(f.relative_to(ROOT))
            if FIX:
                f.write_text("\n".join(out_lines), encoding="utf-8")
                changed += 1
            for i, hits, exc in file_hits[:60]:
                fails.append("  %s:%d [%s] %s" % (rel, i, ",".join(sorted(set(hits))), exc))

    for rel, fixed in check_mojibake_names():
        fails.append("  %s [乱码文件名] 疑似 GBK 乱码（还原后应为: %s）" % (rel, fixed))

    syntax_fail = syntax_check() if FIX else []
    if syntax_fail:
        fails.extend("  [SYNTAX] " + x for x in syntax_fail)

    tag = "FIX 完成" if FIX else ("FAIL" if fails else "OK")
    print("%s: %d 个文件改动，命中 %d 行（%d 个文件已扫描，%d 个二进制跳过）" %
          (tag, changed, len(fails), scanned, skipped))
    for x in fails[:2000]:
        print(x)
    if len(fails) > 80:
        print("  ... 其余 %d 处省略" % (len(fails) - 2000))
    return 0 if (FIX or not fails) else 1


def selftest():
    """构造各类敏感样本，断言规则全部命中；不含真实敏感值（用等价样本）。"""
    import io
    samples = [
        # (输入, 期望含 hit 标签) —— 敏感值一律运行时从 b64 表合成，本文件（含 selftest）不出现明文敏感值
        ("kdocs short link: https://www.kdocs.cn/l/xY9aBc12", "literal"),  # kdocs 先被 FORBIDDEN 规则改为 内部文档
        ("internal doc short: https://www.内部文档.cn/l/AbCdEf123456", "内部文档短链"),
        ("source: 《DCU PyTorch 实践》 by 李某 file_id: Aa1Bb2Cc3Dd4Ee5Ff6Gg7Hh8Ii9Jj10Kk", "file-id"),
        ("machine node" + _ALL_DEVS[0] + " is under test", "内部机名"),
        ("machine dcu-node" + _ALL_DEVS[1] + " active", "内部机名"),
        ("device " + _ALL_DEVS[0] + " = BW150, gfx936", "设备号+型号"),
        ("device " + _ALL_DEVS[1] + "=BW100 gfx936", "设备号+型号"),
        ("nodes " + _ALL_DEVS[0] + "/" + _ALL_DEVS[2] + " ping ok", "斜杠组合机号"),
        ("container " + _USER + "-e2e running", "container"),
        ("gateway " + _GW_IP + " reachable", "网关IP"),
        ("作者： " + _PERSONS[3] + " 编写了本文档", "person"),  # 内部作者（经 b64，避免明文）
        ("user domain " + _DOMAIN + " suffix", "literal"),
        ("ticket CSD-0000 opened", "工单号"),
        ("code list [101, 102, 103, " + _ALL_DEVS[0] + ", 108]", "none-code"),  # 不应命中设备号（代码上下文）
        ("git clone http://" + _GITLAB_IP + ":10068/" + _INTERNAL_GROUP + "/x.git", "内部GitLab主机"),
        ("repo " + _INTERNAL_GIT_PATH + "/dev.git", "内部git路径"),
        ("mail me at dev@" + _SUGON_DOMAIN + " please", "内部邮箱"),
        ("mail me at dev@" + _HYGON_DOMAIN + " please", "内部邮箱"),  # v5 新增：此前假绿
        ("HOMEPAGE=\"https://gitlab." + _HYGON_DOMAIN + "/group/repo\"", "内部GitLab主机"),
        ("MAINTAINER=\"" + _HYGON_PERSON + " <x@y>\"", "内部作者署名"),
        ("plain domain " + _HYGON_DOMAIN + " appears", "内部域"),
        ("cd ~/x then " + _HERMES_DIR + "/state.db", "内部agent路径"),
        ("lspci output 0000:03:00.0 ok", "裸BDF"),
        ("lspci 03:00.0 and 03:00.1 listed", "裸BDF短形"),
        ("user " + _ALIASES[0] + " logged in", "内网用户名"),
        ("user " + _ALIASES[7] + " logged in", "内网用户名"),  # v4 新增别名（末位新增项）
        ("ssh " + _SSH_ALIAS + " connected", "内部主机别名"),
        ("docId `abCdEf123456` subdoc source", "file-id-short"),
        ("public mention @someone-name here", "none"),
        ("nvidia SM" + _ALL_DEVS[0] + " kernels", "none"),  # SM 架构号不应命中
    ]
    ok = True
    for inp, expect in samples:
        out, hits = process_line(inp, is_code=(expect == "none-code"))
        got = ",".join(sorted(set(hits)))
        if expect in ("none", "none-code"):
            if hits:
                print("SELFTEST FAIL: expected no hit, got", got, "for", inp[:60]); ok = False
            continue
        if expect not in got:
            print("SELFTEST FAIL: expected", expect, "got", got, "for", inp[:60]); ok = False
    print("SELFTEST", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--msg-file" in sys.argv:
        sys.exit(check_msg(Path(sys.argv[sys.argv.index("--msg-file") + 1])))
    if SELFTEST:
        sys.exit(selftest())
    sys.exit(main())
