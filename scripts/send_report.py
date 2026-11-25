#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""send_report —— 把进展汇报发到邮箱。

## 凭证放哪

**绝不放仓库**。凭证读 `~/.config/dcu-toolkit/mail.json`（chmod 600）：

    {"host":"smtp.163.com","port":465,
     "user":"...","password":"...","to":"..."}

仓库里只放这个脚本，不含任何密钥。`test_mail_creds_not_in_repo` 守着这条。

## 为什么单独一个脚本

汇报要能**自动定时发**（每半小时）。做成脚本才能被 cron / 循环调用，
而不是依赖对话。

## 用法

    python3 -I scripts/send_report.py --subject "..." --body-file /tmp/r.md
    python3 -I scripts/send_report.py --selftest
"""
import argparse
import json
import smtplib
import ssl
import sys
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path

CRED = Path.home() / ".config" / "dcu-toolkit" / "mail.json"


def load_cred(p=None):
    p = Path(p) if p else CRED
    if not p.is_file():
        raise SystemExit(
            "缺 %s —— 先把凭证写到那里（chmod 600），不要提交到仓库" % p)
    d = json.loads(p.read_text(encoding="utf-8"))
    for k in ("host", "port", "user", "password", "to"):
        if not d.get(k):
            raise SystemExit("凭证缺 %r" % k)
    return d


def send(subject, body, cred=None):
    c = cred or load_cred()
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = c["user"]
    msg["To"] = c["to"]
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(c["host"], int(c["port"]), context=ctx,
                          timeout=30) as s:
        s.login(c["user"], c["password"])
        s.send_message(msg)
    return c["to"]


def selftest():
    """不联网，只验凭证格式与凭据不泄露。"""
    c = load_cred()
    assert "@" in c["user"] and c["port"], c
    print("[ok] 凭证格式正确：%s @ %s:%s" % (c["user"], c["host"], c["port"]))
    # 反向：确保凭证文件不在仓库里
    repo = Path(__file__).resolve().parents[1]
    for f in repo.rglob("mail.json"):
        raise AssertionError("凭证文件在仓库里：%s —— 必须移出去" % f)
    print("[ok] 反向：仓库内没有 mail.json")
    # 反向：仓库里的任何文件都不该含授权码
    pw = c["password"]
    hits = []
    for f in repo.rglob("*"):
        if not f.is_file() or f.suffix in (".so", ".pyc", ".git"):
            continue
        if ".git/" in str(f):
            continue
        try:
            if pw in f.read_text(encoding="utf-8", errors="ignore"):
                hits.append(str(f.relative_to(repo)))
        except (OSError, UnicodeDecodeError):
            continue
    assert not hits, "授权码泄露到仓库文件里：%s" % hits
    print("[ok] 反向：仓库内没有任何文件含授权码")
    print("\nselftest PASS（3 项，未联网）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="DCU 算子优化进展汇报")
    ap.add_argument("--body-file")
    ap.add_argument("--body", default="（无内容）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    body = (Path(a.body_file).read_text(encoding="utf-8")
            if a.body_file else a.body)
    to = send(a.subject, body)
    print("已发送到 %s" % to)
    return 0


if __name__ == "__main__":
    sys.exit(main())
