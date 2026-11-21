# Registry 拉取速度诊断（单连接 vs 聚合）

`docker pull` 从内网 registry（Harbor 类）慢得反常时使用：判断瓶颈是单连接限速还是链路共享上限，决定并行是否值得做。

## 步骤

1. 从本机 docker config 读该 registry 的凭证（base64 auth）：
```bash
REG=<host:port>; REPO=<ns>/<image>
USERPASS=$(python3 -c "import json,os; d=json.load(open(os.path.expanduser('~/.docker/config.json'))); print(d['auths']['$REG']['auth'])" | base64 -d)
```

2. 从 `GET /v2/` 的 `Www-Authenticate` 头解析 token 服务（realm/service），换 bearer token：
```bash
WWW=$(curl -sk -o /dev/null -D - "https://$REG/v2/" | grep -i www-authenticate | tr -d '\r')
REALM=$(echo "$WWW" | sed -n 's/.*realm="\([^"]*\)".*/\1/p')
SERVICE=$(echo "$WWW" | sed -n 's/.*service="\([^"]*\)".*/\1/p')
TOKEN=$(curl -sk -u "$USERPASS" "$REALM?service=$SERVICE&scope=repository:$REPO:pull" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('token') or d.get('access_token') or '')")
```

3. manifest 取 top3 大层 digest，并行 curl 3 个 blob 各 60s，测单连接与聚合速度：
```bash
DIGS=$(curl -sk -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json" \
  "https://$REG/v2/$REPO/manifests/<tag>" \
  | python3 -c "import json,sys; ls=sorted(json.load(sys.stdin)['layers'],key=lambda l:-l['size']); [print(l['digest'].split(':')[1]) for l in ls[:3]]")
i=0; for dg in $DIGS; do i=$((i+1))
  ( t0=$(date +%s)
    curl -sk -H "Authorization: Bearer $TOKEN" -o /tmp/bt_$i "https://$REG/v2/$REPO/blobs/sha256:$dg" --max-time 60 2>/dev/null
    t1=$(date +%s); bsz=$(stat -c%s /tmp/bt_$i 2>/dev/null || echo 0)
    echo "conn$i: $((bsz/1024))KB/60s = $((bsz/60/1024))KB/s" ) & done; wait
rm -f /tmp/bt_*
```

## 判定

| 实测形态 | 结论 | 动作 |
|---|---|---|
| 3 连接聚合 ≈ 3× 单连接 | 单连接限速 | 提高 pull 并发（daemon `max-concurrent-downloads`），提速有意义 |
| 3 连接聚合 ≈ 1.0~1.5× 单连接 | 链路共享带宽上限 | 并行无用；别折腾 daemon，让镜像 owner 推一份到内网仓库（daemon.json `insecure-registries`）走内网拉 |

> 实测参考（2026-10 <device-C>←NAS registry）：单连接 585KB/s；3 并行连接仅 +449KB/s 聚合（总 ≈1.03MB/s）→ 判为链路共享上限，~10GB 下载按 4h+ 报 ETA。

## 注意

- 测速不必先杀进行中的 `docker pull`；要杀先 `pgrep -af` 取精确 PID 再 kill（见 dcu-ssh pitfall 16 的自匹配坑）。
- blob curl 从 blob 开头起测速即可，无需 Range；测完删 `/tmp/bt_*`。
