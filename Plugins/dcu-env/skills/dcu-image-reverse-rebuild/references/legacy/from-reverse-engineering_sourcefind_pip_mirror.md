# 海光 DCU 对外 pip 镜像：pypi.sourcefind.cn

## 这是什么
海光（Hygon）对外提供的 PyPI 兼容镜像站（DAS Mirror），托管 DCU 定制框架轮：vLLM（das 系列）、PyTorch、numpy、amdsmi 等。是 `10.16.x:9929` 内网 dtk 私有索引的**对外代理**，外网可达。这是「内网 dtk 索引不可达时重建 vLLM 层」的唯一公开换源出路。

## 可用索引路径（关键）
不是 `/simple/` 或 `/pypi/simple/`（那些返回前端 SPA 外壳）。正确路径按 DTK 版本：
```
https://pypi.sourcefind.cn/release/dtk<DTK版本号>/
```
- DTK 26.04 → `https://pypi.sourcefind.cn/release/dtk2604/`
- DTK 25.04.1 → `https://pypi.sourcefind.cn/release/dtk25041/`（首页示例）
- 通用/最新 → `https://pypi.sourcefind.cn/release/dtk/`
DTK 版本号取镜像里 `DTK_VERSION` 环境变量（如 2604），去掉点。

## 在 Dockerfile / pip 里用
```dockerfile
RUN pip install --no-cache-dir vllm==0.15.1 -i https://pypi.sourcefind.cn/release/dtk2604/
# 或写进 pip.conf：[global] index-url = https://pypi.sourcefind.cn/release/dtk2604/
```
实测可装到的 wheel（2026-08-03 于 <device-B> 上验证）：
- `vllm-0.15.1+das.opt1.alpha.dtk2604.torch290.2605201058.g0bef83-cp310-cp310-manylinux_2_24_x86_64.whl`（与原镜像 `0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba` 同系列，仅构建号不同；cp310 = python3.10）
- numpy 有 1.25.0（匹配原 install_dep.sh 的 `numpy==1.25.0`）
- amdsmi 有 24.5.3+02cbffb.dirty（与原镜像一致）

只读验证（不污染容器/镜像）：
```bash
pip index versions vllm -i https://pypi.sourcefind.cn/release/dtk2604/
pip download vllm==0.15.1+das.opt1.alpha.dtk2604.torch290.2605201058.g0bef83 --no-deps \
    -i https://pypi.sourcefind.cn/release/dtk2604/ -d /tmp/x
```

## ⚠️ 排查坑：curl 抓首页会骗你（本会话真实踩过）
- `curl https://pypi.sourcefind.cn/` 或 `/simple/vllm/` 只返回 SPA 外壳 HTML（`<title>DAS Hub</title>`），grep 不到 wheel → **绝不要据此下"不是 pip 源"的结论**。本会话最初就因此误判，用户纠正后才用 `pip -i` 实测验证通。
- 真实索引在 `/release/dtk<ver>/` 下：`curl -L`（跟 302 重定向）到 `/release/dtk<ver>/<pkg>/` 才看得到 wheel 链接；其 href 指向后端 `http://<internal_ip_pypi_storage>:9929/...`，但**用户只需用 pypi.sourcefind.cn 域名即可**，内网 IP 从多数 DCU 主机直连不通，不用管。
- 后端 API `/api/pypi/packages` 有"仅官方 Web UI"来源校验（curl 直调返回 `Access Denied: Invalid Client Source`），但 pip 走 `/release/dtk<ver>/` 简单索引不受影响，无需登录/token。
- 本机浏览器（GLIBC<2.29）跑不起 agent-browser，无法渲染 SPA；用上面的 `pip` / `curl -L` 验证即可，不必强求浏览器。

## 与内网 <internal_ip_src>:9929 的关系
原 Dockerfile 用 `index-url = http://<internal_ip_src>:9929/nightly/dtk${DTK_VERSION}/+simple/`。该内网地址从多数 DCU 主机（如 <device-B>）`nc -z` 不通 → build 必败。替换为 `pypi.sourcefind.cn/release/dtk<ver>/` 即可在出网受限机上重建 vLLM 层。

## 判定流程
1. `nc -z -w5 <internal_ip_src> 9929` 通 → 用原内网索引（最快）。
2. 不通 → `pip index versions vllm -i https://pypi.sourcefind.cn/release/dtk<ver>/`；通 → 改用它。
3. 连 sourcefind 也不通 → vLLM 层只能 (a) 从容器抽轮（违反"不抽产物"约束，需用户放宽）或 (b) 在内网可达的构建机重建。
