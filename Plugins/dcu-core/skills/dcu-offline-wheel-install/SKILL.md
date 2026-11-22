---
name: dcu-offline-wheel-install
description: "无外网DCU容器缺少量纯Python包时用：探测、wheel传送与离线pip安装坑。"
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# 无外网 DCU 容器离线装纯 Python 依赖

## 何时用
- 目标 DCU 服务器无外网（先确认：`curl -m 8 https://pypi.org` 超时），容器缺**少量纯 Python**
  依赖（如 omegaconf、pyyaml、小工具包）。
- 只需补几个包，不值得专门构建镜像。需要大批量依赖时改走「外网机构建 → docker
  export/import」（见 dcu-docker 打包章节）。
- 二进制/+das 适配 wheel 不适用本 skill，走内网 `pypi.sourcefind.cn/release/dtk/`
  私有源（见 dcu-env-packages）。

## 步骤

1. **探测基础镜像缺什么**（不先起容器，用 `--rm` 一次性跑）：
```bash
docker run --rm --entrypoint bash <基础镜像> -lc \
 'python --version; python -c "import omegaconf" 2>&1 | tail -1'
```
   注意：未挂 hyhal/DCU 设备的探测容器里 import torch 会报
   `librocm_smi64.so.2 not found`——属预期噪声，只看 python 版本与纯 Python 依赖的结果。

2. **外网机下 wheel**：用 PyPI JSON API 拿直链 URL（比 `pip download` 更可控）：
```python
import json, urllib.request
d = json.load(urllib.request.urlopen("https://pypi.org/pypi/<pkg>/<ver>/json"))
for u in d["urls"]:
    if "py3-none-any" in u["filename"]: print(u["url"]); break
```
   **提前记录哪些依赖只有 sdist**（`urls` 里只有 `.tar.gz`、无 py3 wheel）。典型例：
   `antlr4-python3-runtime` 的 4.9.* 全部只有 sdist（omegaconf 2.3.0 恰好 pin `4.9.*`）
   → 无法 `--no-index` 安装，按第 4 步的 `--no-deps` 处理。

3. **wheel 传进容器**（ssh stdin → `docker exec -i`，避开 docker cp 的 overlay 配额报错）：
```bash
ssh <dcu> "docker exec -i <容器> bash -c 'cat > /tmp/<dist>-<ver>-py3-none-any.whl'" < <dist>-<ver>-py3-none-any.whl
```

4. **离线安装**：
```bash
docker exec <容器> bash -c 'pip install --no-index /tmp/<dist>-<ver>-py3-none-any.whl'
# 子依赖是 sdist-only 装不了时：
docker exec <容器> bash -c 'pip install --no-index --no-deps /tmp/<dist>-<ver>-py3-none-any.whl'
```

5. **import 验证**（pip 退出码不够，必须真 import）：
```bash
docker exec <容器> python -c "import <pkg>; print(<pkg>.__version__)"
```

## 坑

1. **wheel 文件名必须 PEP 427 规范**（`<dist>-<ver>-<abi>-<平台>.whl`）。
   随手命名如 `omegaconf.whl` 会被 pip 直接拒：`Invalid wheel filename (wrong number of
   parts)`。先改名再传，别在容器里临时改名（容易传旧文件）。
2. **`--no-index` 必须带**：不带时 pip 在无外网机上会尝试联网，长时间卡在
   "looking at multiple versions to determine which version is compatible" 然后才失败——
   别把卡住当进度，看到就 kill。
3. **sdist-only 子依赖用 `--no-deps`** 的前提是主包顶层不硬 import 该子依赖（例：
   omegaconf 只在解析时才 import antlr4，顶层 import 无碍）。用 `--no-deps` 前先解 wheel
   源码确认无顶层 import，装完必须走第 5 步 import 验证，不能只看 pip 成功。
4. **pyproject 的 `requires-python` 是保守声明，不是硬门槛**：仓库声明 `>=3.11` 而容器是
   py3.10 时，别按字符串判不兼容——先在目标解释器上跑
   `python3.10 -m compileall -q src/ scripts/`，exit 0 即兼容（开源仓库声明偏保守很常见）。
5. 只适用于纯 Python 依赖；带 C 扩展/编译组件的包需要对应 cp 标签的预编译 wheel 或源码
   构建，不要硬离线装。

## 验收

1. 目标包及其全部子依赖 `import` 成功。
2. `pip check` 无冲突。
3. 版本锁定：`pip freeze` 里该包版本与来源 wheel 一致。
4. 镜像重建后仍可复现（依赖清单入库）。

## 不适用场景

容器有外网；
依赖含需要 CUDA/HIP 编译的扩展（走源码构建流程）；
只想临时修一个包而不管依赖一致性。
