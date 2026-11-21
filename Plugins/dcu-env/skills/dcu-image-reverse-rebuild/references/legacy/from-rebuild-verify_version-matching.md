# DCU 镜像重建：Python 包版本对齐方法论（sourcefind 实测）

> 适用场景：把内网友源 DCU 镜像（vLLM/DTK 定制轮）反推成「公开源可重建、且 DCU 定制包
> 版本与原镜像一致」的 Dockerfile。本文件是 `dcu-image-rebuild-verify` 的版本对齐专章，
> 与 `dcu-image-reverse-engineering` 的源替换表配套使用。

## 0. 为什么必须做版本对齐（用户硬需求）

用户原话：「你构建的时候需要关注环境里的 python 包版本跟 dockerfile 里安装的是不是一样，
特别是 dcu 单独适配的包版本」「换设备能通过这个 dockerfile 构建出相同的镜像然后运行容器」。

→ 不能让 pip 自己解析最新版。必须：抓原容器 `pip freeze` → 提取 DCU 定制包 + vllm 关键依赖
的精确版本 → 写死 pin 进 Dockerfile → 重建后做完整 freeze diff 确认一致。

## 1. 标准流程（五步）

```bash
# ① 抓原容器完整包列表（在目标机，原容器必须 Up）
docker exec <orig-container> pip list --format=freeze 2>/dev/null | sort > /tmp/orig.txt
# ② 第一次 build 后，抓重建镜像的包列表
timeout 120 docker run --rm <rebuilt-image> pip list --format=freeze 2>/dev/null | sort > /tmp/reb.txt
# ③ python diff：只关注 DCU 定制包(+das/+dtk) + vllm 关键依赖
python3 - <<'PY'
def load(p):
    d={}
    for l in open(p):
        l=l.strip()
        if '==' in l:
            n,v=l.split('==',1); d[n.strip().lower()]=v
    return d
o=load('/tmp/orig.txt'); r=load('/tmp/reb.txt')
def dcu(v): return ('das' in v.lower()) or ('dtk' in v.lower())
KEY={'vllm','torch','torchvision','triton','flash_attn','flash_mla','lightop','lmslim',
     'tensorizer','setuptools','msgspec','safetensors','numpy','transformers','xgrammar',
     'amdsmi','cmake','outlines_core','llguidance','gguf','mistral_common','numba','grpcio',
     'timm','runai-model-streamer','cinotificate','citest','opencompass'}
for k in sorted(set(o)|set(r)):
    if k in o and k in r and o[k]!=r[k] and (dcu(o[k]) or dcu(r[k]) or k in KEY):
        print(f'{k}: orig={o[k]} rebuild={r[k]}')
print('多出(原无):',[k for k in r if k not in o])
print('缺失(原有):',[k for k in o if k not in r])
PY
```

④ 对每个 `orig≠rebuild` 的 DCU 包：
   - **确认 nightly/dtk 上真实可得**（见 §2 的 INSTALLED 陷阱）——可得则写死精确 pin；
   - 不可得（nightly 未保留旧构建号）→ 选同 torch 系列（如 torch290）**最新可用**版 pin；
   - 辅助小包（citest/cinotificate 之类评测工具，非 vllm 运行核心）若 nightly 无原版，
     可漂移或 `--no-deps` 装本体。

⑤ 重建 install_dep.sh 末尾把所有 DCU 包精确 pin 成原容器/同系列版本，再 build + 重做 ③ 的 diff，
   直到 DCU 包全部一致（或同系列）为止。

## 2. 致命坑：INSTALLED 行陷阱（本会话踩 3 次）

`pip index versions <pkg> -i https://pypi.sourcefind.cn/nightly/dtk/` 的输出里，
**`INSTALLED: X` 是「原容器已装版本」，不是「nightly 可下载版本」**。把 INSTALLED 当成
可用 → 写死 pin → `No matching distribution found` 让 build 失败（vllm、flash_attn、
flash_mla 都踩过）。

**正确探测法**：取 nightly 实际可下载版本 = 排除 `INSTALLED` 行后的纯版本号：
```bash
docker exec <orig-container> bash -c \
  "pip index versions flash_attn -i https://pypi.sourcefind.cn/nightly/dtk/ 2>/dev/null" \
  | tr ',' '\n' | grep -E 'torch290' | grep -v INSTALLED | sort | tail -1
```
只有**非 INSTALLED 行**才是 nightly 真实可得。写 pin 前必须这一句确认。

## 3. 关键版本关系（写错必现运行时错）

- **torch ↔ torchvision 严格匹配**：torch 2.9.0(torch290) 必须 torchvision 0.24.0(同 torch290)。
  装成 torch2100 系列的 0.25.0 → `vllm serve` 报 `RuntimeError: operator torchvision::nms does not exist`。
  `import torch/vllm` 能过但 `vllm serve` 才暴露，验证必须跑 `vllm serve -h`。
- **cmake==3.29 死结**：vllm METADATA 声明 `Requires-Dist: cmake==3.29`，但所有源都**没有
  3.29.0 精确版**（只有 3.29.0.1）。vllm 发的是预编译 wheel 不需要本地编译。
  解法：`pip install vllm --no-deps`（跳过 cmake 约束）装 wheel → 从 wheel METADATA 提取
  其余 Requires-Dist（排除 cmake）逐个补装 → cmake 本身装 3.29.0.1（构建期工具，不影响运行）。
- **flash_attn / flash_mla / lightop / lmslim**：DCU 定制后端，nightly 上旧构建号常不保留，
  用同 torch 系列最新可用版（torch 系列对，主版本差影响小，比 torch251 系列安全）。
- **tensorizer / setuptools / transformers / xgrammar** 等：vllm 有精确要求（如
  transformers==5.2.0、xgrammar==0.1.29、setuptools<80、tensorizer==2.10.1），逐包补依赖会
  装成不兼容最新版 → 末尾显式 pin 回原容器版本（均已在 nightly 确认可得）。

## 4. opencompass 网络不稳 → docker cp 本地装（本会话第 15/16 次 build 卡死根因）

原镜像 citest 0.0.8 依赖 `opencompass==0.5.1`，但该 whl 在 sourcefind `/open/` 镜像**反复
下载中断**（2.6MB/3.2MB 断流，pip 6 次重试失败）→ build 整层挂掉。sourcefind 对该文件的
服务器问题，版本/网络配置都救不了。

解法（公开包，合法从原容器抽，非 DTK 私有物）：
```bash
# 从原容器拷出已安装的 opencompass 包目录 + dist-info（零下载）
docker cp <orig-container>:/usr/local/lib/python3.10/dist-packages/opencompass \
         /data/buildctx_final/oc_pkg/
docker cp <orig-container>:/usr/local/lib/python3.10/dist-packages/opencompass-0.5.1.dist-info \
         /data/buildctx_final/oc_pkg/
# Dockerfile: COPY oc_pkg /opt/oc_pkg
# install_dep.sh:
pip install /opt/oc_pkg/opencompass          # 本地目录安装，零网络
pip install --no-deps "citest==0.0.8" -i ${VLLM_INDEX}   # 不再触发 opencompass 下载
pip install --no-deps "cinotificate==0.0.3" -i ${VLLM_INDEX}
```
注意：原容器里包路径用**绝对路径字面值**，`docker cp` 命令里的变量不会进容器展开。

## 5. 网络容错（sourcefind 偶发抖动）

所有 `pip install` 加 `--timeout 120 --retries 10`（注：下载中断续传由 `--resume-retries`
控制，提示里会写；`--retries` 覆盖连接重试）。python 逐包补依赖逻辑里也加超时/重试
（单包 6 次 + 8s 间隔），避免单包网络抖动拖垮整层。

## 6. 验证判据（不能只看 Successfully built）

1. 完整 `pip freeze` diff：DCU 定制包 + vllm 关键依赖**逐版本一致**（或同系列），无意外多出/缺失。
2. `docker run <镜像> vllm serve -h` 退出 0（触发完整 import 链，暴露 torchvision nms 等
   仅 import 时不报的错）。
3. 原容器 `import torch/vllm` 与重建镜像行为一致（无卡环境 HSA 告警属正常）。
