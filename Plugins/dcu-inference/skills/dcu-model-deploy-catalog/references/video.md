# 视频生成 部署配方（3 个模型）

> 来源：《<某能源集团客户>2026年网络设备项目测试项汇总》「模型适配支持度」sheet 部署文档交付链接列（标黄=BW1100 已验证）。

> 命令均摘自各模型《海光 DCU 模型部署手册》原文，照抄即跑；路径/卡号/容器名按实际调整。

### HunyuanVideo-1.5
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202`  
**框架**：torchrun, PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it \
--name modelstest-5 \
--network=host \
--ipc=host \
--shm-size=16G \
--device=/dev/kfd \
--device=/dev/mkfd \
--device=/dev/dri \
-v /opt/hyhal:/opt/hyhal \
-v /data/model/:/models/ \
-v <deploy_home>:/workspace/ \
-v /data/scripts/bench:/bench/ \
--group-add video \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202 \
/bin/bash
```

**依赖/环境**：
```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git
cd HunyuanVideo-1.5

pip install loguru==0.7.3 -i https://mirrors.aliyun.com/pypi/simple/
pip install imageio -i https://mirrors.aliyun.com/pypi/simple/
pip install "diffusers==0.35.2"-i https://mirrors.aliyun.com/pypi/simple/
pip install pycountry -i https://mirrors.aliyun.com/pypi/simple/
pip install "transformers==4.51.3"-i https://mirrors.aliyun.com/pypi/simple/
pip install "imageio[ffmpeg]"-i https://mirrors.aliyun.com/pypi/simple/
```

**启动/推理**：
```bash
#!/bin/bash
# ============================================================
# 
基本参数
# ============================================================
PROMPT='A girl holding a paper with words "Hello, world!"'

IMAGE_PATH=none                
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p                
# 480p 
最稳
OUTPUT_PATH=./outputs/output.mp4
MODEL_PATH=/models/Hunyuan/HunyuanVideo-1.5/
# ============================================================
# GPU 
配置
# ============================================================
N_INFERENCE_GPU=1
# ============================================================
# 
推理速度配置
# ============================================================
CFG_DISTILLED=true
SAGE_ATTN=false                
# 
关闭，避免依赖缺失
# 
关闭，最稳定
SPARSE_ATTN=false
OVERLAP_GROUP_OFFLOADING=false
ENABLE_CACHE=false             
CACHE_TYPE=deepcache
ENABLE_STEP_DISTILL=false
# ============================================================
# 
质量配置
# ============================================================
REWRITE=false                  
ENABLE_SR=false                
# 
关闭提示词重写，不需要
 vLLM
# 
关闭超分，先跑通再说
# ============================================================
# 
执行推理
# ============================================================
torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py \--prompt "$PROMPT" \--image_path $IMAGE_PATH \--resolution $RESOLUTION \--aspect_ratio $ASPECT_RATIO \--seed $SEED \--rewrite $REWRITE \--cfg_distilled $CFG_DISTILLED \--enable_step_distill $ENABLE_STEP_DISTILL \--sparse_attn $SPARSE_ATTN \--use_sageattn $SAGE_ATTN \--enable_cache $ENABLE_CACHE \--cache_type $CACHE_TYPE \--overlap_group_offloading $OVERLAP_GROUP_OFFLOADING \--sr $ENABLE_SR \--save_pre_sr_video \--output_path $OUTPUT_PATH \--model_path $MODEL_PATH
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Tencent-Hunyuan/HunyuanVideo-1.5

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202

vim /workspace/HY-WorldPlay/hyvideo/utils/flash_attn_no_pad.py

找到第 35 行左右的代码：
删除最后一个多余的变量 used_seqlens_in_batch
```

### Wan2.1-T2V
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202`  
**框架**：torchrun  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run --shm-size 100g --network=host --name=wan --privileged --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v 项目地址(绝对路径):/home/ -v /opt/hyhal:/opt/hyhal:ro -it harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202 bash
```

**依赖/环境**：
```bash
git clone https://developer.sourcefind.cn/codes/modelzoo/wan2.1_pytorch.git
cd /workspace/CNPC/wan2.1_pytorch
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install "xfuser==0.4.2" --no-deps torch
pip install "peft>=0.17.0" -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

**启动/推理**：
```bash
torchrun --nproc_per_node=4 generate.py \
--task t2v-1.3B \
--size 832*480 \
--ckpt_dir /models/Wan-AI/Wan2.1-T2V-1.3B \
--dit_fsdp \
--t5_fsdp \
--ulysses_size 4 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

**其他命令**：
```bash
访问魔搭社区下载：
https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B
https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202
```

**注意**：
- 注意：若遇到显存不够的问题，可以尝试--offload_model True，--t5_cpu

### Wan2.2-T2V-A14B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.11`  
**框架**：torchrun  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name {docker_name} --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.11 bash
```

**依赖/环境**：
```bash
git clone https://developer.sourcefind.cn/codes/modelzoo/wan2.2_pytorch.git
cd wan2.2_pytorch
pip install-r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install decord -i https://mirrors.aliyun.com/pypi/simple/
pip install librosa -i https://mirrors.aliyun.com/pypi/simple/
```

**启动/推理**：
```bash
torchrun --nproc_per_node=8 generate.py \
--task t2v-A14B \
--size 832*480 \
--ckpt_dir /
models/Wan2.2-T2V-A14B/ \
--dit_fsdp \
--t5_fsdp \
--ulysses_size 8 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves
fight intensely on a spotlighted stage."
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.11
```
