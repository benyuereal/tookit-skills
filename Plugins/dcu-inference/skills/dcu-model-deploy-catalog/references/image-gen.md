# 图像生成 部署配方（8 个模型）

> 来源：《<某能源集团客户>2026年网络设备项目测试项汇总》「模型适配支持度」sheet 部署文档交付链接列（标黄=BW1100 已验证）。

> 命令均摘自各模型《海光 DCU 模型部署手册》原文，照抄即跑；路径/卡号/容器名按实际调整。

### FLUX.1
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run --shm-size 10g --network=host --name=flux --privileged --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v 项目地址(绝对路径):/home/ -v /opt/hyhal:/opt/hyhal:ro -it harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202 bash
```

**依赖/环境**：
```bash
pip install -U diffusers==0.30.3 -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip install transformers==4.45.0 -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
git clone https://github.com/black-forest-labs/flux.git
cd flux
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]" -i http://mirrors.aliyun.com/pypi/simple --trustedhost mirrors.aliyun.com
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/MusePublic/489_ckpt_FLUX_1

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202

import torch
import time
from diffusers import FluxPipeline

MODEL_PATH = "/models/FLUX.1-dev"

print(f"Loading model from {MODEL_PATH} ...")
pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A cat holding a sign that says hello world"

print("Generating image ...")
start = time.time()
image = pipe(
            prompt,
            height=1024，
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cuda").manual_seed(0)
            ).images[0]

elapsed = time.time() - start
image.save("flux-dev.png")
print(f"Image saved to flux-dev.png")
print(f"Time: {elapsed:.2f}s")

if torch.cuda.is_available():
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory - Allocated: {a:.2f}GB, Reserved: {r:.2f}GB, Total:
{t:.2f}GB")
```

### GLM-Image
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10`  
**框架**：SGLang  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run \
--shm-size 500g \
--ipc=host \
--network=host \
--name=modelstest_sglang \
--privileged \
--device=/dev/kfd \
--device=/dev/dri \
--group-add video \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
-v /opt/hyhal:/opt/hyhal \
-v /data/model/:/models/ \
-v <deploy_home>:/workspace/ \
-v /data/scripts/bench:/bench/ \
-it \
harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10 \
/bin/bash
```

**启动/推理**：
```bash
sglang serve --model-path=/models2/ZhipuAI/GLM-Image --served-model-name qwen-image --port 1000
```

**测试**：
```bash
curl http://localhost:1000/v1/images/generations -H "Content-Type: application/json" -d '{ "model": "qwen-image", "prompt": "a cute cat.", "n": 1, "response_format": "b64_json", "size": "1024x1024" }' --output /workspace/scripts/CNPC/output/image.png
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/ZhipuAI/GLM-Image

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10
```

### HunyuanImage 3.0-Instruct
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202`  
**框架**：PyTorch  
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
git clone https://github.com/Tencent-Hunyuan/HunyuanImage-3.0.git
cd HunyuanImage-3.0

pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install tencentcloud-sdk-python 
pip install -r requirements.txt
pip install numpy==1.25.0
pip install flashinfer-python==0.5.0
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Tencent-Hunyuan/HunyuanImage-3.0-Instruct

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202

from transformers import AutoModelForCausalLM, AutoConfig
model_id = "/models/Hunyuan/HunyuanImage-3.0-Instruct/"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
if not hasattr(config, "model_version"):
config.model_version = "3.0"
config.moe_impl = "eager"
config.moe_drop_tokens = True
model = AutoModelForCausalLM.from_pretrained(
model_id,
config=config,
attn_implementation="sdpa",
trust_remote_code=True,
torch_dtype="auto",
device_map="auto",
)
model.load_tokenizer(model_id)
prompt = "
基于图一的
logo
，参考图二中冰箱贴的材质，制作一个新的冰箱贴
"
input_img1 = "./assets/demo_instruct_imgs/input_1_0.png"
cot_text, samples = model.generate_image(
prompt=prompt,
image=input_img1,
seed=42,
image_size="512x512",
# 
缩小输出尺寸
use_system_prompt="en_unified",
bot_task="think_recaption",
infer_align_image_size=False, # 
不对齐输入尺寸，用指定尺寸
diff_infer_steps=30,
# 
减少采样步数
verbose=0,
)
samples[0].save("image_edit.png")
```

### Kolors
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run --shm-size 50g --network=host --name=kolors --privileged --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v 项目地址(绝对路径):/home/ -v /opt/hyhal:/opt/hyhal:ro -it harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202 bash
```

**依赖/环境**：
```bash
git clone https://developer.sourcefind.cn/codes/modelzoo/kolors_pytorch.git
cd /workspace/scripts/CNPC/kolors_pytorch
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install accelerate==0.31.0 -i https://mirrors.aliyun.com/pypi/simple/
pip install peft==0.6.2 -i https://mirrors.aliyun.com/pypi/simple/
python setup.py install
```

**其他命令**：
```bash
访问魔搭社区下载：https://modelscope.cn/models/Kwai-Kolors/Kolors

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202

def infer(prompt):
    ckpt_dir = f'/models/Kolors'
```

### Qwen-Image-2512
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10`  
**框架**：SGLang  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run \
--shm-size 500g \
--ipc=host \
--network=host \
--name=modelstest_sglang \
--privileged \
--device=/dev/kfd \
--device=/dev/dri \
--group-add video \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
-v /opt/hyhal:/opt/hyhal \
-v /data/model/:/models/ \
-v <deploy_home>:/workspace/ \
-v /data/scripts/bench:/bench/ \
-it \
harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10 \
/bin/bash
```

**启动/推理**：
```bash
export HIP_VISIBLE_DEVICES=0
sglang serve \
--model-path=/models/Qwen/Qwen-Image-2512/ \
--port 1000
```

**测试**：
```bash
curl http://localhost:1000/v1/images/generations -H "Content-Type: application/json" -d '{ "model": "qwen-image", "prompt": "a cute cat.", "n": 1, "response_format": "b64_json", "size": "720x720" }' --output /workspace/scripts/CNPC/output/generated_image.png
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Qwen/Qwen-Image-2512

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10
```

### Stable Diffusion
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run --rm --shm-size 10g --network=host --name=stablediffussion --privileged --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v path_to_stablediffussion:/home/sd -v /opt/hyhal:/opt/hyhal:ro -it harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip install diffusers
pip install pycountry
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5/summary

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://developer.sourcefind.cn/codes/modelzoo/stablediffussion.git
```

### Stable Diffusion 1.5
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
docker exec -it test bash
pip install diffusers
pip install simplejson
pip install modelscope
```

**其他命令**：
```bash
访问社区下载：https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2

pipe = pipeline(task=Tasks.text_to_image_synthesis, 
                model='AI-ModelScope/stable-diffusion-v1-5',
                model_revision='v1.0.0')

prompt = 'a small cat'
output = pipe({'text': prompt})
cv2.imwrite('result.png', output['output_imgs'][0])
```

### Stable Diffusion XL
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
docker exec -it test bash
pip install diffusers
pip install simplejson
pip install modelscope
pip install addict
pip install "transformers<5.0" "huggingface_hub<1.0"
```

**其他命令**：
```bash
访问社区下载：https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

from diffusers import StableDiffusionXLPipeline
import torch

model_path = "/root/.cache/modelscope/hub/models/AI-ModelScope/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,   # 关键：强制只读本地，不走 HF Hub 解析
)
pipe.to("cuda")

image = pipe(
    prompt="a photo of an astronaut riding a horse on mars",
    num_inference_steps=30,
).images[0]
image.save("output.png")
print("Done! Saved to output.png")
```
