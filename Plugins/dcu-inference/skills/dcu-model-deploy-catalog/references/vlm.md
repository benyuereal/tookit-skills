# VLM（视觉语言模型） 部署配方（14 个模型）

> 来源：《<某能源集团客户>2026年网络设备项目测试项汇总》「模型适配支持度」sheet 部署文档交付链接列（标黄=BW1100 已验证）。

> 命令均摘自各模型《海光 DCU 模型部署手册》原文，照抄即跑；路径/卡号/容器名按实际调整。

### ERNIE-4.5-VL-28/424B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260510-2242`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it \
    --shm-size 200g \
    --network=host \
    --name ernie \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mkfd \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -u root \
    -v /opt/hyhal/:/opt/hyhal/:ro \
    -v /path/your_code_data/:/path/your_code_data/ \
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260510-2242 bash
```

**依赖/环境**：
```bash
git clone https://developer.sourcefind.cn/codes/modelzoo/ernie-4.5-vl-28b-a3b-pt.git
cd ernie-4.5-vl-28b-a3b-pt
pip install-r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps
pip install "tokenizers>=0.22.0,<=0.23.0"-i https://mirrors.aliyun.com/pypi/simple/
pip install joblib -i https://mirrors.aliyun.com/pypi/simple/
pip install proglog -i https://mirrors.aliyun.com/pypi/simple/
pip install decorator -i https://mirrors.aliyun.com/pypi/simple/
pip install imageio  -i https://mirrors.aliyun.com/pypi/simple/
pip install imageio_ffmpeg  -i https://mirrors.aliyun.com/pypi/simple/
```

**启动/推理**：
```bash
## serve启动
vllm serve baidu/ERNIE-4.5-VL-28B-A3B-PT \
    --port 8001 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 --allowed-local-media-path /path/your_pic_data/
```

**测试**：
```bash
## client访问
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "baidu/ERNIE-4.5-VL-28B-A3B-PT",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the image."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///path/your_pic_data/doc/example1.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 128,
    "temperature": 0.0,
    "trust_remote_code": true
  }'
```

**其他命令**：
```bash
访问魔搭社区下载：
https://www.modelscope.cn/models/PaddlePaddle/ERNIE-4.5-VL-28B-A3B-PT
https://www.modelscope.cn/models/PaddlePaddle/ERNIE-4.5-VL-424B-A47B-PT

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260510-2242
```

### GLM-4.1v-9B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 60g --network=host --name glm-41v --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202 bash
```

**启动/推理**：
```bash
export HIP_VISIBLE_DEVICES=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
## 
启动服务
vllm serve /workspace/models/GLM4.1V-9B \--limit-mm-per-prompt '{"image":32}' \--allowed-local-media-path / \--trust-remote-code \--max-model-len 32768 \--served-model-name glm-4.1v-9b \--host 0.0.0.0 \--port 9800 \
```

**测试**：
```bash
curl http://localhost:9800/v1/chat/completions  \-H "Content-Type: application/json"  \-d '{
"model": "glm-4.1v-9b",
"messages": [
{
"role": "user",
"content": [
{
"type": "image_url",
"image_url": {"url": "file:////workspace/sugon.jpg"}
},
{
"type": "text",
"text": "
描述一下这个图片呢
"
}
]
}],
"temperature": 0.7
}'
```

**其他命令**：
```bash
访问魔搭社区下载：
https://www.modelscope.cn/models/ZhipuAI/GLM-4.1V-9B-Base
https://www.modelscope.cn/models/ZhipuAI/GLM-4.1V-9B-Thinking

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202
```

### GLM-4.5V
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-20260220`  
**框架**：vLLM, PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --network=host --hostname=localhost --name=glm-45v -v /opt/hyhal:/opt/hyhal:ro -v /data1/:/data1/ --ipc=host --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri --shm-size=200G --privileged --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-20260220 /bin/bash
```

**依赖/环境**：
```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

pip install pycountry
pip install git+https://github.com/huggingface/transformers.git@159219a6601ab87787b75cdc270b3b2438a9f0cf
```

**启动/推理**：
```bash
## serve启动
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export MOE_NN=0

vllm serve /datav/models/GLM-4.5V  \
     --tensor-parallel-size 8 \
     --gpu-memory-utilization 0.95 \
     --port 8071 \
     --max-model-len 32768 \
     --tool-call-parser glm45 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --allowed-local-media-path / \
     --mm-encoder-tp-mode data \
     --mm_processor_cache_type shm \
     --served-model-name GLM-4.5V
```

**测试**：
```bash
curl http://localhost:8071/v1/chat/completions   \
    -H "Content-Type: application/json"  \
    -d '{
        "model": "GLM-4.5V",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "file//./Grayscale_8bits_palette_sample_image.png"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请描述这张图片的内容。"
                    }
                ]
            }
        ],
        "temperature": 0.8,
        "chat_template_kwargs": {
            "thinking": true
        }
    }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/ZhipuAI/GLM-4.5V

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-20260220
```

### GLM-4.6V
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-20260220`  
**框架**：vLLM, PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it \
    --shm-size 200g \
    --network=host \
    --name glm-4.6v \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mkfd \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -u root \
    -v /opt/hyhal/:/opt/hyhal/:ro \
    -v /path/your_code_data/:/path/your_code_data/ \
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-20260220 bash
```

**依赖/环境**：
```bash
pip install pycountry
pip install git+https://github.com/huggingface/transformers.git@159219a6601ab87787b75cdc270b3b2438a9f0cf
```

**启动/推理**：
```bash
## serve启动
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export MOE_NN=0

vllm serve ZhipuAI/GLM-4.6V \
     --tensor-parallel-size 8 \
     --gpu-memory-utilization 0.95 \
     --port 8001 \
     --max-model-len 32768 \
     --tool-call-parser glm45 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --allowed-local-media-path / \
     --mm-encoder-tp-mode data \
     --mm_processor_cache_type shm \
     --served-model-name GLM-4.6V
```

**测试**：
```bash
curl http://localhost:8001/v1/chat/completions   \
    -H "Content-Type: application/json"  \
    -d '{
        "model": "GLM-4.6V",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "file:///path/of/doc/Grayscale_8bits_palette_sample_image.png"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请描述这张图片的内容。"
                    }
                ]
            }
        ],
        "temperature": 0.8,
        "chat_template_kwargs": {
            "thinking": true
        }
    }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/ZhipuAI/GLM-4.6V

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-20260220
```

### GLM-4v-9B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --name glm_4v --privileged --shm-size=64G  --device=/dev/kfd --device=/dev/dri/ --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ipc=host --network host --group-add video -v /opt/hyhal:/opt/hyhal -v <Host Path>:<Container Path> harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305 /bin/bash
```

**启动/推理**：
```bash
vllm serve model_path  --trust-remote-code --port 8000 --host 0.0.0.0 --allowed-local-media-path xxxx --hf-overrides '{"architectures": ["GLM4VForCausalLM"]}' --chat-template examples/template_chatml.jinja
```

**测试**：
```bash
curl -X POST http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{ 
    "model": "model_path", 
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is the content of this image？"},
          {"type": "image_url", "image_url": {"url": "xxx"}} 
        ]
      }
    ]
  }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/ZhipuAI/glm-4v-9b

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305

git clone https://developer.sourcefind.cn/codes/modelzoo/glm-4v_vllm.git
```

### InternVL-2
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro --shm-size=128G --privileged=true --device=/dev/kfd --device=/dev/dri/ --group-add video --name internvl2 harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
cd /path/your_code_data/

pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
pip install 'numpy<2' -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/OpenGVLab/InternVL2-40B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://developer.sourcefind.cn/codes/modelzoo/internvl2_pytorch.git
```

### InternVL2-8/40B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro --shm-size=128G --privileged=true --device=/dev/kfd --device=/dev/dri/ --group-add video --name internvl2 harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202 bash
```

**依赖/环境**：
```bash
git clone https://developer.sourcefind.cn/codes/modelzoo/internvl2_pytorch.git
cd internvl2_pytorch/
pip install -r requirements.txt \
-i https://mirrors.aliyun.com/pypi/simple \
--trusted-host mirrors.aliyun.com
pip install numpy==1.25.0 -i https://mirrors.aliyun.com/pypi/simple/
```

**其他命令**：
```bash
访问魔搭社区下载：
https://www.modelscope.cn/models/OpenGVLab/InternVL2-8B
https://www.modelscope.cn/models/OpenGVLab/InternVL2-40B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202

# 推理前需要修改模型路径和图片路径
# path = 'OpenGVLab/InternVL2-40B'
# pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# generation_config = dict(max_new_tokens=1024, do_sample=False)

python internvl_chat.py
```

### InternVL3-8/38/78B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size=1024G -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal:/opt/hyhal --network=host --privileged=true --device=/dev/kfd --device=/dev/dri/ --group-add video --name internvl3 harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202 bash
```

**依赖/环境**：
```bash
git clone http://developer.sourcefind.cn/codes/modelzoo/internvl3_pytorch.git
cd /path/your_code_data/

# 修改依赖安装⽂件requirements/segmentation.txt，注释掉如下内容
# mmcv-full==1.6.2

pip install "mmcv>=2.0.0,<2.2.0" --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple/
pip install-r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps
```

**启动/推理**：
```bash
# 修改推理脚本权重路径和图⽚路径

# 单机单卡
python internvl3_inference.py

# 单机多卡
HIP_VISIBLE_DEVICES=0,1,2,3 python internvl3_inference.py
```

**其他命令**：
```bash
访问魔搭社区下载：
https://www.modelscope.cn/models/OpenGVLab/InternVL3-8B
https://www.modelscope.cn/models/OpenGVLab/InternVL3-38B
https://www.modelscope.cn/models/OpenGVLab/InternVL3-78B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202
```

### LLaVA
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --name llava_vllm --privileged --shm-size=64G  --device=/dev/kfd --device=/dev/dri/ --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ipc=host --network host --group-add video -v /opt/hyhal:/opt/hyhal -v <Host Path>:<Container Path> harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305 /bin/bash
```

**启动/推理**：
```bash
vllm serve model_path --chat-template template_llava.jinja --port 8000 --allowed-local-media-path xxx
```

**测试**：
```bash
curl http://localhost:8000/v1/chat/completions \ompletions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "model_path",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": " What is the content of this image？"},
          {"type": "image_url", "image_url": {"url": "xxx"}}
        ]
      }
    ],
    "max_tokens": 300
  }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/llava-hf/llava-1.5-7b-hf/summary

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305
```

### PaddleOCR-VL
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260510-2242`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name {docker_name} --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260510-2242 bash
```

**依赖/环境**：
```bash
git clone https://developer.sourcefind.cn/codes/modelzoo/paddleocr-vl_paddle.git
cd /paddleocr-vl_paddle
pip install-r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install paddlepaddle-dcu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/dcu/
python -m pip install-U "paddleocr[doc-parser]"-i https://mirrors.aliyun.com/pypi/simple/
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
pip install paddlex==3.7.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install numpy==1.25.0 -i https://mirrors.aliyun.com/pypi/simple/
```

**启动/推理**：
```bash
export PADDLE_PDX_DISABLE_DEV_MODEL_WL=1

vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --max-model-len 16384 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.8 --served-model-name PaddleOCR-VL-0.9B
```

**测试**：
```bash
curl http://localhost:8000/v1/chat/completions   \
    -H "Content-Type:application/json"  \
    -d '{
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png"}},
                    {"type": "text", "text": "OCR:"}
                ]
            }
        ],
        "temperature": 0.7
    }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260510-2242
```

### Qwen2.5-VL-3/7/32/72B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it \
    --shm-size 60g \
    --network=host \
    --name qwen2.5 \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mkfd \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -u root \
    -v /opt/hyhal/:/opt/hyhal/:ro \
    -v /path/your_code_data/:/path/your_code_data/ \
    harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10  bash
```

**依赖/环境**：
```bash
pip install -r requirements.txt
vllm推理需要确认qwen_vl_utils库，若不存在，请执行
pip install qwen_vl_utils

# 适用于3B/7B模型
# 启动命令
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --trust-remote-code \
    --max-model-len 32768 \
    --served-model-name qwen-vl \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9

# 适用于72B模型
# 启动命令
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --served-model-name "qwen-vl" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --trust-remote-code \
  --port 8000
```

**测试**：
```bash
## client访问
curl http://localhost:8000/v1/chat/completions   \
    -H "Content-Type: application/json"  \
    -d '{
        "model": "qwen-vl",
        "messages": [
            {
                "role": "user",
                "content": "牛顿提出了哪三大运动定律？请简要说明。"
            }
        ]
    }'

## client访问
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "qwen-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            }
          },
          {
            "type": "text",
            "text": "描述这张图片的内容。"
          }
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.8
  }'
```

**其他命令**：
```bash
访问魔搭社区下载：
https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct
https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct
https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-32B-Instruct
https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10
```

### Qwen3-VL-30/235B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it \
    --shm-size 60g \
    --network=host \
    --name qwen3_vl \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mkfd \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -u root \
    -v /opt/hyhal/:/opt/hyhal/:ro \
    -v /path/your_code_data/:/path/your_code_data/ \
    harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip install -r requirements.txt
```

**启动/推理**：
```bash
## serve启动
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
--trust-remote-code \
--max-model-len 32768 \
--served-model-name qwen-vl \
--dtype bfloat16 \
--tensor-parallel-size 1 \
--gpu-memory-utilization 0.9 \
--allowed-local-media-path local_images_dir
```

**测试**：
```bash
## client访问
curl http://localhost:8000/v1/chat/completions   \
    -H "Content-Type: application/json"  \
    -d '{
        "model": "qwen-vl",
        "messages": [
            {
                "role": "user",
                "content": "牛顿提出了哪三大运动定律？请简要说明。"
            }
        ]
    }'

# /path/to/your/project 请改为图像文件存储的目录
curl http://x.x.x.x:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///path/to/your/project/doc/dog.png"
            }
          },
          {
            "type": "text",
            "text": "请详细描述这张图片的内容。"
          }
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

**其他命令**：
```bash
访问魔搭社区下载：
https://www.modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Instruct
https://www.modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Thinking
https://www.modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Instruct
https://www.modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Thinking

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10

##### 环境变量 #####
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export VLLM_HOST_IP=x.x.x.x # 对应计算节点的IP，选择IB口SOCKET_IFNAME对应IP地址
export NCCL_SOCKET_IFNAME=ibxxxx
export GLOO_SOCKET_IFNAME=ibxxxx
export NCCL_IB_HCA=mlx5_0:1 # 环境中的IB网卡名字
unset NCCL_ALGO
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_NET_GDR_READ=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_SPEC_DECODE_EAGER=1
export VLLM_MLA_DISABLE=0
export VLLM_USE_FLASH_MLA=1

# K100_AI集群建议额外设置的环境变量：
export VLLM_ENFORCE_EAGER_BS_THRESHOLD=44
export VLLM_RPC_TIMEOUT=1800000

# 海光CPU绑定核
export VLLM_NUMA_BIND=1
export VLLM_RANK0_NUMA=0
export VLLM_RANK1_NUMA=1
export VLLM_RANK2_NUMA=2
export VLLM_RANK3_NUMA=3
export VLLM_RANK4_NUMA=4
export VLLM_RANK5_NUMA=5
export VLLM_RANK6_NUMA=6
export VLLM_RANK7_NUMA=7

##### 启动RAY集群 #####
# head节点执行
ray start --head --node-ip-address=x.x.x.x --port=6379 --num-gpus=8 --num-cpus=32
# worker节点执行
ray start --address='x.x.x.x:6379' --num-gpus=8 --num-cpus=32

##### 启动模型服务 #####
vllm serve Qwen/Qwen3-VL-235B-A22B-Thinking \
  --host x.x.x.x \
  --port 8000 \
  --distributed-executor-backend ray \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --block-size 64 \
  --gpu-memory-utilization 0.90 \
  --allowed-local-media-path / \
  --served-model-name qwen-vl \
  --override-generation-config '{"temperature": 0.7, "top_p":0.8, "top_k":20, "repetition_penalty": 1.05}'
```

### Qwen3-VL-32B-Instruct
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it \
    --shm-size 60g \
    --network=host \
    --name qwen3 \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mkfd \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -u root \
    -v /opt/hyhal/:/opt/hyhal/:ro \
    -v /path/your_code_data/:/path/your_code_data/ \
    harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip install -r requirements.txt
export ALLREDUCE_STREAM_WITH_COMPUTE=1
unset NCCL_ALGO
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_NET_GDR_READ=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_SPEC_DECODE_EAGER=1
export VLLM_MLA_DISABLE=0
export VLLM_USE_FLASH_MLA=1

# 海光CPU绑定核
export VLLM_NUMA_BIND=1
export VLLM_RANK0_NUMA=0
export VLLM_RANK1_NUMA=1
export VLLM_RANK2_NUMA=2
export VLLM_RANK3_NUMA=3
export VLLM_RANK4_NUMA=4
export VLLM_RANK5_NUMA=5
export VLLM_RANK6_NUMA=6
export VLLM_RANK7_NUMA=7
```

**启动/推理**：
```bash
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --port 8000 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --block-size 64 \
  --gpu-memory-utilization 0.90 \
  --allowed-local-media-path / \
  --served-model-name qwen-vl \
  --override-generation-config '{"temperature": 0.7, "top_p":0.8, "top_k":20, "repetition_penalty": 1.05}'
```

**测试**：
```bash
# /path/to/your/project 请改为图像文件存储的目录
curl http://x.x.x.x:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///path/to/your/project/doc/dog.png"
            }
          },
          {
            "type": "text",
            "text": "请详细描述这张图片的内容。"
          }
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10
```

### Unimernet
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name unimernet --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
docker exec -it unimernet bash

pip install unimernet

git clone https://ghfast.top/https://github.com/opendatalab/UniMERNet.git

#模型下载到下面目录下
cd UniMERNet/models
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/wanderkid/unimernet_base

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

import argparse
import os
import sys
import numpy as np

import cv2
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.getcwd(), ".."))
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor

class ImageProcessor:
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processor = self.load_model_and_processor()

    def load_model_and_processor(self):
        args = argparse.Namespace(cfg_path=self.cfg_path, options=None)
        cfg = Config(args)
        task = tasks.setup_task(cfg)
        model = task.build_model(cfg).to(self.device)
        vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)

        return model, vis_processor

    def process_single_image(self, image_path):
        try:
            raw_image = Image.open(image_path)
        except IOError:
            print(f"Error: Unable to open image at {image_path}")
            return
        # Convert PIL Image to OpenCV format
        open_cv_image = np.array(raw_image)
        # Convert RGB to BGR
        if len(open_cv_image.shape) == 3:
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()
        # Display the image using cv2

        image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image})
        pred = output["pred_str"][0]
        print(f'Prediction:\n{pred}')

        #cv2.imshow('Original Image', open_cv_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return pred

if __name__ == "__main__":

    root_path = os.path.abspath(os.getcwd())
    config_path = os.path.join(root_path, "configs/demo.yaml")

    processor = ImageProcessor(config_path)

    # Process a single image located at the specified path
    image_path = os.path.join(root_path, 'asset/test_imgs', '0000001.png')
    latex_code = processor.process_single_image(image_path)
```
