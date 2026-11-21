# Embedding / Reranker 部署配方（11 个模型）

> 来源：《<某能源集团客户>2026年网络设备项目测试项汇总》「模型适配支持度」sheet 部署文档交付链接列（标黄=BW1100 已验证）。

> 命令均摘自各模型《海光 DCU 模型部署手册》原文，照抄即跑；路径/卡号/容器名按实际调整。

### BGE-M3
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**框架**：XInference, PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -itd \
    --shm-size 256g \
    --network=host \
    --name xinference-dcu \
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
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install xinference
pip install sentence-transformer
pip uninstall -y cupy-cuda12x
pip uninstall -y cupy
pip install -U transformers
pip install "peft<0.12.0"
```

**启动/推理**：
```bash
xinference launch -mp /public/bge-m3 --n-gpu 1 -n bge-m3 --model-type embedding
```

**测试**：
```bash
curl -X 'POST' 'http://localhost:9997/v1/embeddings' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": "bge-m3","input": "hello"}'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/BAAI/bge-m3

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

docker exec -it xinference-dcu bash
```

### BGE-Reranker-v2-M3
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**框架**：XInference, PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -itd \
    --shm-size 256g \
    --network=host \
    --name xinference-dcu \
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
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install xinference
pip install sentence-transformer
pip uninstall -y cupy-cuda12x
pip uninstall -y cupy
pip install -U transformers
pip install "peft<0.12.0"
```

**启动/推理**：
```bash
xinference launch -mp /public/bge-reranker-v2-m3 --n-gpu 1 -n bge-reranker-v2-m3 --model-type rerank
```

**测试**：
```bash
curl -X 'POST' 'http://localhost:9997/v1/rerank' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": "bge-reranker-v2-m3", "query": "What is Deep Learning?", "documents": ["Deep Learning is ...", "hello"]}'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/BAAI/bge-reranker-v2-m3

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

docker exec -it xinference-dcu bash
```

### Conan-embedding-v1
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**框架**：XInference, PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -itd \
    --shm-size 256g \
    --network=host \
    --name xinference-dcu \
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
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install xinference
pip install sentence-transformer
pip uninstall -y cupy-cuda12x
pip uninstall -y cupy
pip install -U transformers
pip install "peft<0.12.0"
```

**启动/推理**：
```bash
xinference launch -mp /public/Conan-embedding-v1/ --n-gpu 1 -n Conan-embedding-v1 --model-type embedding
```

**测试**：
```bash
curl -X 'POST' 'http://localhost:9997/v1/embeddings' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": "Conan-embedding-v1","input": "hello"}'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/TencentBAC/Conan-embedding-v1

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

docker exec -it xinference-dcu bash
```

### M3E
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**框架**：XInference, PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -itd \
    --shm-size 256g \
    --network=host \
    --name xinference-dcu \
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
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install xinference
pip install sentence-transformer
pip uninstall -y cupy-cuda12x
pip uninstall -y cupy
pip install -U transformers
pip install "peft<0.12.0"
```

**启动/推理**：
```bash
xinference launch -mp /public/m3e-base --n-gpu 1 -n m3e-base --model-type embedding
```

**测试**：
```bash
curl -X 'POST' 'http://localhost:9997/v1/embeddings' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": "m3e-base","input": "hello"}'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/AI-ModelScope/m3e-base

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

docker exec -it xinference-dcu bash
```

### Qwen3-Embedding
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.18.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name {docker_name} --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.18.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip install sentence-transformers>=2.7.0
```

**启动/推理**：
```bash
## 必须添加HF_ENDPOINT环境变量
export HF_ENDPOINT=https://hf-mirror.com

git clone https://developer.sourcefind.cn/codes/modelzoo/qwen3-embedding_pytorch.git

## model_name_or_path 模型地址参数
python infer_sentence_transformers.py --model_name_or_path /path/your_model_path/
```

**其他命令**：
```bash
访问魔搭社区下载：
https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B
https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-4B
https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-8B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.18.1-ubuntu22.04-dtk26.04-py3.10

import argparse

from sentence_transformers import SentenceTransformer


parse = argparse.ArgumentParser()
parse.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-Embedding-0.6B")
args = parse.parse_args()

# Load the model
model = SentenceTransformer(args.model_name_or_path)

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
# model = SentenceTransformer(
#     "Qwen/Qwen3-Embedding-0.6B",
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
```

**注意**：
- ## 必须添加HF_ENDPOINT环境变量

### Qwen3-Embedding-0.6B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name {docker_name} --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305 bash
```

**依赖/环境**：
```bash
pip install sentence-transformers>=2.7.0
```

**启动/推理**：
```bash
## 必须添加HF_ENDPOINT环境变量
export HF_ENDPOINT=https://hf-mirror.com
## model_name_or_path 模型地址参数
python infer_sentence_transformers.py --model_name_or_path /path/your_model_path/
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305
```

**注意**：
- ## 必须添加HF_ENDPOINT环境变量

### Qwen3-Reranker-4B
**镜像**：`docker`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it \
    --shm-size 256g \
    --network=host \
    --name qwen3-reranker \
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

**启动/推理**：
```bash
export VLLM_USE_NN=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1

vllm serve Qwen/Qwen3-Reranker-0.6B \
    --max-model-len 4096 \
    --block-size 16 \
    --trust-remote-code \
    --enforce-eager \
    --enable-prefix-caching \
    --served-model-name Qwen3-reranker \
    --task score \
    --disable-log-requests \
    --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
```

**测试**：
```bash
curl http://127.0.0.1:8000/score   \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "text_1": "ping",
        "text_2": "pong",
        "model": "Qwen3-reranker"
    }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-4B

docker pull docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10
```

### Qwen3-VL-Embedding-2B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-20260220`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it \
    --shm-size 60g \
    --network=host \
    --name qwen3-vl-embedding \
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
```

**启动/推理**：
```bash
export VLLM_USE_FUSED_RMS_ROPE=0
## serve启动
vllm serve Qwen/Qwen3-VL-Embedding-2B \
    --runner pooling \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name qwen3-vl-embedding \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95
```

**测试**：
```bash
## client访问
curl -s http://127.0.0.1:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{"model": "qwen3-vl-embedding",
        "input": "这是一个用于测试 vLLM embedding 服务是否正常的句子。"
    }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Qwen/Qwen3-VL-Embedding-2B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-20260220
```

### bge-large-en-v1.5
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**框架**：XInference, PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -itd \
    --shm-size 256g \
    --network=host \
    --name xinference-dcu \
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
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install xinference
pip install sentence-transformer
pip uninstall -y cupy-cuda12x
pip uninstall -y cupy
pip install -U transformers
pip install "peft<0.12.0"
```

**启动/推理**：
```bash
xinference launch -mp /public/bge-large-en-v1.5 --n-gpu 1 -n bge-large-en-v1.5 --model-type embedding
```

**测试**：
```bash
curl -X 'POST' 'http://localhost:9997/v1/embeddings' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": "bge-large-en-v1.5","input": "hello"}'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/BAAI/bge-large-en-v1.5

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

docker exec -it xinference-dcu bash
```

### bge-large-zh-v1.5
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**框架**：XInference, PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -itd \
    --shm-size 256g \
    --network=host \
    --name xinference-dcu \
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
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install xinference
pip install sentence-transformer
pip uninstall -y cupy-cuda12x
pip uninstall -y cupy
pip install -U transformers
pip install "peft<0.12.0"
```

**启动/推理**：
```bash
xinference launch -mp /public/bge-large-zh-v1.5 --n-gpu 1 -n bge-large-en-v1.5 --model-type embedding
```

**测试**：
```bash
curl -X 'POST' 'http://localhost:9997/v1/embeddings' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": "bge-large-zh-v1.5","input": "hello"}'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/BAAI/bge-large-zh-v1.5

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

docker exec -it xinference-dcu bash
```

### clip-vit-base-patch32
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name clip-vit-base-patch32 --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/openai-mirror/clip-vit-base-patch32/summary

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel

# 本地模型路径
MODEL_PATH = "/public/clip-vit-base-patch32"

try:
    # 从本地加载模型和处理器
    print(f"正在从本地路径加载模型: {MODEL_PATH}")
    model = CLIPModel.from_pretrained(MODEL_PATH)
    processor = CLIPProcessor.from_pretrained(MODEL_PATH)
    print("✅ 模型和处理器加载成功")

    # 下载测试图片
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    print("正在下载测试图片...")
    image = Image.open(requests.get(url, stream=True).raw)
    print("✅ 图片下载并打开成功")

    # 预处理输入
    texts = ["a photo of a cat", "a photo of a dog"]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    # 推理（不计算梯度以节省资源）
    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # image-text similarity score
    probs = logits_per_image.softmax(dim=1)      # label probabilities

    # 打印结果
    print("\n📊 推理结果:")
    for text, prob in zip(texts, probs[0]):
        print(f"  '{text}' -> 概率: {prob.item():.4f}")

    predicted_label = texts[probs.argmax(dim=1).item()]
    print(f"\n🎯 预测标签: {predicted_label}")
    print("\n✅ 模型运行正常！")

except Exception as e:
    print(f"\n❌ 运行出错: {type(e).__name__}: {e}")
```
