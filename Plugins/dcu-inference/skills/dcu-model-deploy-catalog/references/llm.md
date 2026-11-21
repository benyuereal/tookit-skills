# LLM（语言模型） 部署配方（29 个模型）

> 来源：《<某能源集团客户>2026年网络设备项目测试项汇总》「模型适配支持度」sheet 部署文档交付链接列（标黄=BW1100 已验证）。

> 命令均摘自各模型《海光 DCU 模型部署手册》原文，照抄即跑；路径/卡号/容器名按实际调整。

### Baichuan2-7/13B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -dit --shm-size 80g --network=host --name=baichuan2 --privileged --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 /bin/bash
```

**依赖/环境**：
```bash
pip install -r requirements.txt  -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

**其他命令**：
```bash
访问魔搭社区下载：
https://www.modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat
https://www.modelscope.cn/models/baichuan-inc/Baichuan2-7B-Base
https://www.modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat
https://www.modelscope.cn/models/baichuan-inc/Baichuan2-13B-Base

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://developer.sourcefind.cn/codes/modelzoo/baichuan2_pytorch.git
```

### ChatGLM3-6B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -dit --network=host --name=chatglm3 --privileged --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size=16G   -v /opt/hyhal/:/opt/hyhal/:ro --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root --ulimit stack=-1:-1 --ulimit memlock=-1:-1 harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10
```

**依赖/环境**：
```bash
docker exec -it chatglm3 /bin/bash

git clone https://developer.sourcefind.cn/codes/modelzoo/chatglm3-6b_pytorch.git

cd chatglm3-6b_pytorch
pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

pip install transformers==4.39.2  # 降级
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/ZhipuAI/chatglm3-6b

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

cd ./basic_demo/
export MODEL_PATH=/path/to/chatglm3-6b/
export TOKENIZER_PATH=/path/to/chatglm3-6b/

python cli_demo.py
```

### CodeLLaMA-34B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro --shm-size=80G --privileged=true --device=/dev/kfd --device=/dev/dri/ --group-add video --name docker_name harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202 bash
```

**启动/推理**：
```bash
export HIP_VISIBLE_DEVICES=0
vllm serve /models/Llama/CodeLlama-34b-hf \-tp 1 \--port 8000 \--max-model-len 4096 \--gpu-memory-utilization 0.9 \--served-model-name codellama \--host 0.0.0.0 \--port 9800 \
```

**测试**：
```bash
curl -X POST http://localhost:9800/v1/completions \-H "Content-Type: application/json" \-d '{
"model": "codellama",
"prompt": "def fibonacci(n):\n    
"max_tokens": 200,
"temperature": 0.1
}'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/codefuse-ai/CodeFuse-CodeLlama-34B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-0130-py3.10-20260202
```

### DeepSeek-R1
**镜像**：`image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.3.0-ubuntu22.04-dtk24.04.3-py3.10`  
**框架**：torchrun  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run --shm-size 500g --network=host --name=dpskr1 --privileged --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v 项目地址(绝对路径):/home/ -v /opt/hyhal:/opt/hyhal:ro -it image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.3.0-ubuntu22.04-dtk24.04.3-py3.10 bash
```

**依赖/环境**：
```bash
cd inference
pip install -r requirements.txt
```

**启动/推理**：
```bash
# chat
torchrun --nnodes 4 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path /path/to/DeepSeek-R1-Demo --config configs/config_671B.json --interactive --temperature 0.7 --max-new-tokens 200
# 直接推理
torchrun --nnodes 4 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path /path/to/DeepSeek-R1-Demo --config configs/config_671B.json --input-file $FILE
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1

docker pull image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.3.0-ubuntu22.04-dtk24.04.3-py3.10
```

**注意**：
- 注意：需要将/path/to/fp8_weights中的json文件复制到/path/to/DeepSeek-V3-Demo中。

### DeepSeek-R1-Distill-Llama-70B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

### DeepSeek-R1-Distill-Llama-70B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

### DeepSeek-R1-Distill-Llama-8B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1-ubuntu22.04-`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

### DeepSeek-R1-Distill-Qwen-1.5B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1-ubuntu22.04-`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

### DeepSeek-R1-Distill-Qwen-14B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1-ubuntu22.04-`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

### DeepSeek-R1-Distill-Qwen-32B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1-ubuntu22.04-`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

### DeepSeek-R1-Distill-Qwen-7B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1-ubuntu22.04-`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

### DeepSeek-V3.2
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it \
    --shm-size 200g \
    --network=host \
    --name deepseek-v32 \
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
export VLLM_USE_V32_ENCODE=1

export ALLREDUCE_STREAM_WITH_COMPUTE=1
export VLLM_HOST_IP=x.x.x.x # 对应计算节点的IP，建议选择IB口SOCKET_IFNAME对应IP地址
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_SOCKET_IFNAME=ibxxxx
export GLOO_SOCKET_IFNAME=ibxxxx
export NCCL_IB_HCA=mlx5_0:1 # 根据 ibstat 查看
unset NCCL_ALGO
export NCCL_IB_DISABLE=0
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_NET_GDR_READ=1
export NCCL_DEBUG=INFO
export NCCL_MIN_P2P_NCHANNELS=16
export NCCL_NCHANNELS_PER_PEER=16
export HIP_USE_GRAPH_QUEUE_POOL=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_FLASH_MLA=1

# 海光CPU绑定核,intel cpu可不加
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
python inference/fp8_cast_bf16.py --input-fp8-hf-path /path/to/DeepSeek-V3.2 --output-bf16-hf-path /path/to/DeepSeek-V3.2-bf16
```

**测试**：
```bash
curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/path/to/DeepSeek-V3.2-bf16",
        "messages": [
            {
                "role": "user",
                "content": "请介绍下你自己。"
            }
        ],
        "temperature": 1.0,
        "chat_template_kwargs": {
            "thinking": true
        }
    }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3.2

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10

git clone https://developer.sourcefind.cn/codes/modelzoo/deepseek-v3.2_vllm.git
```

### GLM-4.7
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-20260220`  
**框架**：vLLM, PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it \
    --shm-size 60g \
    --network=host \
    --name glm-4.7 \
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
pip install -U transformers
```

**启动/推理**：
```bash
vllm serve ZhipuAI/GLM-4.7 \
     --distributed-executor-backend ray \
     --tensor-parallel-size 8 \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 1 \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --gpu-memory-utilization 0.95 \
     --port 8001 \
     --served-model-name glm-4.7
```

**测试**：
```bash
curl http://localhost:8001/v1/chat/completions   \
    -H "Content-Type: application/json"  \
    -d '{
        "model": "glm-4.7",
        "messages": [
            {
                "role": "user",
                "content": "请介绍一下你自己"
            }
        ],
        "temperature": 0.7
    }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/ZhipuAI/GLM-4.7

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-20260220
```

### GLM4-9B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name {docker_name} --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
cd /your_code_path/glm-4_pytorch
pip install -r inference/requirements.txt
pip install transformers==4.42.0
```

**启动/推理**：
```bash
python infer_glm4.py --model_path /path/of/model/ --message "你好"
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/ZhipuAI/glm-4-9b

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://developer.sourcefind.cn/codes/modelzoo/glm-4_pytorch.git
```

### Kimi-K2.5
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.10rc0-ubuntu22.04-dtk26.04-py3.10-20260428`  
**框架**：SGLang  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it \
    --shm-size 200g \
    --network=host \
    --name Kimi-K2.5 \
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
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.10rc0-ubuntu22.04-dtk26.04-py3.10-20260428 bash
```

**依赖/环境**：
```bash
pip install pycountry
```

**测试**：
```bash
## client访问
curl http://<DCU_NODE_IP>:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "/path/to/model/Kimi-K2.5 ",
      "messages": [
        {"role": "user", "content": "你好，请用一句话介绍你自己。"}
      ],
      "max_tokens": 128,
      "temperature": 0.6
    }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/moonshotai/Kimi-K2.5

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.10rc0-ubuntu22.04-dtk26.04-py3.10-20260428

#serve启动
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export SGLANG_TORCH_PROFILER_DIR=/workspace/profiling
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1

python3 -m sglang.launch_server \
  --model-path /path/to/model/Kimi-K2.5 \
  --kv-cache-dtype fp8_e4m3 \
  --host $(hostname -I | awk '{print $1}') \
  --port 30000 \
  --trust-remote-code \
  --page-size 64 \
  --dist-init-addr $(hostname -I | awk '{print $1}'):5001 \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --nnodes 1 \
  --node-rank 0 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --mem-fraction-static 0.98 \
  --attention-backend dcu_mla \
  --enable-torch-compile \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size -1 \
  --max-running-requests 512 \
  --context-length 65536
```

### Llama-2-70b-hf
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1-ubuntu22.04-`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

### Llama-3-8/70B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：torchrun  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro --shm-size=80G --privileged=true --device=/dev/kfd --device=/dev/dri/ --group-add video --name docker_name harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip install -r requirements.txt
```

**启动/推理**：
```bash
torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir Meta-Llama-3-8B/original/ \
    --tokenizer_path Meta-Llama-3-8B/original/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```

**其他命令**：
```bash
访问魔搭社区下载：
https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B
https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-70B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://developer.sourcefind.cn/codes/modelzoo/llama3_pytorch.git
```

### Llama3.1-8B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：torchrun  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro --shm-size=80G --privileged=true --device=/dev/kfd --device=/dev/dri/ --group-add video --name docker_name harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip install -r requirements.txt
```

**启动/推理**：
```bash
torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir Meta-Llama-3.1-8B/original/ \
    --tokenizer_path Meta-Llama-3.1-8B/original/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://developer.sourcefind.cn/codes/modelzoo/llama3_pytorch.git
```

### PP-OCRv5
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:paddlepaddle3.2.0-dtk25.04`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:paddlepaddle3.2.0-dtk25.04 bash
```

**启动/推理**：
```bash
export PADDLE_PDX_DISABLE_DEV_MODEL_WL=1
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png     --use_doc_orientation_classify False     --use_doc_unwarping False     --use_textline_orientation False     --save_path ./output     --device DCU
python run_inference.py
```

**其他命令**：
```bash
访问社区下载：https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:paddlepaddle3.2.0-dtk25.04

docker exec -it test bash
```

### Qwen2-7B/72B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size=1024G -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal:/opt/hyhal --privileged=true --device=/dev/kfd --device=/dev/dri/ --group-add video --name qwen2_72B_pytorch harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip install https://download.sourcefind.cn:65024/file/4/vllm/DAS1.8/vllm-0.9.2+das.opt1.dtk2604.torch251-cp310-cp310-manylinux_2_28_x86_64.whl
cd /path/your_code_data/
cd LLaMA-Factory
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/  --trusted-host mirrors.aliyun.com
pip install e . -i https://mirrors.aliyun.com/pypi/simple/  --trusted-host mirrors.aliyun.com
pip install "transformers==4.57.0"
```

**启动/推理**：
```bash
#注意：根据自己的模型切换文件中的模型位置并调整其他参数
cd /path/your_code_data/
python ./inference_vllm/Qwen2_7B_inference.py
```

**其他命令**：
```bash
访问魔搭社区下载：
https://www.modelscope.cn/models/Qwen/Qwen2-7B
https://www.modelscope.cn/models/Qwen/Qwen2-72B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://developer.sourcefind.cn/codes/modelzoo/qwen2_pytorch.git
```

**注意**：
- #注意：根据自己的模型切换文件中的模型位置并调整其他参数

### Qwen2.5-7B/14B/32B/72B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1--ubuntu22.04-dtk26.04-py3.10-20260409`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --name qwen2.5_vllm --privileged --shm-size=64G  --device=/dev/kfd --device=/dev/dri/ --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ipc=host --network host --group-add video -v /opt/hyhal:/opt/hyhal -v <Host Path>:<Container Path> harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1--ubuntu22.04-dtk26.04-py3.10-20260409 /bin/bash
```

**启动/推理**：
```bash
vllm serve /your/model/path --enforce-eager --dtype float16 --trust-remote-code --tensor-parallel-size 4
```

**测试**：
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/your/model/path",
        "prompt": "What is deep learning?",
        "max_tokens": 700,
        "temperature": 0
    }'
```

**其他命令**：
```bash
访问魔搭社区下载：
https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct
https://www.modelscope.cn/models/Qwen/Qwen2.5-14B-Instruct
https://www.modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct
https://www.modelscope.cn/models/Qwen/Qwen2.5-72B-Instruct

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1--ubuntu22.04-dtk26.04-py3.10-20260409
```

### Qwen3-235B-A22B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1-ubuntu22.04-`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

### Qwen3-30B-A3B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1-ubuntu22.04-`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

### Qwen3-32B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10`  
**框架**：SGLang  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -itd     --name <container>     --shm-size=16G     --group-add video     --cap-add=SYS_PTRACE     --security-opt seccomp=unconfined     --network=host     --ipc=host     --privileged     --device=/dev/kfd     --device=/dev/mkfd     --device=/dev/dri     -v /opt/hyhal:/opt/hyhal:ro     -v /root/.ssh:/root/.ssh -v /public:/public -v /data:/data -v /home:/home   harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10     /bin/bash
```

**启动/推理**：
```bash
sglang serve --model-path /public/Qwen3-32B/ --tp 2 --attention-backend fa3 --page-size 64
```

**测试**：
```bash
curl http://0.0.0.0:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
"model": "/public/Qwen3-32B/",
"max_tokens": 100,
"messages":[{"role":"user","content": "树下一个猴，树上骑个猴，一共几个猴？"}],
"stream": false
}'
```

**其他命令**：
```bash
harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10

docker exec -it <container> bash
```

### Qwen3-32B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1-ubuntu22.04-`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

### Qwen3-Coder
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305`  
**框架**：vLLM, PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name qwen3-coder --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v {/path/your_code_data/}:{/path/your_code_data/} -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305 bash
```

**依赖/环境**：
```bash
cd /your_code_path/qwen3-coder_vllm
pip install transformers==4.51.3
```

**启动/推理**：
```bash
export ALLREDUCE_STREAM_WITH_COMPUTE=1
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --trust-remote-code --dtype bfloat16 --max-seq-len-to-capture 32768 -tp 4 --gpu-memory-utilization 0.85 --override-generation-config '{"temperature": 0.7, "top_p":0.8, "top_k":20, "repetition_penalty": 1.05}' --max-model-len 32768
```

**测试**：
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "quare the number 1024."
                }
            ]
    }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Qwen/Qwen3-Coder-30B-A3B-Instruct/

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305
```

### Qwen3.5-122B-A10B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.15.1-ubuntu22.04-`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

### Qwen3.6-27B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -itd \
    --shm-size 256g \
    --network=host \
    --name qwen3.6 \
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
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423 bash
```

**启动/推理**：
```bash
vllm serve Qwen/Qwen3.6-27B \
    --port 8001 \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

**测试**：
```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-27B",
    "messages": [
      {"role": "user", "content": "帮我查询上海的天气"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "查询城市天气",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ]
  }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Qwen/Qwen3.6-27B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423

docker exec -it qwen3.6 bash
```

### telechat2-7B/35B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --ipc=host --shm-size=16G --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal:/opt/hyhal --privileged=true --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri/ --group-add video --name telechat2 harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305 bash
```

**启动/推理**：
```bash
vllm serve /root/data/telechat/Telechat2-7B-32K/dir/  -tp 4 --max-model-len 8192 --port 8001 --trust-remote-code  --dtype float16
```

**测试**：
```bash
curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/path/to/TeleChat2-7B-32K/",
        "messages": [
            {
                "role": "user",
                "content": "请介绍下你自己。"
            }
        ],
        "temperature": 1.0,
        "chat_template_kwargs": {
            "thinking": true
        }
    }'
```

**其他命令**：
```bash
访问魔搭社区下载：
https://www.modelscope.cn/models/TeleAI/TeleChat2-7B-32K
https://www.modelscope.cn/models/TeleAI/TeleChat2-35B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk26.04-rc1-py3.10-20260305

# 环境变量
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_LAUNCH_MODE=GROUP
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16

# vllm0.6.2新增环境变量
export NCCL_P2P_LEVEL=SYS
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export VLLM_RANK0_NUMA=0
export VLLM_RANK1_NUMA=1
export VLLM_RANK2_NUMA=2
export VLLM_RANK3_NUMA=3
export VLLM_RANK4_NUMA=4
export VLLM_RANK5_NUMA=5
export VLLM_RANK6_NUMA=6
export VLLM_RANK7_NUMA=7

# 设置使用的卡ID
# export HIP_VISIBLE_DEVICES=4,5,6,7
# export LD_LIBRARY_PATH=/root/data/telechat/benchmark_test/rocblas-install/lib:$LD_LIBRARY_PATH
```
