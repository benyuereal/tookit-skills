# 音频/多模态 部署配方（2 个模型）

> 来源：《<某能源集团客户>2026年网络设备项目测试项汇总》「模型适配支持度」sheet 部署文档交付链接列（标黄=BW1100 已验证）。

> 命令均摘自各模型《海光 DCU 模型部署手册》原文，照抄即跑；路径/卡号/容器名按实际调整。

### Qwen2-Audio
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260510-2242`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro --shm-size=64G --privileged=true --network=host --device=/dev/kfd --device=/dev/dri/ --group-add video --name qwen2-audio harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260510-2242 bash
```

**依赖/环境**：
```bash
git clone https://developer.sourcefind.cn/codes/modelzoo/qwen2-audio_pytorch.git
cd qwen2-audio_pytorch/demo
pip install-r requirements_web_demo.txt -i https://mirrors.aliyun.com/pypi/simple
pip install "transformers==4.51.3"-i https://mirrors.aliyun.com/pypi/simple/
```

**启动/推理**：
```bash
python qwen2_audio_inference.py
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Qwen/Qwen2-Audio-7B-Instruct

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260510-2242
```

### Qwen2.5-Omni-7B
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260510-2242`  
**框架**：vLLM  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size=64G -v $PWD/Qwen2.5-Omni:/home/Qwen2.5-Omni -v /opt/hyhal:/opt/hyhal:ro --privileged=true --device=/dev/kfd --device=/dev/dri/ --group-add video --name qomni harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260510-2242 bash
```

**启动/推理**：
```bash
vllm serve /models/Qwen2.5-Omni-7B/ \
--trust-remote-code \
--max-model-len 32768 \
--served-model-name qwen-omni \
--dtype bfloat16 \
--tensor-parallel-size 1 \
--gpu-memory-utilization 0.9
```

**测试**：
```bash
curl http://localhost:8000/v1/chat/completions   \
  -H "Content-Type: application/json" \
  -d '{
      "model": "qwen-omni",
      "messages": [
        {
        "role": "user",
        "content": "介绍一下自己。"
        }
      ]
      }'
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Qwen/Qwen2.5-Omni-7B

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260510-2242
```
