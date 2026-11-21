# 科学计算 部署配方（7 个模型）

> 来源：《<某能源集团客户>2026年网络设备项目测试项汇总》「模型适配支持度」sheet 部署文档交付链接列（标黄=BW1100 已验证）。

> 命令均摘自各模型《海光 DCU 模型部署手册》原文，照抄即跑；路径/卡号/容器名按实际调整。

### DeePMD
**镜像**：`image.sourcefind.cn:5000/dcu/admin/base/custom:deepmd`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it -v /path/your_code_data/:/path/your_code_data/ --shm-size=32G --privileged=true --device=/dev/kfd --device=/dev/dri/ --group-add video --name docker_name image.sourcefind.cn:5000/dcu/admin/base/custom:deepmd bash
```

**依赖/环境**：
```bash
source /root/env_disc.sh
git clone https://developer.sourcefind.cn/codes/modelzoo/bladedisc_deepmd.git
cd bladedisc_deepmd
pip install-r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

**启动/推理**：
```bash
cd /root/test_case/water    bash run.sh
```

**其他命令**：
```bash
docker pull image.sourcefind.cn:5000/dcu/admin/base/custom:deepmd

disc.enable()
```

**注意**：
- 如果需要关闭BladeDISC加速，需要注释掉main.py文件中的如下内容：

### ESM
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**其他命令**：
```bash
访问社区下载：https://github.com/evolutionaryscale/esm/tree/main

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

docker exec -it test bash
```

### FNO
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
docker exec -it test bash

pip install neuraloperator
```

**其他命令**：
```bash
访问社区下载：https://github.com/neuraloperator/neuraloperator

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

import torch
from neuralop.models import FNO

def test_neuraloperator_gpu():
    # 1. 检查 GPU 可用性
    if not torch.cuda.is_available():
        print("❌ 未检测到可用的 GPU，请检查 CUDA 安装或 PyTorch 版本。")
        return

    device = torch.device("cuda")
    print(f"✅ 检测到 GPU: {torch.cuda.get_device_name(0)}")

    # 2. 创建 FNO 模型并移至 GPU
    model = FNO(
        n_modes=(32, 32),      # 傅里叶模式的分辨率
        hidden_channels=32,    # 隐藏层通道数
        in_channels=1,         # 输入通道数
        out_channels=1         # 输出通道数
    ).to(device)

    print(f"✅ FNO 模型已成功加载至 GPU")
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 创建随机输入 (batch_size=2, channels=1, height=64, width=64)
    x = torch.randn(2, 1, 64, 64, device=device)
    y = torch.randn(2, 1, 64, 64, device=device)

    # 4. 前向传播测试
    model.train()
    try:
        output = model(x)
        print(f"✅ 前向传播成功 | 输入shape: {x.shape} → 输出shape: {output.shape}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return

    # 5. 反向传播测试（验证梯度计算在GPU上正常）
    try:
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        print(f"✅ 反向传播成功 | Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ 反向传播失败: {e}")
        return

    # 6. 显存占用检查
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    print(f"✅ GPU 显存占用: {allocated:.1f} MB")
    print("\n🎉 NeuralOperator GPU 验证全部通过！")

if __name__ == "__main__":
    test_neuraloperator_gpu()
```

### Openfold
**镜像**：`image.sourcefind.cn:5000/dcu/admin/base/custom:deepmd`  
**框架**：TensorRT  
**原文**：[内部文档](<internal-doc-link>)

**依赖/环境**：
```bash
git clone https://github.com/aqlaboratory/openfold.git
cd openfold
# 安装python依赖
pip install biopython ml-collections pytorch-lightning absl-py dm-haiku dm-tree modelcif typing-extensions openmm pdbfixer -i https://mirrors.aliyun.com/pypi/simple/
```

**测试**：
```bash
mkdir-p resources/openfold_params
cd resources/openfold_params
# 
有模板模式
wget https://hf-mirror.com/nz/OpenFold/resolve/main/finetuning_ptm_2.pt
```

**其他命令**：
```bash
docker pull image.sourcefind.cn:5000/dcu/admin/base/custom:deepmd

images=image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.4.1-ubuntu22.04-dtk25.04.1-py3.11
cname=pytorch-onescience
docker run -it \
--shm-size 256g \
--network=host \
--name=$cname \
--privileged \
--device=/dev/kfd \
--device=/dev/dri \
--device=/dev/mkfd \
--ipc=host \
--group-add video \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
-u root --ulimit stack=-1:-1 --ulimit memlock=-1:-1 \
-v /opt/hyhal/:/opt/hyhal/:ro \
-v /data/models:/data/models \
-v <deploy_home>:/workspace \
image.sourcefind.cn:5000/dcu/admin/base/custom:deepmd /bin/bash

# 跳过 TensorRT 导入
cp openfold/utils/tensorrt_lazy_compiler.py
openfold/utils/tensorrt_lazy_compiler.py.bak
sed -i 's/^import cuda.cudart as cudart/# import cuda.cudart as cudart/'
openfold/utils/tensorrt_lazy_compiler.py
sed -i 's/^import tensorrt as trt/# import tensorrt as trt/'
openfold/utils/tensorrt_lazy_compiler.py
sed -i '31,40s/^/# /' openfold/utils/tensorrt_lazy_compiler.py
sed -i '/G_LOGGER/s/^/# /' openfold/utils/tensorrt_lazy_compiler.py
# 替换自定义 CUDA 内核为 PyTorch 原生实现
cp openfold/utils/kernel/attention_core.py
openfold/utils/kernel/attention_core.py.bak
cat > openfold/utils/kernel/attention_core.py << 'EOF'
import torch
import torch.nn.functional as F

def attention_core(q, k, v, bias1=None, bias2=None):
    scale = q.shape[-1] ** -0.5
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    if bias1 is not None:
        logits = logits + bias1
    if bias2 is not None:
        logits = logits + bias2
    weights = F.softmax(logits, dim=-1)
    return torch.matmul(weights, v)
EOF

#跳过 structure\_module 中的 CUDA 内核调用
sed -i 's/attn_core_inplace_cuda =
importlib.import_module("attn_core_inplace_cuda")/attn_core_inplace_cuda =
None/' openfold/model/structure_module.py
sed -i 's/inplace_safe=inplace_safe/inplace_safe=False/'
openfold/model/structure_module.py
```

### PINN
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**其他命令**：
```bash
访问社区下载：https://github.com/maziarraissi/PINNs

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

docker exec -it test bash
```

### alphafold2
**镜像**：`image.sourcefind.cn:5000/dcu/admin/base/custom:alphafold2-dtk24.04.1-py310`  
**原文**：[内部文档](<internal-doc-link>)

**其他命令**：
```bash
# 使用该方法不需要下载本仓库，镜像中已包含可运行代码，只需要挂载相应的数据文件

docker pull image.sourcefind.cn:5000/dcu/admin/base/custom:alphafold2-dtk24.04.1-py310

docker run --shm-size 100g --network=host --name=alphafold2 --privileged --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v 本地数据地址:镜像数据地址 -v /opt/hyhal:/opt/hyhal:ro -it <your IMAGE ID> bash

cd /app/alphafold2/dcu_build

# 此过程较为耗时
bash build.sh

source env.sh

export PATH=/app/softwares/hh-suite/build/bin:/app/softwares/hh-suite/build/scripts:$PATH

$DOWNLOAD_DIR/                             
    bfd/  
        bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt_hhm.ffindex
        bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt_hhm.ffdata 
        bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt_cs219.ffindex                           
        ...
    mgnify/                                
        mgy_clusters_2022_05.fa
    params/                                
        params_model_1.npz
        params_model_2.npz
        params_model_3.npz
        ...
    pdb70/                                
        pdb_filter.dat
        pdb70_hhm.ffindex
        pdb70_hhm.ffdata
        ...
    pdb_mmcif/                            
        mmcif_files/
            100d.cif
            101d.cif
            101m.cif
            ...
        obsolete.dat
    pdb_seqres/                            
        pdb_seqres.txt
    small_bfd/                           
        bfd-first_non_consensus_sequences.fasta
    uniref30/                            
        UniRef30_2021_03_hhm.ffindex
        UniRef30_2021_03_hhm.ffdata
        UniRef30_2021_03_cs219.ffindex
        ...
    uniprot/                               
        uniprot.fasta
    uniref90/                             
        uniref90.fasta

./scripts/download_all_data.sh 数据集下载目录
```

**注意**：
- 注意：在运行前请修改相应脚本中的参数。

### powerflownet
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**其他命令**：
```bash
访问社区下载：https://github.com/StavrosOrf/PoweFlowNet

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

docker exec -it test bash
```
