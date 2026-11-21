# CV（检测/分割/分类） 部署配方（21 个模型）

> 来源：《<某能源集团客户>2026年网络设备项目测试项汇总》「模型适配支持度」sheet 部署文档交付链接列（标黄=BW1100 已验证）。

> 命令均摘自各模型《海光 DCU 模型部署手册》原文，照抄即跑；路径/卡号/容器名按实际调整。

### BiRefNet
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：PyTorch + transformers（trust_remote_code）  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 32g --network=host --name birefnet --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /root/BiRefNet:/root/BiRefNet -v /root/birefnet_weights:/root/birefnet_weights -v /opt/hyhal:/opt/hyhal:ro harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# torch 镜像已预装 DCU 适配版，勿 pip install torch 覆盖；numpy 锁 <2 防被 kornia 等升级
pip install transformers timm kornia einops "numpy<2"
```

**启动/推理**：
```bash
git clone https://github.com/ZhengPeng7/BiRefNet.git /root/BiRefNet
mkdir -p /root/birefnet_weights/birefnet && cd /root/birefnet_weights/birefnet
wget -O model.safetensors https://hf-mirror.com/ZhengPeng7/BiRefNet/resolve/main/model.safetensors
for f in birefnet.py BiRefNet_config.py config.json handler.py requirements.txt; do
  wget -O $f https://hf-mirror.com/ZhengPeng7/BiRefNet/resolve/main/$f
done

source /opt/dtk/env.sh
HIP_VISIBLE_DEVICES=4 python /root/verify_birefnet.py
```

**其他命令**：
```python
# 推理要点（verify_birefnet.py 核心）：
# 1) 输入 H/W 必须能被 32 整除（einops rearrange 限制），先 resize 到 32 倍数
# 2) 输出取 model(x)[-1].sigmoid()，再 interpolate 回原尺寸
model = AutoModelForImageSegmentation.from_pretrained(WEIGHT, trust_remote_code=True).to(DEVICE).eval()
w2, h2 = w // 32 * 32, h // 32 * 32
x = torch.from_numpy(np.asarray(im_pil.resize((w2, h2)), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(DEVICE)
preds = model(x)[-1].sigmoid()
```

### CycleGAN
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 60g --network=host --name cyclegan --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /opt/hyhal/:/opt/hyhal/:ro -v /path/your_code_path/:/path/your_code_path/ harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip install dominate
pip install wandb
```

**其他命令**：
```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git

bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

### DepthAnythingV3 (DA3-SMALL)
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 32g --network=host --name da3 --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /root/Depth-Anything-3:/root/Depth-Anything-3 -v /root/da3_weights:/root/da3_weights -v /opt/hyhal:/opt/hyhal:ro harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# 不用 pip install -e .（会拉 xformers/open3d/pycolmap 等 DCU 装不了的重依赖）
# 只装纯推理依赖 + PYTHONPATH 指向 src；torch 镜像已预装，勿覆盖；numpy 锁 <2
pip install einops huggingface_hub imageio opencv-python omegaconf safetensors addict evo "numpy<2"
```

**DCU 适配（1 处代码修改）**：
```python
# src/depth_anything_3/api.py 顶部：导出链依赖 moviepy/pycolmap/trimesh（DCU 装不了），
# 纯推理（不传 export_dir）用不到，改容错导入
try:
    from depth_anything_3.utils.export import export
except ImportError:
    export = None
# _export_results 开头加保护：
#   if export is None: raise ImportError("3D 导出依赖未安装，纯推理请不传 export_dir。")
```

**启动/推理**：
```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git /root/Depth-Anything-3
mkdir -p /root/da3_weights/da3 && cd /root/da3_weights/da3
wget -O config.json https://hf-mirror.com/depth-anything/DA3-SMALL/resolve/main/config.json
wget -O model.safetensors https://hf-mirror.com/depth-anything/DA3-SMALL/resolve/main/model.safetensors

source /opt/dtk/env.sh
HIP_VISIBLE_DEVICES=4 PYTHONPATH=/root/Depth-Anything-3/src python /root/verify_da3.py
```

**其他命令**：
```python
# 推理要点（verify_da3.py 核心）：
model = DepthAnything3.from_pretrained(WEIGHT).to(device=DEVICE)
prediction = model.inference([IM_FILE])   # 不传 export_dir 即纯推理
depth, conf = prediction.depth[0], prediction.conf[0]
```

### DETR
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal:/opt/hyhal:ro --shm-size=32G --privileged=true --device=/dev/kfd --device=/dev/dri/ --group-add video --name docker_name harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
cd /your_code_path/detr_pytorch
pip install -r requirements.txt
```

**其他命令**：
```bash
官方文档model zoo章节下载：https://github.com/facebookresearch/detr

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://developer.sourcefind.cn/codes/modelzoo/detr_pytorch.git
```

### DINO
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
docker exec -it test bash

pip install simplejson addict yapf opencv-python-headless modelscope
pip install groundingdino-py

pip install 'numpy<2.0'
wget -O dog-3.jpg https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=800
```

**其他命令**：
```bash
访问社区下载：https://www.modelscope.cn/models/AI-ModelScope/GroundingDINO

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

import cv2
from modelscope.pipelines import pipeline
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('AI-ModelScope/GroundingDINO')

pipe = pipeline('grounding-dino-task', model=model_dir, trust_remote_code=True)
inputs = {
    "IMAGE_PATH":"dog-3.jpg",
    "TEXT_PROMPT":"chair . person . dog .",
    "BOX_TRESHOLD":0.35,
    "TEXT_TRESHOLD":0.25
}
output = pipe(inputs)
cv2.imwrite("annotated_image.jpg", output["annotated_frame"])
```

### MobilenetV2-1.0
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：torchrun  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --name docker_name -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal:/opt/hyhal:ro --network=host --shm-size=32G --privileged=true --device=/dev/kfd --ipc=host --device=/dev/dri/ --group-add video harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 /bin/bash
```

**依赖/环境**：
```bash
pip install -r requirments.txt
export HIP_VISIBLE_DEVICES=0
mkdir logs #若已经创建不需要重新创建
python driver.py
```

**其他命令**：
```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

https://developer.sourcefind.cn/codes/modelzoo/mobilenet_v2_pytorch.git

mkdir logs #若已经创建不需要重新创建
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 driver.py
```

### MobilenetV3large100
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --name docker_name -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal:/opt/hyhal:ro --network=host --shm-size=32G --privileged=true --device=/dev/kfd --ipc=host --device=/dev/dri/ --group-add video harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 /bin/bash
```

**其他命令**：
```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://developer.sourcefind.cn/codes/modelzoo/mobilenetv3_pytorch.git

# 运行前需设置一下数据集路径
bash train.sh
```

### ResNet101
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --name resnet --privileged --shm-size=64G  --device=/dev/kfd --device=/dev/dri/ --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --network host --group-add video -v /opt/hyhal:/opt/hyhal -v <Host Path>:<Container Path> harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**测试**：
```bash
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

**其他命令**：
```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

import torch
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()

# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### ResNet18
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --name resnet --privileged --shm-size=64G  --device=/dev/kfd --device=/dev/dri/ --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --network host --group-add video -v /opt/hyhal:/opt/hyhal -v <Host Path>:<Container Path> harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**测试**：
```bash
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

**其他命令**：
```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()

# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### ResNet50
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --name resnet --privileged --shm-size=64G  --device=/dev/kfd --device=/dev/dri/ --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --network host --group-add video -v /opt/hyhal:/opt/hyhal -v <Host Path>:<Container Path> harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**测试**：
```bash
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

**其他命令**：
```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

import torch
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()

# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### RT-DETRv2
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 32g --network=host --name rtdetrv2 --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /root/rtdetrv2_pytorch:/root/rtdetrv2_pytorch -v /root/rtdetrv2_weights:/root/rtdetrv2_weights -v /opt/hyhal:/opt/hyhal:ro harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# requirements.txt 是 >= 宽松约束，镜像预装 DCU 版 torch 已满足，pip 跳过不覆盖
pip install -r requirements.txt
```

**DCU 适配（2 处代码修改）**：
```python
# 1) src/zoo/rtdetr/__init__.py：mamba 系 import 改容错（mamba_ssm 需 CUDA 编译，DCU 装不了；
#    ResNet 系 RT-DETRv2 不需要 mamba，仅 rtdemrv2 需要）
try:
    from .rtdemrv2_mamba2_encoder import Mamba2_HybridEncoder
    from .rtdemrv2_mamba1_encoder import Mamba_HybridEncoder
    from .rtdemrv2_mamba2_decoder import Mamba2_Decoder
except ImportError:
    Mamba2_HybridEncoder = Mamba_HybridEncoder = Mamba2_Decoder = None
# 2) src/zoo/rtdetr/rtdetr.py：eval 分支 latency 统计加 track 开关
#    （deploy 脚本不传 start_time/latency 列表，原代码 float-None 报错）
#    track = start_time is not None and backbone_latency_list is not None
```

**启动/推理**：
```bash
git clone https://github.com/zheli-hub/RT-DETRv2.git
cd RT-DETRv2/rtdetrv2_pytorch
wget -O /root/rtdetrv2_weights/rtdetrv2_r18vd_120e_coco.pth \
  https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_120e_coco.pth

source /opt/dtk/env.sh
HIP_VISIBLE_DEVICES=4 PYTHONPATH=/root/rtdetrv2_pytorch \
python references/deploy/rtdetrv2_torch.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
  -r /root/rtdetrv2_weights/rtdetrv2_r18vd_120e_coco.pth \
  --im-file bus.jpg --device cuda:0
```

**其他命令**：
```bash
# 选用 ResNet 系（rtdetrv2_r18vd_120e_coco.yml），不依赖 mamba_ssm
# 仓库 rtdemrv2（Mamba 系）依赖 CUDA 编译的 mamba_ssm，DCU 不可用
```

### SwinTransformer-base
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：torchrun  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro --shm-size=80G --privileged=true --device=/dev/kfd --device=/dev/dri/ --group-add video --name docker_name harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip install yacs
```

**启动/推理**：
```bash
# Swin-Base模型
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin/swin_base_patch4_window7_224.yaml --resume swin_base_patch4_window7_224.pth --data-path <imagenet-path>
```

**其他命令**：
```bash
访问官方github下载：https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md

docker pull  harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://github.com/microsoft/Swin-Transformer.git
cd Swin-Transformer
```

### SwinTransformer-tiny
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：torchrun  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro --shm-size=80G --privileged=true --device=/dev/kfd --device=/dev/dri/ --group-add video --name docker_name harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip install yacs
```

**启动/推理**：
```bash
# Swin-Tiny模型
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path <imagenet-path>
```

**其他命令**：
```bash
访问官方github下载：https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md

docker pull  harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://github.com/microsoft/Swin-Transformer.git
cd Swin-Transformer
```

### aligned-reid
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**其他命令**：
```bash
访问社区下载：https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

docker exec -it test bash

git clone https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch.git
cd AlignedReID-Re-Production-Pytorch/
#修改部分代码
aligned_reid/model/resnet.py
文件内容替换为以下内容
```

### convnextsmall
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**其他命令**：
```bash
访问社区下载：https://www.modelscope.cn/models/timm/convnext_small.in12k

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

docker exec -it test bash
```

### sam3
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name sam3 --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
docker exec -it sam3 bash

git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install pycocotools -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/facebook/sam3

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

import os
import sys
import torch
import requests
from PIL import Image
from io import BytesIO
import warnings

# 忽略无关及误导性警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*memory efficient attention.*")
warnings.filterwarnings("ignore", message=".*sdpa adopt the new interface.*")
warnings.filterwarnings("ignore", message=".*Casting complex values.*")

# ======================== 配置区 ========================
MODEL_DIR = "/public/sam3"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "sam3.pt")
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
TEXT_PROMPT = "a cat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ✅ 仅在计算时使用 bfloat16，不改变模型参数本身的存储精度
COMPUTE_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
# =======================================================


def verify_image_inference():
    """验证 SAM3 图像分割功能"""
    print("=" * 50)
    print(f"🔧 运行设备: {DEVICE} | 计算精度: {COMPUTE_DTYPE}")
    print(f"📁 模型权重文件: {CHECKPOINT_PATH}")
    print("🖼️  开始验证图像推理...\n")

    if not os.path.isfile(CHECKPOINT_PATH):
        print(f"  ❌ 权重文件不存在: {CHECKPOINT_PATH}")
        return False

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # 1. 加载模型 —— ✅ 不再指定 dtype，保留原始精度
        print("  正在加载图像模型...")
        model = build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH)
        model.to(DEVICE)  # 仅移动到设备，不做类型转换
        processor = Sam3Processor(model)
        print("  ✅ 模型加载成功\n")

        # 2. 下载网络图片
        print(f"  正在下载测试图片: {IMAGE_URL}")
        response = requests.get(IMAGE_URL, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"  ✅ 图片下载成功, 尺寸: {image.size}\n")

        # 3. 推理 —— ✅ 整个流程包裹在统一的 autocast 中
        print(f"  文本提示: '{TEXT_PROMPT}'")
        with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE, enabled=(DEVICE == "cuda")):
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(
                state=inference_state,
                prompt=TEXT_PROMPT
            )

        # 4. 校验输出结果
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        print(f"\n  📊 推理结果:")
        print(f"     masks shape : {tuple(masks.shape)}")
        print(f"     boxes shape : {tuple(boxes.shape)}")
        print(f"     scores      : {scores.tolist()}")
        print(f"     检测到 {len(scores)} 个目标")

        assert masks.dim() >= 2, "masks 维度异常"
        assert len(scores) > 0, "未检测到任何目标"
        print("\n  ✅ 图像推理验证通过!")
        return True

    except Exception as e:
        print(f"\n  ❌ 验证失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_image_inference()
    print("=" * 50)
    if success:
        print("🎉 SAM3 图像模型验证完成!")
    else:
        print("⚠️  验证未通过，请检查上方错误信息。")
        sys.exit(1)
```

### segformer-b0-finetuned-cityscapes
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：PyTorch + transformers  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 32g --network=host --name segformer --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /root/segformer_weights:/root/segformer_weights -v /opt/hyhal:/opt/hyhal:ro harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# torch 镜像已预装 DCU 适配版，勿 pip install torch 覆盖；numpy 锁 <2
pip install transformers "numpy<2"
```

**DCU 适配（1 处：手动加载权重）**：
```python
# transformers 4.57 的 from_pretrained 内部 torch.load 要求 torch>=2.6，DCU 为 2.5.1，
# 故 config 走 from_pretrained，权重手动 torch.load + load_state_dict
cfg = SegformerConfig.from_pretrained(WEIGHT)
model = SegformerForSemanticSegmentation(cfg)
state = torch.load(os.path.join(WEIGHT, "pytorch_model.bin"), map_location="cpu", weights_only=True)
model.load_state_dict(state)
processor = SegformerImageProcessor.from_pretrained(WEIGHT)
```

**启动/推理**：
```bash
mkdir -p /root/segformer_weights/segformer && cd /root/segformer_weights/segformer
wget -O config.json https://hf-mirror.com/nvidia/segformer-b0-finetuned-cityscapes-1024-1024/resolve/main/config.json
wget -O preprocessor_config.json https://hf-mirror.com/nvidia/segformer-b0-finetuned-cityscapes-1024-1024/resolve/main/preprocessor_config.json
wget -O pytorch_model.bin https://hf-mirror.com/nvidia/segformer-b0-finetuned-cityscapes-1024-1024/resolve/main/pytorch_model.bin

source /opt/dtk/env.sh
HIP_VISIBLE_DEVICES=4 python /root/verify_segformer.py
```

**其他命令**：
```bash
# 权重 config.json 的 id2label 是 0-18（非 cityscapes 原始 ID）：
# 0=road 1=sidewalk 2=building 3=wall 4=fence 5=pole 6=traffic light 7=traffic sign
# 8=vegetation 9=terrain 10=sky 11=person 12=rider 13=car 14=truck 15=bus
# 16=train 17=motorcycle 18=bicycle
```

### yolov10
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size=32G -v $PWD/yolov10:/home/yolov10 -v /opt/hyhal:/opt/hyhal:ro --privileged=true --device=/dev/kfd --device=/dev/dri/ --group-add video --name yolov10 harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
cd /home/yolov10
pip install -r requirements.txt # requirements.txt
```

**启动/推理**：
```bash
python predict.py
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/AI-ModelScope/Yolov10

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://developer.sourcefind.cn/codes/modelzoo/yolov10_pytorch.git
```

### yolov11
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it \
--network=host \
--hostname=localhost \
--name=yolo \
-v /opt/hyhal:/opt/hyhal:ro \
-v /mnt/data:/mnt/data \
--ipc=host \
--device=/dev/kfd \
--device=/dev/mkfd \
--device=/dev/dri \
--shm-size=64G \
--privileged \
--group-add video \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 \
/bin/bash
```

**依赖/环境**：
```bash
pip install ultralytics -i https://mirrors.aliyun.com/pypi/simple/
pip install 'numpy<2' -i https://mirrors.aliyun.com/pypi/simple/
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Ultralytics/YOLO11

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

from ultralytics import YOLO
import os

# 1. 加载模型 
model = YOLO("/workspace/YOLO11/yolo11l.pt")

# 2. 定义输入图像路径
img_path = "/workspace/bus.jpg"  # 替换为你实际的图像路径

# 3. 执行预测并保存结果
# save=True 会自动将带有检测框的图像保存到 ./runs/detect/predict/ 目录下
# conf=0.25 是默认的置信度阈值，可根据需要调整
results = model.predict(
    source=img_path,
    save=True,
    project="runs/detect",  # 保存的根目录
    name="exp",             # 保存的子文件夹名称
    exist_ok=True           # 如果文件夹已存在则覆盖，不报错
)

# 4. (可选) 在控制台打印检测结果信息
for result in results:
    print(f"图像路径: {result.path}")
    print(f"检测到 {len(result.boxes)} 个目标:")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        print(f"  - 类别: {cls_name}, 置信度: {conf:.2f}")
```

### yolov5
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it -v /path/your_code_data/:/path/your_code_data/ --shm-size=32G --privileged=true --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /opt/hyhal:/opt/hyhal:ro --name docker_name harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**测试**：
```bash
# 提前下载字体避免连接问题
wget https://ultralytics.com/assets/Arial.ttf

 python detect.py --source bus.jpg --weights /data2/yolo/yolov5s.pt --img 640
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/yolo_master/YOLOv5/summary

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

git clone https://developer.sourcefind.cn/codes/modelzoo/yolov5_pytorch.git
```

### yolov8
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/migraphx:5.2.0-ubuntu22.04-dtk26.04-py3.11`  
**框架**：MIGraphX  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run --shm-size 16g --network=host --name=yolov8_migraphx --privileged --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /opt/hyhal:/opt/hyhal:ro -v $PWD/yolov8_migraphx:/home/yolov8_migraphx -it harbor.sourcefind.cn:5443/dcu/admin/base/migraphx:5.2.0-ubuntu22.04-dtk26.04-py3.11 /bin/bash
```

**依赖/环境**：
```bash
source /opt/dtk/env.sh
export PYTHONPATH=/opt/dtk/lib:$PYTHONPATH
下载代码：
git clone https://developer.sourcefind.cn/codes/modelzoo/yolov8_migraphx.git

# 进入python示例目录
cd <path_to_yolov8_migraphx>/Python

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**启动/推理**：
```bash
python YoloV8_infer_migraphx.py --staticInfer
```

**其他命令**：
```bash
访问魔搭社区下载：https://www.modelscope.cn/models/Ultralytics/YOLOv8

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/migraphx:5.2.0-ubuntu22.04-dtk26.04-py3.11
```
