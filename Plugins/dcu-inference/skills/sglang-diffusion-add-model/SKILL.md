---
name: sglang-diffusion-add-model
description: "把 Diffusers 或参考流水线的图像/视频扩散模型接入 SGLang 原生 pipeline / stage / model / config 结构（sglang diffusion add model）。何时用：扩散模型移植与服务集成；不是自回归 LLM。"
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# SGLang Diffusion Add Model

## Overview

Guide for integrating new diffusion models into SGLang's native serving pipeline. The SGLang diffusion pipeline uses a decomposition into pipeline/stage/model/config layers that differs from Diffusers' pipeline abstraction.

First verify that the current checkout contains the native diffusion backend and identify the target GPU/toolchain. CUDA, ROCm, DCU/HCU, and vendor-specific attention or quantization backends are not interchangeable. Preserve reference-pipeline seeds, scheduler parameters, prompts, dimensions, step counts, and output quality checks, and prove that native execution did not silently fall back to Diffusers.

## When To Use

- Adding support for a new diffusion model family (e.g., a new DiT architecture)
- Porting a model from HuggingFace Diffusers to SGLang native serving
- Adapting a reference pipeline implementation to SGLang's serving structure
- Setting up a new model for benchmarking with `sglang-diffusion-benchmark-profile`

## Pipeline Architecture

SGLang diffusion models are structured as:

```
Pipeline (top-level orchestration)
  ├── Stage 1: Text Encoding (CLIP, T5, etc.)
  ├── Stage 2: Denoising Loop
  │   ├── DiT/UNet Model (forward pass)
  │   ├── Scheduler (noise schedule)
  │   └── CFG Handling (classifier-free guidance)
  └── Stage 3: VAE Decode (latent → image/video)
```

## Integration Steps

### 1. Analyze the Reference Model

From the Diffusers/reference pipeline, extract:
- Model architecture (DiT, UNet, transformer-based)
- Text encoder(s) used (CLIP, T5, FLAN-T5)
- VAE configuration (spatial compression, channels)
- Scheduler type (DDPM, DDIM, Euler, Flow Matching)
- Special features (CFG, spectral diffusion, progressive distillation)

### 2. Create Config

Create model config mapping to SGLang's structure:

```python
# <model>_config.py
class <Model>Config:
    # Text encoder config
    text_encoder = {
        "type": "clip",  # or "t5", "dual"
        "model_id": "openai/clip-vit-large-patch14",
        "hidden_size": 768,
    }

    # DiT model config
    dit_config = {
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "patch_size": 2,
        "in_channels": 16,   # latent channels
        "out_channels": 16,
    }

    # VAE config
    vae_config = {
        "scale_factor": 8,
        "latent_channels": 16,
    }
```

### 3. Implement Pipeline Stages

#### Stage 1: Text Encoding
```python
class <Model>TextEncoderStage:
    def encode(self, prompt: str) -> Tensor:
        # Encode prompt → text embeddings
        pass
```

#### Stage 2: Denoising Model
```python
class <Model>DiT:
    def forward(self, latent: Tensor, timestep: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        # DiT forward pass
        pass
```

#### Stage 3: VAE Decode
```python
class <Model>VAEDecodeStage:
    def decode(self, latent: Tensor) -> Tensor:
        # Decode latent → image/video
        pass
```

### 4. Register in SGLang

Add model registration to SGLang's model registry:

```python
# In sglang/multimodal_gen/runtime/models/
@register_model("my-diffusion-model")
class MyDiffusionModel(DiffusionPipeline):
    config_class = MyDiffusionConfig
    # ...
```

### 5. Verify Backend Integration

After integration, verify:
```bash
# Check model loads correctly
python -m sglang.launch_server --model my-diffusion-model --tp 1

# Verify no fallback
grep -i "diffusers\|fallback" server.log  # Should be empty

# Quick generation test
curl http://localhost:30000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test image", "n": 1, "size": "512x512"}'
```

## Common Patterns

### DiT-Based Models (FLUX, Wan, Qwen-Image)

- Use transformer backbone instead of UNet
- Patchify input latents (patch_size=2 common)
- Text conditioning via cross-attention or adaLN
- Usually use flow-matching scheduler

### UNet-Based Models (SDXL, older models)

- Traditional UNet backbone
- Cross-attention for text conditioning
- DDPM/DDIM schedulers
- May need VAE tiling for high resolutions

### Video Models (LTX, Cosmos)

- 3D convolutions or temporal attention
- Causal padding for autoregressive generation
- Higher memory requirements
- May need frame-by-frame scheduling

## Output Contract

After integration, confirm:
- [ ] Model loads without fallback to diffusers
- [ ] Native SGLang diffusion backend active
- [ ] Text encoding produces correct embeddings
- [ ] Denoising produces valid latents (no NaN/Inf)
- [ ] VAE decode produces valid images/videos
- [ ] Performance is comparable or better than Diffusers baseline
- [ ] Model config documented for `sglang-diffusion-benchmark-profile`

## Integration

- **After adding model**: benchmark with `sglang-diffusion-benchmark-profile`
- **For performance**: tune with `sglang-diffusion-performance`
- **Reference**: SGLang's existing diffusion models (FLUX, Wan, LTX) as templates
