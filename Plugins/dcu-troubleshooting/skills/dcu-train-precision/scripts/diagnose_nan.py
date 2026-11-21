#!/usr/bin/env python3
# DCU 训练精度排查：NaN/Inf 快速诊断脚本
# 来源：DCU平台训练精度排查最佳实践指南 主文档 附录（已从 内部文档 导出原文归一化）
# 用途：遍历 dataloader 前几个 batch，检查模型输出 loss/logits 是否含 NaN/Inf。
# 用法：
#   diagnose_model(model, dataloader, device)
import torch


def diagnose_model(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            if hasattr(outputs, 'loss') and torch.isnan(outputs.loss):
                print(f"Batch {batch_idx}: Loss is NaN!")
            if hasattr(outputs, 'logits') and torch.isnan(outputs.logits).any():
                print(f"Batch {batch_idx}: Logits contain NaN!")

            if batch_idx >= 5:
                break
    print("Diagnostic complete.")
