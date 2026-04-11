# 快速检查 CUDA
import torch
print(f"PyTorch CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"显卡数量: {torch.cuda.device_count()}")
    print(f"当前显卡: {torch.cuda.get_device_name(0)}")