import torch
import numpy as np

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
print("Supported arches:", torch.cuda.get_arch_list())

# 测试 NumPy 互操作
arr = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(arr)
back = tensor.numpy()
print("NumPy <-> PyTorch OK")

# 简单 GPU 计算
x = torch.randn(1000, 1000).cuda()
y = x @ x.T
print("GPU matrix multiplication OK, mean:", y.mean().item())

print("Environment is ready!")