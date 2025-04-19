#!/usr/bin/env python3

import torch
import time

# Pick device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("Device name:", torch.cuda.get_device_name(0))

# Create two large tensors on the selected device
size = 2048  # Reduced size so it's not too heavy on CPU
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Warm up (especially important for GPU)
_ = torch.matmul(a, b)

# Time the operation
start = time.time()
_ = torch.matmul(a, b)
if device.type == "cuda":
    torch.cuda.synchronize()  # Ensure GPU ops finish
end = time.time()

print(f"Time for {size}x{size} matmul on {device}: {end - start:.4f} seconds")
