import torch
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create large random 2D tensors (2048 rows x 784 columns)
print("Generating tensor1.bin (2048x784)...")
tensor1 = torch.rand(2048, 784, dtype=torch.float32) * 100

# Save as binary
with open('tensor1.bin', 'wb') as f:
    f.write(tensor1.numpy().tobytes())

print("Generating tensor2.bin (2048x784)...")
tensor2 = (torch.rand(2048, 784, dtype=torch.float32) - 0.5) * 100

# Save as binary
with open('tensor2.bin', 'wb') as f:
    f.write(tensor2.numpy().tobytes())

print("Done! Generated 2 tensors of size 2048x784")
print(f"File size: ~{2048 * 784 * 4 / 1024 / 1024:.2f} MB each")
