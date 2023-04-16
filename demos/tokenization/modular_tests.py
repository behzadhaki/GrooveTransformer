import os
import torch
import numpy

os.chdir("../../")

tensor = torch.randint(2, (16, 1000, 19))


sliced = tensor[:, :, :1]

print(f"Sliced Shape: {sliced.shape}")
print(sliced[:1, :5, :])

n_token_types = 2
d_model = 19

embedding = torch.nn.Embedding(n_token_types, d_model,dtype=torch.float32)

x = embedding(sliced)
print(x.shape)