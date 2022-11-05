import torch

v1 = torch.Tensor([1, 2, 3]).view(1, -1)
v2 = torch.Tensor([1, 2, 3]).view(1, -1)

sim1 = torch.cosine_similarity(v1, v2)
print(sim1)
