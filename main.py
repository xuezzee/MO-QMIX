import torch
import random
import numpy as np
a=torch.Tensor([[[1,2],[100,5],[9,0],[8,1]]])
# print(a)
# print(torch.Tensor([[[0],[3],[0],[1]]]))
b=torch.Tensor([[2,3], [8,9], [5,5]])
c = np.array([[[2],[3]], [[8],[9]], [[5],[5]]])
# print(torch.gather(a, -2, torch.Tensor([[[0, 1],[0, 2],[0, 1],[1, 2]]]).type(torch.int64)))
# print(torch.gather(a, -2, torch.Tensor([[[0],[1],[0],[1]]]).type(torch.int64)))
# print(torch.gather(a, -2, torch.Tensor([[[1],[2],[0],[1]]]).type(torch.int64)))
# print(torch.gather(a, -2, torch.Tensor([[[0],[3],[0],[1]]]).type(torch.int64)))
# print("expand:", b.repeat((*b.shape, 4)))
a = torch.Tensor([[1,2], [2,3], [3,4]])
b = torch.Tensor([[9,8], [8,7], [7,6]])
print(torch.norm(a, p=2, dim=-1))
print(torch.norm(a, p=2, dim=-1).sum())
print([1,2,3] - 1)