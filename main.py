import torch
import random
a=torch.Tensor([[1,2],[100,5],[0,0],[1,1]])
b=torch.Tensor([[2,3], [8,9], [5,5]])
print("argmax", torch.argmax(a, dim=-1))
print(a.shape, b.shape)
print(torch.mm(a, b.T))
print(torch.matmul(a,b.T))

print(random.randint(0, 11))