import torch

x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
y = torch.Tensor([1, 2, 3])

print(x)
print(x.shape)# 2, 3

print(y)
print(y.shape) #3

print(torch.mul(x, y))
"""
batch_x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])   #128, 512
ws = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])  #256, 512
#output must be 128, 2, 4 -> 3, 4, 2 -> 3, 8

print(torch.mul(batch_x,ws))"""