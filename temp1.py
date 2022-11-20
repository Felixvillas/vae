'''
# for...else...
# 只要for循环中没有执行break语句(可以理解为for循环正常结束), 则else正常执行
flag = 10
for i in range(10):
    a = i
    if (a < flag) and (a ** 2 > flag):
        print(f"if a: {a}")
    a *= 2
else:
    a *= 1
    print(f"else a: {a}")
'''

import torch
import torch.nn as nn
import numpy as np
def check(x, device, requires_grad=True):
    if isinstance(x, torch.Tensor):
        return x.to(**device).requires_grad_(requires_grad)
    elif isinstance(x, np.ndarray):
        return torch.as_tensor(x).to(**device).requires_grad_(requires_grad)
    else:
        raise NotImplementedError


device = dict(
    dtype=torch.float32, 
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
)
x = torch.as_tensor([[1, 2, 3], [4, 5, 6]])
x = check(x, device)
w = torch.as_tensor([[1, 2], [3, 4], [5, 6]])
w = check(w, device)

y = torch.matmul(x, w) ** 3 + torch.sin(torch.matmul(x, w))

dy = 3 * x ** 2 + torch.cos(x)
d2y = 6 * x - torch.sin(x)

dydx = torch.autograd.grad(y, x, grad_outputs=check(torch.ones(x.shape), device, False), create_graph=True, retain_graph=True)
print(dydx)
print(dy)
d2ydx2 = torch.autograd.grad(dydx, x, grad_outputs=check(torch.ones(x.shape), device, False), create_graph=True, retain_graph=True)
print(d2ydx2)
print(d2y)

