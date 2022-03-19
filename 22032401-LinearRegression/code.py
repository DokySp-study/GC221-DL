import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


x_train = torch.FloatTensor([[3.8, 700, 80, 50], [3.2, 650, 90, 30], [3.7, 820, 70, 40], [4.2, 830, 50, 70], [2.6, 550, 90, 60], [3.4, 910, 30, 40], [4.1, 990, 70, 20], [3.3, 870, 60, 60], [3.9, 650, 80, 50]])
y_train = torch.FloatTensor([[85], [80], [78], [87], [85], [70], [81], [88], [84]])

W = torch.zeros(4, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.00000005)


num_epochs = 200000

for epoch in range(num_epochs):
  hypothesis = x_train.matmul(W) + b
  cost = torch.mean((hypothesis - y_train) ** 2)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 2000 == 0:
    print("Epoch(%d), Cost(%f)"%(epoch, cost))

print("Epoch(%d), Cost(%f)"%(epoch, cost))
print(W)
print(b)
# predict = W * 71 + b
# print("predict: %d" %(predict))

