
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import torch.nn.functional as F

x_data = [[80,220],[75,167],[86,210],[110,330],[95,280],[67,190],[79,210],[98,250]] 
y_data = [[1],[0],[1],[1],[1],[0],[0],[1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)

W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr=0.000001)

for e in range(100):
  optimizer.zero_grad()

  # 풀어서 씀
  # hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
  # loss = -(y_train * torch.log(hypothesis) + ( 1 - y_train) * torch.log(1-hypothesis))
  # cost = loss.mean()

  # Internal Function 활용
  hypothesis = torch.sigmoid(x_train.matmul(W) + b)
  cost = F.binary_cross_entropy(hypothesis, y_train)

  cost.backward()
  optimizer.step()

  print("epoch %d %d" %(e, cost))