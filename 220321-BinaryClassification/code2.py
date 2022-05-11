
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

x_data = [
  [80, 220, 6300],
  [75, 167, 4500],
  [86, 210, 7500],
  [110, 330, 9000],
  [95, 280, 8700],
  [67, 190, 6800],
  [79, 210, 5000],
  [98, 250, 7200],
]
y_data = [2,3,1,0,0,3,2,1]


x_train = torch.FloatTensor(x_data)
y_train = torch.LongTensor(y_data)

W = torch.zeros((3,4), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr=0.00001)

for e in range(10000):
  optimizer.zero_grad()

  z = torch.sigmoid(x_train.matmul(W) + b)
  cost = F.cross_entropy(z, y_train)
  cost.backward()
  optimizer.step()

  print("epoch: %d / cost: %d" %(e, cost))