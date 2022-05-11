
import numpy as np
import torch
import torch.optim as optim
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.ticker import FormatStrFormatter



# Plot in 2D graph
def draw_2d_plot(x_axis, y_axis, xlabel, ylabel):

  fig = plt.figure()
  ax = fig.add_subplot()

  # Make data.
  X = np.arange(0, len(x_axis), 1)
  Y = np.array(y_axis)

  # Plot the surface.
  plot = ax.plot(X, Y, linewidth=1, antialiased=False)

  # Customize axis.
  ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
  ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)

  plt.show()


# Plot in 3D space
def draw_3d_plot(x_axis, y_axis, z_axis, xlabel, ylabel, zlabel):

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  # Make data.
  X = np.arange(0, len(x_axis), 1)
  Y = np.arange(0, len(y_axis), 1)
  X, Y = np.meshgrid(X, Y)
  Z = np.array(z_axis)

  # Plot the surface.
  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.1, antialiased=False)

  # Customize axis.
  ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
  ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=1, aspect=5)

  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_zlabel(zlabel)

  plt.show()



# Default learning rate and epochs values
# lr_list = [0.000005, 0.000004, 0.000003, 0.000002, 0.000001, 0.0000005, 0.0000001, 0.00000005, 0.00000001, 0.000000005, 0.000000001, 0.0000000005, 0.0000000001]
# epoch_list = [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]

# To find cliff
lr_list = [0.0000016020, 0.0000016015, 0.0000016010, 0.0000016005, 0.00000160, 0.0000015995, 0.0000015990,] 
epoch_list = [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]

# To plot in 2-D graph
# lr_list = [0.000005, 0.000004, 0.000003, 0.000002, 0.000001, 0.0000005, 0.0000001, 0.00000005, 0.00000001, 0.000000005, 0.000000001, 0.0000000005, 0.0000000001] 
# epoch_list = [5000, 10000]


# Save costs and minimal values
cost_list = []
best_cost = 10000
best_lr = -1
best_epoch = -1


# Dataset
x_train = torch.FloatTensor([[3.8, 700, 80, 50], [3.2, 650, 90, 30], [3.7, 820, 70, 40], [4.2, 830, 50, 70], [2.6, 550, 90, 60], [3.4, 910, 30, 40], [4.1, 990, 70, 20], [3.3, 870, 60, 60], [3.9, 650, 80, 50]])
y_train = torch.FloatTensor([[85], [80], [78], [87], [85], [70], [81], [88], [84]])


# For presenting progress of training
total_iter_count = len(lr_list) * len(epoch_list)
iter_count = 1


# Training code
for curr_lr in lr_list:

  cost_tmp = []

  # Initialize variables
  # Weight
  W = torch.zeros((4,1), requires_grad=True)
  # Intercept
  b = torch.zeros(1, requires_grad=True)
  optimizer = optim.SGD([W, b], lr=curr_lr)

  
  for curr_epoch in epoch_list:
    
    print(f"{iter_count}/{total_iter_count} in progress. epoch: {curr_epoch} / lr: {curr_lr}", end=" / ")
    
    for epoch in range(curr_epoch):

      # H(x)
      hypothesis = x_train.matmul(W) + b

      # MSE :: Σ((y'-y)^2) / row#
      cost = torch.mean((hypothesis - y_train) ** 2)

      optimizer.zero_grad()
      cost.backward()
      optimizer.step()
    

    # Set infinite value to 700 to plot data
    cost_val = 700
    if not torch.isnan(cost):
      cost_val = cost.squeeze().detach().numpy()

    # Change huge value to 700 to plot data
    if cost_val > 700:
      cost_val = 700

    # Save the cost for plot data
    cost_tmp.append(cost_val)
    
    # Save the minimize error value
    if cost_val < best_cost:
      best_cost = cost_val
      best_lr = curr_lr
      best_epoch = curr_epoch
    
    print(f"Best(Min) error: {best_cost}")
    iter_count += 1

  cost_list.append(cost_tmp)


# After find the best learning rate and number of epochs value, 
# adopt this to the Mulitple LR Model
W = torch.zeros((4,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W, b], lr=best_lr)


for epoch in range(best_epoch):
  # H(x)
  hypothesis = x_train.matmul(W) + b
  # MSE :: Σ((y'-y)^2) / row#
  cost = torch.mean((hypothesis - y_train) ** 2)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()


# Predict A's score
test_data = torch.FloatTensor([3.3, 700, 77, 84])
predict = test_data.matmul(W) + b

# Tensor to Numpy  배열풀기    grad풀기  numpy화
pred_val = predict.squeeze().detach().numpy()

# Print the results
print("")
print('Total score is estimated a %d'%(pred_val))
print("")

print('The value of Trained W:')
print(pd.DataFrame(W.squeeze().detach().numpy().reshape(1,-1), columns=["GPA", "TOEIC", "Award", "Etc."], index=["Trained W"]))

print("")
print('The value Trained b: %f' %(b.squeeze().detach().numpy()))

print("")
print('Best epoch: %d (idx: %d)' %(best_epoch, epoch_list.index(best_epoch)))
print('Best lr: %.8f (idx: %d)' %(best_lr, lr_list.index(best_lr)))
print('Best cost (minimize is the best): %f' %(best_cost))

# Plot trained lr and # of epochs
# draw_2d_plot(lr_list, cost_list, "Learning Rate", "MSE")
draw_3d_plot(epoch_list, lr_list, cost_list, 'Number of epoch', 'Learning Rate', 'MSE')


# Visualization predicted and actual value
test_data = x_train
predict = test_data.matmul(W) + b

y_pred = predict.squeeze().detach().numpy()
y_actual = y_train.squeeze().detach().numpy()

y = []

for i in [5,2,1,6,8,0,4,3,7]:
  y.append([y_actual[i], y_pred[i]])

index = [0,1,2,3,4,5,6,7,8]

draw_2d_plot(index, y, "index", "Total")
