# BUILDING MODELS WITH PYTORCH
# # Other Layers and Functions

import torch

# Data Manipulation Layers
print("\n# Data Manipulation Layers - MaxPool")
my_tensor = torch.rand(1, 6, 6)
print(my_tensor)

maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))


print("\n# Data Manipulation Layers - Batch Normalization")
my_tensor = torch.rand(1, 4, 4) * 20 + 5
print(my_tensor)

print(my_tensor.mean())

norm_layer = torch.nn.BatchNorm1d(4)
normed_tensor = norm_layer(my_tensor)
print(normed_tensor)

print(normed_tensor.mean())


print("\n# Data Manipulation Layers - Dropout")
my_tensor = torch.rand(1, 4, 4)

dropout = torch.nn.Dropout(p=0.4)
print(dropout(my_tensor))
print(dropout(my_tensor))


print("\n# Data Manipulation Layers - Activation Functions")
relu = torch.nn.LeakyReLU()
print(relu.forward(torch.tensor(10.0)))
print(relu.forward(torch.tensor(0.0)))
print(relu.forward(torch.tensor(-10.0)))


print("\n# Data Manipulation Layers - Loss Functions")
criterion = torch.nn.CrossEntropyLoss()

# High entropy -> toward to 0.5
# Batch = 2, output class = 3
loss = criterion(torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]), torch.tensor([1, 0]))
print(loss)

# Low entropy -> toward to 1 or 0
loss = criterion(torch.tensor([[0.0, 0.5, 0.0], [0.0, 0.5, 0.0]]), torch.tensor([0, 1]))
print(loss)
