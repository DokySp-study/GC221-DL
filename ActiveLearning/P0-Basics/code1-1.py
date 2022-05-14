# PyTorch Tensors

import torch

z = torch.zeros(5, 3)
print(z)
print(z.dtype)

# Result:
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])
# torch.float32


i = torch.ones((5, 3), dtype=torch.int16)
print(i)

# Result:
# tensor([[1, 1, 1],
#         [1, 1, 1],
#         [1, 1, 1],
#         [1, 1, 1],
#         [1, 1, 1]], dtype=torch.int16)


# It shows same result with same seed value.
torch.manual_seed(1729)
r1 = torch.rand(2, 2)
print("A random tensor:")
print(r1)

r2 = torch.rand(2, 2)
print("\nA different random tensor:")
print(r2)

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print("\nShould match r1:")
print(r3)

# Result:
# A random tensor:
# tensor([[0.3126, 0.3791],
#         [0.3087, 0.0736]])

# A different random tensor:
# tensor([[0.4216, 0.0691],
#         [0.2332, 0.4047]])

# Should match r1:
# tensor([[0.3126, 0.3791],
#         [0.3087, 0.0736]])


# Arithmetic operation with tensor
ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2  # every element is multiplied by 2
print(twos)

threes = ones + twos  # addition allowed because shapes are similar
print(threes)  # tensors are added element-wise
print(threes.shape)  # this has the same dimensions as input tensors

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)

r = (torch.rand(2, 2) - 0.5) * 2  # values between -1 and 1
print("\nA random matrix, r:")
print(r)

# Common mathematical operations are supported:
print("\nAbsolute value of r:")
print(torch.abs(r))

# ...as are trigonometric functions:
print("\nInverse sine of r:")
print(torch.asin(r))

# ...and linear algebra operations like determinant and singular value decomposition
print("\nDeterminant of r:")
print(torch.det(r))
print("\nSingular value decomposition of r:")
print(torch.svd(r))

# ...and statistical and aggregate operations:
print("\nAverage and standard deviation of r:")
print(torch.std_mean(r))
print("\nMaximum value of r:")
print(torch.max(r))
print("\nMinimum value of r:")
print(torch.min(r))

# Result:
# Should match r1:
# tensor([[0.3126, 0.3791],
#         [0.3087, 0.0736]])
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])
# tensor([[2., 2., 2.],
#         [2., 2., 2.]])
# tensor([[3., 3., 3.],
#         [3., 3., 3.]])
# torch.Size([2, 3])

# A random matrix, r:
# tensor([[ 0.9956, -0.2232],
#         [ 0.3858, -0.6593]])

# Absolute value of r:
# tensor([[0.9956, 0.2232],
#         [0.3858, 0.6593]])

# Inverse sine of r:
# tensor([[ 1.4775, -0.2251],
#         [ 0.3961, -0.7199]])

# Determinant of r:
# tensor(-0.5703)

# Singular value decomposition of r:
# torch.return_types.svd(
# U=tensor([[-0.8353, -0.5497],
#         [-0.5497,  0.8353]]),
# S=tensor([1.1793, 0.4836]),
# V=tensor([[-0.8851, -0.4654],
#         [ 0.4654, -0.8851]]))

# Average and standard deviation of r:
# (tensor(0.7217), tensor(0.1247))

# Maximum value of r:
# tensor(0.9956)

# Minimum value of r:
# tensor(-0.6593)
