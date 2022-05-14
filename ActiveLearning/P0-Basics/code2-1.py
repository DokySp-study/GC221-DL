import torch
import math


# Creating Tensors
print("\n# Creating Tensors")
x = torch.empty(3, 4)
print(type(x))
print(x)


zeros = torch.zeros(2, 3)
print("zeros", zeros)

ones = torch.ones(2, 3)
print("ones", ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print("random", random)


# Random Tensors and Seeding
torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print("random1", random1)

random2 = torch.rand(2, 3)
print("random2", random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print("random3", random3)

random4 = torch.rand(2, 3)
print("random4", random4)


# Tensor Shapes
print("\n# Tensor Shapes")
x = torch.empty(2, 2, 3)
print(x.shape)
print(x, "\n")

# no init.
empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x, "\n")

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x, "\n")

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x, "\n")

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x, "\n")


some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print("some_constants", some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print("some_integers", some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print("more_integers", more_integers)


# Tensor Data Types
print("\n# Tensor Data Types")
a = torch.ones((2, 3), dtype=torch.int16)
print(a, "\n")

b = torch.rand((2, 3), dtype=torch.float64) * 20.0
print(b, "\n")

# type casting
c = b.to(torch.int32)
print(c, "\n")
# type of tensor
# torch.bool, torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.half, torch.float, torch.double, torch.bfloat


# Math & Logic with PyTorch Tensors
print("\n# Math & Logic with PyTorch Tensors")
ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
# power
fours = twos**2
sqrt2s = twos**0.5

print("ones", ones)
print("twos", twos)
print("threes", threes)
print("fours", fours)
print("sqrt2s", sqrt2s)

powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print("powers2", powers2)

fives = ones + fours
print("fives", fives)

dozens = threes * fours
print("dozens", fives)


# This will make run-time error
# a = torch.rand(2, 3)
# b = torch.rand(3, 2)
# print(a * b)
# -> a.shape != b.shape

rand = torch.rand(2, 4)
# multiple with same shape
# doubled = rand * (torch.ones(1, 4) * 2)
doubled = rand * (torch.ones(2, 4) * 2)  # no error occured!
# -> The exception to the same-shapes rule is tensor broadcasting.
# -> The common example is multiplying a tensor of learning weights by a batch of input tensors.

# The rules of broadcasting:
#  - Each tensor must have at least one dimension - no empty tensors.
#  - Comparing the dimension sizes of the two tensors, going from last to first:
#    - Each dimension must be equal, or (ex> 2x4 * 2x4)
#    - One of the dimensions must be of size 1, or (ex> 2x4 * 1x4)
#    - The dimension does not exist in one of the tensors (ex> 2x4 * 3(value))

print("rand", rand)
print("doubled", doubled)


a = torch.ones(4, 3, 2)

b = a * torch.rand(3, 2)  # 3rd & 2nd dims identical to a, dim 1 absent
print("b", b)

c = a * torch.rand(3, 1)  # 3rd dim = 1, 2nd dim identical to a
print("c", c)

d = a * torch.rand(1, 2)  # 3rd dim identical to a, 2nd dim = 1
print("d", d)


a = torch.ones(4, 3, 2)

# dimensions must match last-to-first
# b = a * torch.rand(4, 3)
# -> a: 4-3-2 * 2-...

# both 3rd & 2nd dims different
# c = a * torch.rand(2, 3)
# -> a: 3-D / b: 2-D

# can't broadcast with an empty tensor
# d = a * torch.rand((0,))


# More Math with Tensors
print("\n# More Math with Tensors")
# common functions
a = torch.rand(2, 4) * 2 - 1
print("Common functions:")
print("abs", torch.abs(a))
print("ceil", torch.ceil(a))
print("floor", torch.floor(a))
print("clamp", torch.clamp(a, -0.5, 0.5))
# Clamp
# y = min( max(x, min_value), max_value )

# trigonometric functions and their inverses
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print("\nSine and arcsine:")
print(angles)
print("sin", sines)
print("asin", inverses)


# bitwise operations
print("\n# Bitwise XOR:")
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))


# comparisons:
print("\n# Broadcasted, element - wise equality comparison:")
d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
e = torch.ones(1, 2)  # many comparison ops support broadcasting!
print(torch.eq(d, e))  # returns a tensor of type bool


# reductions:
print("\n# Reduction ops:")
print(torch.max(d))  # returns a single-element tensor
print(torch.max(d).item())  # extracts the value from the returned tensor
print(torch.mean(d))  # average
print(torch.std(d))  # standard deviation
print(torch.prod(d))  # product of all numbers
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2])))  # filter unique elements

# vector and linear algebra operations
v1 = torch.tensor([1.0, 0.0, 0.0])  # x unit vector
v2 = torch.tensor([0.0, 1.0, 0.0])  # y unit vector
m1 = torch.rand(2, 2)  # random matrix
m2 = torch.tensor([[3.0, 0.0], [0.0, 3.0]])  # three times identity matrix


print("\n# Vectors & Matrices:")
print(m1)
print(m2)
print(torch.cross(v2, v1))  # negative of z unit vector (v1 x v2 == -v2 x v1)
print(m1)
m3 = torch.matmul(m1, m2)  # same as (m1 * 3)
print(m3)  # 3 times m1
print(torch.svd(m3))  # singular value decomposition


# Altering Tensors in Place
print("\n# Altering Tensors in Place (Inplace)")
a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print("a:")
print(a)
print(torch.sin(a))  # this operation creates a new tensor in memory
print(a)  # a has not changed

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print("\nb:")
print(b)
# underscore == inplace option in Pandas
print(torch.sin_(b))  # note the underscore
print(b)  # b has changed


a = torch.ones(2, 2)
b = torch.rand(2, 2)

print("Before:")
print(a)
print(b)
print("\nAfter adding:")
print(a.add_(b))
print(a)
print(b)
print("\nAfter multiplying")
print(b.mul_(b))
print(b)


a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)

print(c)
d = torch.matmul(a, b, out=c)
print(c)  # contents of c have changed

assert c is d  # test c & d are same object, not just containing equal values
assert id(c), old_id  # make sure that our new c is the same object as the old one

torch.rand(2, 2, out=c)  # works for creation too!
print(c)  # c has changed again
assert id(c), old_id  # still the same object!


# Copying Tensors
print("\n# Copying Tensors")

a = torch.ones(2, 2)
b = a

# Shallow copy
a[0][1] = 561  # we change a...
print(b)  # ...and b is also altered

# Deep copy
a = torch.ones(2, 2)
b = a.clone()

assert b is not a  # different objects in memory...
print(torch.eq(a, b))  # ...but still with the same contents!

a[0][1] = 561  # a changes...
print(b)  # ...but b is still all ones


a = torch.rand(2, 2, requires_grad=True)  # turn on autograd
print(a)

b = a.clone()
print(b)

c = a.detach().clone()
print(c)
print(a)


# Moving to GPU
print("\n# Moving to GPU")
if torch.cuda.is_available():
    print("We have a GPU!")
else:
    print("Sorry, CPU only.")


if torch.cuda.is_available():
    gpu_rand = torch.rand(2, 2, device="cuda")
    print(gpu_rand)
else:
    print("Sorry, CPU only.")


if torch.cuda.is_available():
    my_device = torch.device("cuda")
else:
    my_device = torch.device("cpu")
print("Device: {}".format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)


y = torch.rand(2, 2)
y = y.to(my_device)

# Both tensor must be in same memory space (Memory vs. GPU Memory)
# x = torch.rand(2, 2)
# y = torch.rand(2, 2, device="gpu")
# z = x + y  # exception will be thrown


# Manipulating Tensor Shapes
print("\n# Manipulating Tensor Shapes - Changing the Number of Dimensions")

a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)
# `unsqueeze(0)` adds it as a new zeroth dimension

print("a.shape", a.shape)
print("b.shape", b.shape)

c = torch.rand(1, 1, 1, 1, 1)
print(c)


a = torch.rand(1, 20)
print("a.shape", a.shape)
print("a", a)

b = a.squeeze(0)  # (1, 20) -> (20)
print("b.shape", b.shape)
print("b", b)

a = torch.rand(20, 1)
print("a.shape", a.shape)
print("a", a)

b = a.squeeze(0)  # (20, 1) -> (20, 1)
print("b.shape", b.shape)
print("b", b)

c = torch.rand(2, 2)
print("c.shape", c.shape)

d = c.squeeze(0)  # squeeze will work 1xN tensor only
print("d.shape", d.shape)


a = torch.ones(4, 3, 2)
c = a * torch.rand(3, 1)  # 3rd dim = 1, 2nd dim identical to a
print("c", c)


a = torch.ones(4, 3, 2)
b = torch.rand(3)  # trying to multiply a * b will give a runtime error
c = b.unsqueeze(1)  # change to a 2-dimensional tensor, adding new dim at the end
print(c.shape)
print(a * c)  # broadcasting works again!


batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)


output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# can also call it as a method on the torch module:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)


# NumPy Bridge
print("\n# NumPy Bridge")

import numpy as np

numpy_array = np.ones((2, 3))
print(numpy_array)

pytorch_tensor = torch.from_numpy(numpy_array)
print(pytorch_tensor)

pytorch_rand = torch.rand(2, 3)
print(pytorch_rand)

numpy_rand = pytorch_rand.numpy()
print(numpy_rand)

numpy_array[1, 1] = 23
print(pytorch_tensor)

pytorch_rand[1, 1] = 17
print(numpy_rand)
