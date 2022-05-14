# THE FUNDAMENTALS OF AUTOGRAD
# # Turning Autograd Off and On

import torch
import math

a = torch.ones(2, 3, requires_grad=True)
print(a)

b1 = 2 * a
print(b1)

# Can arithmetic calculation between numbers and tensors what has requires_grad=False option
a.requires_grad = False
b2 = 2 * a
print(b2)


a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = a + b
print(c1)

# Use no_grad option like below
with torch.no_grad():
    c2 = a + b

print(c2)

c3 = a * b
print(c3)


# Can set the function's decorator with no_grad()
def add_tensors1(x, y):
    return x + y


# like below
@torch.no_grad()
def add_tensors2(x, y):
    return x + y


a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = add_tensors1(a, b)
print(c1)

c2 = add_tensors2(a, b)
print(c2)


x = torch.rand(5, requires_grad=True)

# Method 1
x.requires_grad_ = False
y = x

# Method 2
y = x.detach()

print(x)
print(y)


# Autograd and In-place Operations
# inner calculation (2.0 * math.pi) can break calculation of back propagation
# so, the code of below makes error
a = torch.linspace(0.0, 2.0 * math.pi, steps=25, requires_grad=True)
torch.sin_(a)
