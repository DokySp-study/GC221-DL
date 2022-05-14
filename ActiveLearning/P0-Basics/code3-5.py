# THE FUNDAMENTALS OF AUTOGRAD
# # Advanced Topic: More Autograd Detail and the High-Level API

import torch

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print("y", y)
# y.backward()
# -> Error: grad can be implicitly created only for scalar outputs
# * For a multi-dimensional output, autograd expects us to provide
#   gradients for those three outputs that it can multiply into the Jacobian:

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)  # stand-in for gradients
y.backward(v)

print("x.grad", x.grad)
print("y", y)


# The High-Level API


def exp_adder(x, y):
    return 2 * x.exp() + 3 * y


print("\nThe High-Level API")
inputs = (torch.rand(1), torch.rand(1))  # arguments for the function
print(inputs)
torch.autograd.functional.jacobian(exp_adder, inputs)

inputs = (torch.rand(3), torch.rand(3))  # arguments for the function
print(inputs)
torch.autograd.functional.jacobian(exp_adder, inputs)


# Make function to directly compute the vector-Jacobian product.
def do_some_doubling(x):
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    return y


inputs = torch.randn(3)
my_gradients = torch.tensor([0.1, 1.0, 0.0001])
torch.autograd.functional.vjp(do_some_doubling, inputs, v=my_gradients)
