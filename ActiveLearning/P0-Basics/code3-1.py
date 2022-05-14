# THE FUNDAMENTALS OF AUTOGRAD
# # A Simple Example

import torch
import matplotlib.pyplot as plt
import math


a = torch.linspace(0.0, 2.0 * math.pi, steps=25, requires_grad=True)
print(a)
print(a.shape)

b = torch.sin(a)
# plt.plot(a, b) # this makes error
# need to detach before use values
plt.plot(a.detach(), b.detach())
print(b)
print(b.shape)
plt.show()  # TODO Need to add this line

# These have backward() func.
c = 2 * b
print(c)

d = c + 1
print(d)


out = d.sum()
print(out)


print("\nd:")
# c + 1
print(d.grad_fn)
# 2 * b
print(d.grad_fn.next_functions)
# sin(a)
print(d.grad_fn.next_functions[0][0].next_functions)
# a
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
# None -> Accumulated computations
print(
    d.grad_fn.next_functions[0][0]
    .next_functions[0][0]
    .next_functions[0][0]
    .next_functions
)
print("\nc:")
print(c.grad_fn)
print("\nb:")
print(b.grad_fn)
print("\na:")
print(a.grad_fn)
