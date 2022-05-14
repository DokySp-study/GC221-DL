# THE FUNDAMENTALS OF AUTOGRAD
# # Autograd in Training

import torch


BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10


class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(1000, 100)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
gt = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

model = TinyModel()


print("\nWeight & Grad. of weight")
print(model.layer2.weight[0][0:10])  # just a small slice
print(model.layer2.weight.grad)


print("\nSet optimizer and calculate loss")
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

prediction = model(some_input)

loss = (gt - prediction).pow(2).sum()
print(loss)


print("\nBack Propagation")
loss.backward()
print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad[0][0:10])


print("\nUpdate optimizer's learnable variables")
optimizer.step()
print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad[0][0:10])


print("\nDo forward and back propagation in 5 times")
print("Before")
print(model.layer2.weight.grad[0][0:10])

for i in range(0, 5):
    prediction = model(some_input)
    loss = (gt - prediction).pow(2).sum()
    loss.backward()

print("After")
print(model.layer2.weight.grad[0][0:10])

print("\nZero Grad")
optimizer.zero_grad()
print(model.layer2.weight.grad[0][0:10])
