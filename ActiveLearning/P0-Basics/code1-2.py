# PyTorch Models
# LeNet Example

import torch  # for all things PyTorch
import torch.nn as nn  # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function

# LeNet-5 Diagram
# https://pytorch.org/tutorials/_images/mnist.png
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 3x3 square convolution

        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = LeNet()
print(net)
# This shows how network was composed in LeNet()
# LeNet(
#   (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
#   (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
#   (fc1): Linear(in_features=576, out_features=120, bias=True)
#   (fc2): Linear(in_features=120, out_features=84, bias=True)
#   (fc3): Linear(in_features=84, out_features=10, bias=True)
# )

# stand-in for a 32x32 black & white image
# batch 1, channel 1, 32x32 image
input = torch.rand(1, 1, 32, 32)
print("\nImage batch shape:")
print(input.shape)

# The function `forward()` will be automatically executed
output = net(input)
print("\nRaw output:")
print(output)
print(output.shape)
# Raw output:
# tensor([[-0.0565,  0.0987, -0.1055, -0.0022,  0.0667,  0.0895, -0.0126,  0.0270,
#           0.0272,  0.0871]], grad_fn=<AddmmBackward0>)
# torch.Size([1, 10])
