from sched import scheduler
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim


# GPU 사용
batch_size = 16
learning_rate = 0.01
epochs = 10
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("[Device]: ", device)


# Load data
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize([112, 112]),
        ]
    ),
)

train_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    **kwargs,
)


test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.Compose(
        # Image to tensor type & Data augumentation part
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize([112, 112]),
        ]
    ),
)


test_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    **kwargs,
)


print("[Trainset]: ", train_dataset)
print("[Testset]: ", test_dataset)


# 모델 콜
model = models.resnet18(pretrained=True)

model.conv1 = nn.Conv2d(
    1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
)

# Weight initialization
linear = nn.Linear(in_features=512, out_features=10, bias=True)

# - init bias
linear.bias.data.fill_(0.01)

# - init weight
torch.nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
# torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")

model.fc = linear


# Transfer Learning
print("Before (parameter setting): ", len(next(model.parameters())), " layers")

params_to_update = []
for name, param in model.named_parameters():
    if "fc" in name:
        # print(name)
        # print(param.requires_grad)
        param.requires_grad = True
        params_to_update.append(param)
    elif "layer4" in name:
        # print(name)
        # print(param.requires_grad)
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False

print("After (parameter setting): ", len(params_to_update), " layers")


model.to(device=device)
# GPU Available check
print("CUDA: ", next(model.parameters()).is_cuda)
# Print Model
print("Tuned Model structure: \n", model)


# 옵티마이져
optimizer = optim.Adam(params_to_update, lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.1)
loss_function = nn.CrossEntropyLoss()


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: data.DataLoader,
    optimizer: optim,
    loss_function: nn.CrossEntropyLoss,
    epoch: int,
):
    # 트레이닝 모드
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)

        loss.backward()  # dloss/dx
        optimizer.step()  # parameter update

        if batch_idx % 1000 == 0:
            print(
                "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(
    model: nn.Module,
    device: torch.device,
    test_loader: data.DataLoader,
    loss_function: nn.CrossEntropyLoss,
):
    # Inference 모드
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():

        # process = 0

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += loss_function(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

            # process += 1
            # if (process / len(test_loader.dataset) * 100) % 10 == 0:
            #     print(
            #         "Testing: {}/{} ({:.0f}%)".format(
            #             process,
            #             len(test_loader.dataset),
            #             process / len(test_loader.dataset) * 100,
            #         )
            #     )

        test_loss /= len(test_loader.dataset)
        print(
            "\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )


# Training
for epoch in range(epochs):

    print("Learning Rate: ", scheduler.get_lr())
    print("Last Learning Rate: ", scheduler.get_last_lr())

    train(
        model=model,
        device=device,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch=epoch,
    )

    test(
        model=model,
        device=device,
        test_loader=test_loader,
        loss_function=loss_function,
    )

    scheduler.step()
