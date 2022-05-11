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
learning_rate = 0.001
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
        # Image to tensor type & Data augumentation part
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # Input Size 키우기
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
    # 이미 다운로드 했으니깐 필요 없음
    transform=transforms.Compose(
        # Image to tensor type & Data augumentation part
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # Input Size 키우기
            transforms.Resize([112, 112]),
        ]
    ),
)


test_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    **kwargs,  # 별 두개 -> 가변인자 / # 별 하나는 키만 나온다
)


print("[Trainset]: ", train_dataset)
print("[Testset]: ", test_dataset)


# 모델 콜
model = models.resnet18(pretrained=True)
# RuntimeError: Given groups=1, weight of size [64, 3, 7, 7], expected input[16, 1, 28, 28] to have 3 channels, but got 1 channels instead
#  - ResNet -> batch 64, 7x7 RGB 입력
#  - MNIST -> batch 16, 28x28 Grey Scale


# 모델 수정

# Input data SIZE mismatching problem
#  1. 모델구조 변경 -> 복잡
#  2. input 사이즈 크기 변경 -> 35, 55라인


# Input data CHANEL mismatching problem
# print("Model structure: \n", model)
# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#                   ~
#   ...
#   (fc): Linear(in_features=512, out_features=1000, bias=True)
#                                              ~~~~
# )
#  -> 모델이 데이터와 channel, class 개수가 다름!
#

model.conv1 = nn.Conv2d(
    1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
)
# -> (64x)3x7x7 에서 (64x)1x3x3 으로 파라미터(=> kernel) 사이즈 변경! (64 -> output channel)
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)


# Transfer Learning issue
# 일부 레이어를 찾아 트레이닝을 시키지 않을 경우
# ex> FC

print("Before (parameter setting): ", len(next(model.parameters())), " layers")

params_to_update = []
for name, param in model.named_parameters():
    if "fc" in name:
        # print(name)
        # print(param.requires_grad)
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False

print("After (parameter setting): ", len(params_to_update), " layers")


# ⭐️⭐️ 중요 ⭐️⭐️
# 모델 수정 한 후에 CPU에서 GPU로 보내야 함! -> 안그러면 requires_grad X / Cuda Fail
model.to(device=device)
# GPU Available check
print("CUDA: ", next(model.parameters()).is_cuda)

# Print Model
print("Tuned Model structure: \n", model)


# 옵티마이져
optimizer = optim.Adam(params_to_update, lr=learning_rate)
# lr을 iter마다 조정해줌
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.1)
# Classification -> MSE(X) => Entropy로 스코어링 한다
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

    # enumuate -> 걍 다 뽑아버리는거
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target도 GPU로 보냄
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

        process = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # Softmax 중 가장 큰 값

            correct += pred.eq(target.view_as(pred)).sum().item()

            process += 1
            if (process / len(test_loader.dataset) * 100) % 10 == 0:
                print(
                    "Testing: {}/{} ({:.0f}%)".format(
                        process,
                        len(test_loader.dataset),
                        process / len(test_loader.dataset) * 100,
                    )
                )

        test_loss /= len(test_loader.dataset)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
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
