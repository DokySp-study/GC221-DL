# # LeNet5
# • Input: 28x28 (grayscale)N x 1 x 28 x 28
# • conv1: 5x5 conv, 6 kernels, stride: 1, padding: 2
# • ReLU, max pooling 2x2
# • conv2: 5x5 conv, 16 kernels, stride: 1, padding: 0
# • ReLU, max pooling 2x2
# • conv3: 5x5 conv, 120 kernels, stride: 1, padding: 0
# • Flatten
# • Linear(120, 84)
# • ReLU
# • Linear(84, 10)

from operator import mod
from sched import scheduler
from typing import OrderedDict
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim


# 파이썬에서 클래스 상속을 할 때,
# class 클라스이름(상속받을_클라스이름):
#   요런 식으로 정의하면 된다.


# 구현방법 #1
class MyLeNet5_1(nn.Module):
    def __init__(self):
        # self -> 생성된 인스턴스를 가리킴!!

        # 부모 생성자 콜
        super(MyLeNet5_1, self).__init__()

        # input: 24 / 1 channel

        self.conv_1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # Stride 1로하면 ex> 4x4 -> 3x3 이 된다!
        # https://www.researchgate.net/figure/Featuring-steps-to-max-pooling-Here-we-use-kernel-size-of-2-2-and-stride-of-1-ie_fig5_343356912

        self.conv_2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_3 = nn.Conv2d(16, 120, kernel_size=5)
        self.relu = nn.ReLU()

        self.fc_1 = nn.Linear(120, 84)
        self.fc_2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv_1(x)  # channel I:1 / O:6
        x = self.relu(x)  # activation func.
        x = self.maxpool_1(x)  # size 24 -> 12

        x = self.conv_2(x)  # channel I:6 / O:16
        x = self.relu(x)  # activation func.
        # x = self.maxpool_2(x)  # size 12 -> 6
        x = self.maxpool_1(x)  # size 24 -> 12
        # TODO: 굳이 맥스풀을 1,2로 나눈 이유가 있나..?

        x = self.conv_3(x)  # channel I:16 / O:120

        x = x.view(-1, 120)  # flatten
        x = self.fc_1(x)  # FC
        x = self.relu(x)  # activation func
        res = self.fc_2(x)  # FC

        return res


# NN 구현방법 #2
class MyLeNet5_2(nn.Module):
    def __init__(self):
        super(MyLeNet5_2, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            #
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),  # 그냥 이렇게 써도 size 2, stride 2로 알아서 잡힘
            nn.ReLU(),
            #
            nn.Conv2d(16, 120, kernel_size=5),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 120)
        res = self.fc_layers(x)
        return res


# 모델을 이렇게도 만들 수 있음
model3 = nn.Sequential(
    # input: 24 / 1 channel
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(20, 64, 5),
    nn.ReLU(),
)

# 모델을 이렇게도 만들 수 있음 #2
model4 = nn.Sequential(
    OrderedDict(
        [
            ("conv1", nn.Conv2d(1, 20, 5)),
            ("relu1", nn.ReLU()),
            ("conv2", nn.Conv2d(20, 64, 5)),
            ("relu2", nn.ReLU()),
        ]
    )
)


model1 = MyLeNet5_1()
model2 = MyLeNet5_2()
print(model1)
print(model2)
print(model3)
print(model4)


# nn.Conv2d(1, 6, kernel_size=5, padding=2),
# -> non-square kernels and unequal stride and with padding
# nn.Conv2d(1, 6, kernel_size=(5,2), stride=(2,1) padding=(4,2)),
# -> dilation: dilated convolution (CNN#1 61p)

# input = torch.randn(20, 16, 50, 100)
# 20x16x50x100 랜덤 숫자 (normal distribution)


# GPU 사용
batch_size = 16
learning_rate = 0.01
epochs = 10

# 메인 코드
use_cuda = torch.cuda.is_available()
# use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print("[Device: ]", device)


# Load data
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        # Image to tensor type & Data augumentation part
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs
)


test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    # 이미 다운로드 했으니깐 필요 없음
    transform=transforms.Compose(
        # Image to tensor type & Data augumentation part
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)


test_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    **kwargs  # 별 두개 -> 가변인자 / # 별 하나는 키만 나온다
)


print("[Trainset: ]", train_dataset)
print("[Testset: ]", test_dataset)


# 모델 콜
model = MyLeNet5_2()
model.to(device=device)  # 모델을 CPU에서 GPU로 보냄
# model.to("cuda:0")

print(next(model.parameters()).is_cuda)


# 옵티마이져
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
# lr을 iter마다 조정해줌
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.1)
# Classification -> MSE(X) => Entropy로 스코어링 한다
loss_function = nn.CrossEntropyLoss()


# 파이썬 문법 참고
# for key, (val1, val2) in enumerate([[1, 2], [3, 4]]):
#     print(key)
#     print(val1)
#     print(val2)


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: data.DataLoader,
    optimizer: optim,
    loss_function: nn.CrossEntropyLoss,
    epoch: int,
):
    # 트레이닝 모드
    # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
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
    # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch

    # # with <exp> as <var> :
    # with open('textfile.txt', 'r') as file:
    # contents = file.read()

    # # 위 구문과 동일한 내용
    # file = open('textfile.txt', 'r')
    # contents = file.read()
    # file.close()

    # # evaluate model:
    # model.eval()
    # with torch.no_grad():
    #     ...
    #     out_data = model(data)
    #     ...

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():

        # test_loader 구조
        # [[[-0.4242, -0.4242, -0.4242,  ..., -0.4242, -0.4242, -0.4242],
        #   [-0.4242, -0.4242, -0.4242,  ..., -0.4242, -0.4242, -0.4242],
        #   [-0.4242, -0.4242, -0.4242,  ..., -0.4242, -0.4242, -0.4242],
        #   ...,
        #   [-0.4242, -0.4242, -0.4242,  ..., -0.4242, -0.4242, -0.4242],
        #   [-0.4242, -0.4242, -0.4242,  ..., -0.4242, -0.4242, -0.4242],
        #   [-0.4242, -0.4242, -0.4242,  ..., -0.4242, -0.4242, -0.4242]]]]),
        # tensor([7, 6, 8, 5, 0, 6, 4, 7, 3, 7, 1, 1, 9, 6, 2, 4])]

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # Softmax 중 가장 큰 값

            # eq == equal
            # print("=============")
            # print(pred)
            # print(target.view_as(pred))
            # print("=============")
            # =============
            # tensor([[4],[8],[3],[1],[8],[7],[4],[7],[3],[1],[3],[2],[2],[3],[0],[1]])
            # tensor([[4],[8],[3],[1],[8],[7],[4],[7],[3],[1],[3],[2],[2],[3],[0],[1]])
            #
            # pred.eq(target.view_as(pred)).sum().item()
            # 같은게 몇개인지 세서, 다 더 하고, (item) tensor 타입을 숫자로 바꿔줌
            # =============

            correct += pred.eq(target.view_as(pred)).sum().item()

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
