import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def word_to_onehot(string):

    global char_list

    start = np.zeros(shape=len(char_list), dtype=int)
    end = np.zeros(shape=len(char_list), dtype=int)

    # 끝에서 두번째, 첫번째
    start[-2] = 1
    end[-1] = 1

    for i in string:
        idx = char_list.index(i)
        zero = np.zeros(shape=len(char_list), dtype=int)
        zero[idx] = 1
        start = np.vstack([start, zero])
    output = np.vstack([start, end])
    return output


def onehot_to_word(onehot_1):
    global char_list
    onehot = torch.Tensor.numpy(onehot_1)
    return char_list[onehot.argmax()]


class myRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.U = nn.Linear(input_size, hidden_size)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, output_size)

        self.activation_func = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input, hidden):
        hidden = self.activation_func(self.U(input) + self.W(hidden))
        output = self.V(hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# main code

chars = "abcdefghijklmnopqrstuvwxyz .,:;?01"
char_list = [i for i in chars]
n_letters = len(char_list)


def run(inputstr):
    n_hidden = 512
    lr = 0.0001
    epochs = 10000

    rnn = myRNN(n_letters, n_hidden, n_letters)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(rnn.parameters(), lr=lr)

    one_hot = torch.from_numpy(word_to_onehot(inputstr)).type_as(torch.FloatTensor())

    for i in range(epochs):

        optimizer.zero_grad()
        total_loss = 0
        hidden = rnn.init_hidden()

        input_ = one_hot[0:1, :]

        for j in range(one_hot.size()[0] - 1):

            target = one_hot[j + 1]
            target_single = (
                torch.from_numpy(np.asarray(target.numpy().argmax()))
                .type_as(torch.LongTensor())
                .view(-1)
            )

            output, hidden = rnn.forward(input_, hidden)
            loss = loss_func(output, target_single)
            total_loss += loss
            input_ = output

        total_loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print("epoch %d" % i)
            print(total_loss)
            start = torch.zeros(1, len(char_list))
            start[:, -2] = 1

            with torch.no_grad():
                hidden = rnn.init_hidden()
                input_ = start
                output_string = ""

                for i in range(len(inputstr)):
                    output, hidden = rnn.forward(input_, hidden)
                    output_string += onehot_to_word(F.softmax(output.data))
                    input_ = output

                print(output_string)


print("Sentence 1")
run("i want to go on a trip these days. how about you?")
print()
print("Sentence 2")
run(
    "i want to go on a trip these days. how about you? i would like to visit france, italy, germany again with my friends."
)
