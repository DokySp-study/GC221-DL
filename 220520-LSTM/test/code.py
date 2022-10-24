import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import fivewords as dataset

chars = "abcdefghijklmnopqrstuvwxyz"
char_list = [i for i in chars]
n_letters = len(char_list)

n_layers = 4

n_five_words = len(dataset.five_words)


def word_to_onehot(string):
    one_hot = np.array([]).reshape(0, n_letters)
    for i in string:
        idx = char_list.index(i)
        zero = np.zeros(shape=n_letters, dtype=int)
        zero[idx] = 1
        one_hot = np.vstack([one_hot, zero])
    return one_hot


def onehot_to_word(onehot_1):
    onehot = torch.Tensor.numpy(onehot_1)
    return char_list[onehot.argmax()]


class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer):
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True,
        )

    def forward(self, x, h0, c0):
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return out, (hn, cn)

    def init_hidden(self):
        return torch.zeros(self.num_layer, 1, self.hidden_size)

    def init_cell(self):
        return torch.zeros(self.num_layer, 1, self.hidden_size)


def main():
    n_hidden = 26
    lr = 0.001
    epochs = 900

    model = myLSTM(n_letters, n_hidden, n_layers)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

    for i in range(epochs):
        total_loss = 0
        for j in range(n_five_words):

            string = dataset.five_words[j]
            one_hot = torch.from_numpy(word_to_onehot(string)).type_as(
                torch.FloatTensor()
            )
            model.zero_grad()

            h0 = model.init_hidden()
            c0 = model.init_cell()

            input = one_hot[0:-1]
            input = torch.unsqueeze(input, 1)

            # Reshape to N(batch size), L(seq_len), H
            # for `batch_first` = True option

            input = input.reshape([1, 4, n_letters])

            target = np.argmax(one_hot[1:], axis=1)
            output, (hn, cn) = model(input, h0, c0)

            # Reshape to L, N, H
            output = output.reshape([4, 1, n_hidden])

            loss = loss_func(output.squeeze(1), target)
            loss.backward()
            optimizer.step()

        if i % 10 == 0:
            print("epoch%d" % i)
            print(loss.detach())

        scheduler.step()

    # torch.save(model.state_dict(), "trained.tar")

    ####################################################

    # model.load_state_dict(torch.load("trained.tar"))

    with torch.no_grad():
        total = 0
        positive = 0
        total_text = 0
        positive_text = 0
        for i in range(n_five_words):
            string = dataset.five_words[i]
            one_hot = torch.from_numpy(word_to_onehot(string)).type_as(
                torch.FloatTensor()
            )

            h0 = model.init_hidden()
            c0 = model.init_cell()

            input = one_hot[0:-1]
            input = torch.unsqueeze(input, 1)
            input = input.reshape([1, 4, n_letters])

            target = np.argmax(one_hot[1:], axis=1)

            output, (hn, cn) = model(input, h0, c0)
            output = output.reshape([4, 1, n_hidden])
            output = output.squeeze()

            output_string = string[0]
            for j in range(output.size()[0]):
                output_string += onehot_to_word(output[j].data)
                total_text += 1
                if string[j + 1] == output_string[-1]:
                    positive_text += 1

            total += 1
            if string[-1] == output_string[-1]:
                positive += 1

            print("%d GT:%s OUT:%s" % (i + 1, string, output_string))

        print("final text accuracy %d/%d (%.4f)" % (positive, total, positive / total))
        print(
            "whole text accuracy %d/%d (%.4f)"
            % (positive_text, total_text, positive_text / total_text)
        )


if __name__ == "__main__":
    main()
