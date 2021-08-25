import torch


class FeedForward(torch.nn.Module):
    def __init__(self, len_stroke, hidden_neurons):
        super(FeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(len_stroke*2, hidden_neurons)
        self.fc2 = torch.nn.Linear(hidden_neurons, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class RNN(torch.nn.Module):
    def __init__(self, features, hidden_neurons, num_layers):
        super(RNN, self).__init__()

        self.rnn = torch.nn.RNN(
            input_size=features,           # number of features
            hidden_size=hidden_neurons,         # number of hidden units
            num_layers=num_layers,           # number of rnn layer
            batch_first=True,       # input & output will have batch size as 1st dimension
        )

        self.hidden = None
        self.out = torch.nn.Linear(hidden_neurons, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        r_out, self.hidden = self.rnn(x, self.hidden)
        out = self.relu(self.hidden)
        out = self.out(out)
        out = self.sigmoid(out)
        return out


class LSTM(torch.nn.Module):
    def __init__(self, features, hidden_neurons, num_layers):
        super(LSTM, self).__init__()

        self.lstm = torch.nn.LSTM(input_size=features,
                                  hidden_size=hidden_neurons,
                                  num_layers=num_layers,
                                  batch_first=True)  # lstm
        self.hidden = None
        self.fc = torch.nn.Linear(hidden_neurons, 1)  # fully connected
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        output, (hn, cn) = self.lstm(x, None)  # lstm with input, hidden, and internal state
        out = self.relu(hn)
        out = self.fc(out)  # Final Output
        out = self.sigmoid(out)
        return out


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.adapt = torch.nn.AdaptiveMaxPool2d((5, 2))
        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(3*5*2, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.adapt(self.relu(self.conv1(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.relu(self.fc(x))
        x = self.sigmoid(x)
        return x
