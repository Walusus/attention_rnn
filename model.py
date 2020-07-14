import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()

        # TODO resolve step_dim
        step_dim = 2

        # TODO replace with nn.linear
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)

        self.weight = nn.Parameter(weight)
        self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x):
        # TODO trim and tune
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class Net(nn.Module):
    def __init__(self, embed_size):
        super(Net, self).__init__()

        # TODO dropout??
        # TODO test embedding

        # TODO tune num_layers
        # TODO test LSTM/GRU layers
        self.rnn = nn.RNN(embed_size, 128, num_layers=3, bidirectional=True, batch_first=True)

        self.attention = Attention(128 * 2)

        self.linear1 = nn.Linear(64 * 2, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.attention(x)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
