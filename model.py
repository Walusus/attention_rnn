import torch
import torch.nn as nn


# Class representing attention layer
class Attention(nn.Module):
    eps = 1e-10

    def __init__(self, feature_num):
        super(Attention, self).__init__()
        self.dense = nn.Linear(feature_num, 1, bias=True)

    def forward(self, x):
        u = torch.tanh(self.dense(x)).squeeze(2)

        v = torch.exp(u)
        a = v / (torch.sum(v, 1, keepdim=True) + Attention.eps)

        # Multiply input by computed weights
        x = x * a.unsqueeze(-1)
        return torch.sum(x, 1)


class Net(nn.Module):
    def __init__(self, vocab_size):
        super(Net, self).__init__()

        embed_size = 500
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.RNN(embed_size, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.attention = Attention(128 * 2)

        # TODO dropout??
        self.linear1 = nn.Linear(128 * 2, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)

        x, _ = self.rnn(x)
        x = self.attention(x)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x
