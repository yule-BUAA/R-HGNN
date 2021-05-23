import torch.nn as nn


class Classifier(nn.Module):
    """
    a single layer Classifier
    """
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.linear = nn.Linear(n_hid, n_out)

    def forward(self, x):
        output = self.linear(x)
        return output
