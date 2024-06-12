import torch.nn as nn

class Gate(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims=[100],
                 dropout_rate=0.2,
                 ):
        super(Gate, self).__init__()

        layers = []
        prev_dim = input_dim
        if len(hidden_dims) > 0:

            for i in range(len(hidden_dims)):
                layers.append(nn.Linear(prev_dim, hidden_dims[i]))
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dims[i]

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
