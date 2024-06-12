import torch
import torch.nn as nn
from src.Gate import Gate


class Dense(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims=[100],
                 dropout_rate=0.2,
                 ):
        super(Dense, self).__init__()

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


class PopNet(nn.Module):
    def __init__(self,
                 network_recency_input_dim,
                 network_recency_hidden_dims,
                 network_recency_dropout_rate,
                 network_content_input_dim,
                 network_content_hidden_dims,
                 network_content_dropout_rate,
                 gate_hidden_units=[100],
                 gate_dropout=0,
                 ):
        super(PopNet, self).__init__()

        self.network_recency = Dense(
            input_dim=network_recency_input_dim,
            hidden_dims=network_recency_hidden_dims,
            dropout_rate=network_recency_dropout_rate
        )
        self.network_content = Dense(
            input_dim=network_content_input_dim,
            hidden_dims=network_content_hidden_dims,
            dropout_rate=network_content_dropout_rate
        )
        self.gate = Gate(
            input_dim=network_recency_input_dim + network_content_input_dim,
            hidden_dims=gate_hidden_units,
            dropout_rate=gate_dropout
        )

    def forward(self, news_recency_emb, news_content_emb):
        pr = self.network_recency(news_recency_emb)
        pc = self.network_content(news_content_emb)
        theta = self.gate(torch.concat([news_recency_emb, news_content_emb], dim=1))  # content-specific aggregator
        p = theta * pr + (1 - theta) * pc
        return p
