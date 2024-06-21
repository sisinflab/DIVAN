import torch
import torch.nn as nn
from src.Gate import Gate
from fuxictr.pytorch.layers import MLP_Block

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
                 recency_input_dim,
                 content_input_dim,
                 pop_hidden_units,
                 pop_activations,
                 pop_dropout,
                 pop_batch_norm,
                 pop_output_activation
                 ):
        super(PopNet, self).__init__()

        # self.network_recency = Dense(
        #     input_dim=network_recency_input_dim,
        #     hidden_dims=network_recency_hidden_dims,
        #     dropout_rate=network_recency_dropout_rate
        # )
        # self.network_content = Dense(
        #     input_dim=network_content_input_dim,
        #     hidden_dims=network_content_hidden_dims,
        #     dropout_rate=network_content_dropout_rate
        # )
        # self.gate = Gate(
        #     input_dim=network_recency_input_dim + network_content_input_dim,
        #     hidden_dims=gate_hidden_units,
        #     dropout_rate=gate_dropout
        # )

        self.dnn = MLP_Block(input_dim=recency_input_dim + content_input_dim,
                             output_dim=1,
                             hidden_units=pop_hidden_units,
                             hidden_activations=pop_activations,
                             output_activation=pop_dropout,
                             dropout_rates=pop_batch_norm,
                             batch_norm=pop_output_activation)

    def forward(self, news_recency_emb, news_content_emb):
        return self.dnn(torch.concat([news_recency_emb, news_content_emb], dim=1))  # content-specific aggregator
