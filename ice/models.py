import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Lilith(nn.Module):

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_size: int,
            dueling=True,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.hidden_size = hidden_size

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.out_dim)
        )

        if dueling:
            self.value = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1)
            )
        else:
            self.value = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)

        r = self.feature_layer(x)
        if self.value is None:
            Q = self.advantage(r)
        else:
            advantage = self.advantage(r) # B x out_dim
            value = self.value(r) # B x 1
            Q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return Q
    

class LSTM(nn.Module):

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_size: int,
            dueling=True,
            num_feature_layers = 3,
            num_head_layers = 2,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.hidden_size = hidden_size

        self.feature_layer = nn.LSTM(in_dim, self.hidden_size, num_feature_layers, batch_first=True)
        self.advantage = nn.LSTM(self.hidden_size, self.hidden_size, num_head_layers, batch_first=True)
        self.advantage_projection = nn.Linear(self.hidden_size, self.out_dim)

        if dueling:
            self.value = nn.LSTM(self.hidden_size, self.hidden_size, 2, batch_first=True)
            self.value_projection = nn.Linear(self.hidden_size, self.out_dim)
        else:
            self.value = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape B x T x F

        f_out, _ = self.feature_layer(x) # B x T x hidden_size
        if self.value is None:
            Q_hidden, _ = self.advantage(f_out) # B x T x hidden_size
            Q = self.advantage_projection(Q_hidden[:, -1, :])
        else:
            adv_hidden, _ = self.advantage(f_out) # B x T x hidden_size
            adv = self.advantage_projection(adv_hidden[:, -1, :]) # B x out_dim
            value, _ = self.value(f_out) # B x T x 1
            value = self.value_projection(value[:, -1, :]) # B x out_dim
            Q = value + adv - adv.mean(dim=1, keepdim=True)
        return Q
    

