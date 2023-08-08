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
    
class HiddenLayer(nn.Module):

    def __init__(self, in_dim, out_dim, layer_norm=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.layer_norm = layer_norm
        if layer_norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.relu(x)
        return x

class Baseline1(nn.Module):

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_size: int,
            n_hidden_layers: int = 1,
            n_hidden_heads: int = 1,
            layer_norm: bool = False,
            dueling=True,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.hidden_size = hidden_size

        input_layer = HiddenLayer(in_dim, hidden_size, layer_norm)
        hidden_layers = [input_layer] + [HiddenLayer(hidden_size, hidden_size, layer_norm) for _ in range(n_hidden_layers)]
        self.feature_layer = nn.Sequential(*hidden_layers)

        advantage_layers = [HiddenLayer(hidden_size, hidden_size, layer_norm) for _ in range(n_hidden_heads)] + [nn.Linear(hidden_size, out_dim)]
        self.advantage = nn.Sequential(*advantage_layers)

        if dueling:
            value_layers = [HiddenLayer(hidden_size, hidden_size, layer_norm) for _ in range(n_hidden_heads)] + [nn.Linear(hidden_size, 1)]
            self.value = nn.Sequential(*value_layers)
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