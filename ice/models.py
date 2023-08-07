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
        r = self.feature_layer(x)
        if self.value is None:
            Q = self.advantage(r)
        else:
            advantage = self.advantage(r) # B x out_dim
            value = self.value(r) # B x 1
            Q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return Q