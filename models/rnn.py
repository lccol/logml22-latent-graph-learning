import torch

from torch import nn
from typing import List, Tuple, Union, Dict, Optional

class LSTMBaseline(nn.Module):
    def __init__(self,
                nfeatures: int,
                hidden_size: int,
                num_layers: int,
                nclasses: int,
                bidirectional: Optional[bool]=True,
                mlp_layers: Optional[List[int]]=None) -> None:
        super(LSTMBaseline, self).__init__()

        self.nfeatures = nfeatures
        self.hidden_size = hidden_size
        self.nclasses = nclasses
        self.bidirectional = bidirectional
        self.mlp_layers = mlp_layers

        self.lstm = nn.LSTM(input_size=nfeatures,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidirectional=bidirectional)

        tmp = []
        start = hidden_size
        for val in mlp_layers:
            tmp.append(nn.Linear(start, val))
            tmp.append(nn.ReLU())
            start = val
        self.fc_intermediate = nn.Sequential(tmp)
        self.fc = nn.Linear(mlp_layers[-1], nclasses)
        self.relu = nn.ReLU()
        return

    def forward(self, x) -> torch.tensor:
        output, (hn, cn) = self.lstm(x)
        hn = hn.view(-1, self.hidden_size)

        out = self.relu(hn)
        out = self.fc(out)
        return out
