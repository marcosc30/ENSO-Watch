
from typing import Any

import torch
from torch import nn

class TemporalAdapter(nn.Module):
    def __init__(self, wrapped: nn.Module) -> None:
        super().__init__()
        self.add_module("wrapped", wrapped)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [BATCH, SEQ_LEN, D_MODEL]
        return self.wrapped(x.transpose(-2, -1)).transpose(-2, -1)
    
class StandardEncoder(nn.Linear):
    def __init__(self, d_input: int, d_model: int, bias: bool = True) -> None:
        super().__init__(in_features=d_input, out_features=d_model, bias=bias)
        self.d_input = d_input
        self.d_model = d_model

class Residual(nn.Module):
    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # noqa
        return y + x


class SequentialWithResidual(nn.Sequential):
    @staticmethod
    def _residual_module(obj: Any) -> bool:
        return isinstance(obj, Residual) or issubclass(type(obj), Residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for module in self:
            if self._residual_module(module):
                y = module(y, x=x)
            else:
                y = module(y)
        return y
