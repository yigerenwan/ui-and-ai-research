from dataclasses import dataclass

import torch


@dataclass
class Cache:
    key_states: torch.Tensor
    value_states: torch.Tensor
