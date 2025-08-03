import math
import torch
from torch.nn import functional as F


def mm_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    p = F.softmax(att, dim=-1)
    return p @ v
