import os
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load


toy_mha_flash_attn = load(
    name="toy_mha_flash_attn",
    sources=[
        "csrc/flash_attn.cu",
        "csrc/binding.cpp",
    ],
    extra_cflags=["-O2", "-std=c++17"],
)


def mm_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
    p = F.softmax(att, dim=-1)
    return p @ v


def test_flash_attn():
    batch_size = 16
    n_head = 12
    seq_len = 64
    head_embd = 64

    q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

    mm_result = mm_attn(q, k, v)
    toy_result = toy_mha_flash_attn.toy_flash_attn_cuda(q, k, v)

    print("attn values sanity check:", torch.allclose(mm_result, toy_result, rtol=0, atol=1e-02))
