import os
import sys
import torch
import pytest
from torch.utils.cpp_extension import load

from utils import mm_attn


toy_mha_flash_attn = load(
    name="toy_mha_flash_attn",
    sources=[
        "csrc/flash_attn.cu",
        "csrc/binding.cpp",
    ],
    extra_cflags=["-O2", "-std=c++17"],
)

device = torch.device("cuda")

torch.manual_seed(0)
torch.cuda.manual_seed(0)


@pytest.mark.parametrize(
    "batch_size, n_head, seq_len, head_embd, dtype",
    [
        (1, 1, 2, 1, torch.float32),
        (16, 12, 64, 64, torch.float32),
        (16, 8, 64, 64, torch.float32),
        (16, 12, 64, 64, torch.float32),
        (64, 4, 256, 16, torch.float32),
    ],
)
def test_flash_attn(
    batch_size: int,
    n_head: int,
    seq_len: int,
    head_embd: int,
    dtype: torch.dtype,
):
    q = torch.randn(batch_size, n_head, seq_len, head_embd, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_head, seq_len, head_embd, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_head, seq_len, head_embd, device=device, dtype=dtype)

    mm_result = mm_attn(q, k, v)
    toy_result = toy_mha_flash_attn.toy_flash_attn_cuda(q, k, v)

    assert torch.allclose(mm_result, toy_result, rtol=0, atol=1e-02), f"mm_result: {mm_result}, toy_result: {toy_result}"


@pytest.mark.skipif("-s" not in sys.argv, reason="Run this test only with pytest -s")
@pytest.mark.parametrize(
    "batch_size, n_head, seq_len, head_embd, dtype",
    [
        (16, 12, 64, 64, torch.float16),
    ],
)
def test_ad_prof(
    batch_size: int,
    n_head: int,
    seq_len: int,
    head_embd: int,
    dtype: torch.dtype,
):
    q = torch.randn(batch_size, n_head, seq_len, head_embd, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_head, seq_len, head_embd, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_head, seq_len, head_embd, device=device, dtype=dtype)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        mm_attn(q, k, v)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("=== profiling minimal flash attention === ")

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        toy_mha_flash_attn.toy_flash_attn_cuda(q, k, v)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
