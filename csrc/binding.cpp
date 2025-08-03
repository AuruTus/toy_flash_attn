#include <torch/extension.h>

torch::Tensor toy_flash_attn_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("toy_flash_attn_cuda", &toy_flash_attn_cuda, "Toy Flash Attention CUDA implementation");
}