#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/types.h>


template<typename scalar_t>
__device__ inline scalar_t* smem_proxy() {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    return reinterpret_cast<scalar_t*>(smem);
}


template<typename scalar_t>
__global__ void flash_attn_kernel_v1(
    const scalar_t* Q, const scalar_t* K, const scalar_t* V, const int N, const int d,
    const int Tc, const int Tr, const int Bc, const int Br, const scalar_t softmax_scale,
    scalar_t* l, scalar_t* m, scalar_t* O
) {
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int nh = gridDim.y;

    // original Q K V shape: {B, nh, N, d}
    // original Q K V stride: {nh * N *d, N *d, d, 1}
    const int qkv_offset = (bx * nh * N * d) + (by * N * d);
    // original l, m shape: {B, nh, N}
    // original l, m stride: {nh * N, N, 1}
    const int lm_offset = (bx * nh * N) + (by * N);

    // split N by `N = Tr * Br = Tc * Bc`

    // paralleled kernel Q K V shape {B, nh, Tc, Bc, d}, assume Tc == Tr
    // paralleled kernel Q K V stride {nh * Tc * Bc * d, Tc * Bc * d, Bc * d, d, 1}

    // paralleled kernel l, m shape {B, nh, Tr, Br}, assume Tc == Tr
    // paralleled kernel l, m stride {nh * Tr * Br, Tr * Br, Br, 1}
    auto smem = smem_proxy<scalar_t>();
    const int TILE_SIZE = Bc * d;
    // shared memory layout:
    // Q, K, V: {Bc, d}, S: {Bc, Bc}
    scalar_t* Q_i = smem;
    scalar_t* K_j = &smem[TILE_SIZE * 1];
    scalar_t* V_j = &smem[TILE_SIZE * 2];
    scalar_t* S = &smem[TILE_SIZE * 3];

    for (int j = 0; j < Tc; ++j) {
        for (int x = 0; x < d; ++x) {
            // a Block owns Bc threads, and each one copy a head-dim K, V vector into smem
            K_j[(tx * d) + x] = K[qkv_offset + (j * TILE_SIZE) + (tx * d) + x];
            V_j[(tx * d) + x] = V[qkv_offset + (j * TILE_SIZE) + (tx * d) + x];
        }
        __syncthreads();

        for (int i = 0; i < Tr; ++i) {
            for (int x = 0; x < d; ++x) {
                // each thread copy a head-dim Q vector into smem
                Q_i[(tx * d) + x] = Q[qkv_offset + (i * TILE_SIZE) + (tx * d) + x];
            }
            // load previous l, m value 
            scalar_t row_m_prev = m[lm_offset + (i * Br) + tx];
            scalar_t row_l_prev = l[lm_offset + (i * Br) + tx];

            scalar_t row_m = -INFINITY;
            for (int y = 0; y < Bc; ++y) {
                scalar_t sum = 0;
                for (int x = 0; x < d; ++x) {
                    // calculate Q_i * K_j^T
                    sum += Q_i[(tx * d) + x] * K_j[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(tx * Bc) + y] = sum;

                if (sum > row_m) {
                    row_m = sum;
                }
            }

            // numeric-safe update: P = exp(S - row_m), row_l = rowsum(P)
            scalar_t row_l = 0;
            for (int y = 0; y < Bc; ++y) {
                S[(tx * Bc) + y] = __expf(S[(tx * Bc) + y] - row_m);
                row_l += S[(tx * Bc) + y];
            }

            // online update l and m
            scalar_t row_m_new = max(row_m, row_m_prev);
            scalar_t row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // write back to HBM
            for (int x = 0; x < d; ++x) {
                // P_ij * V_j
                scalar_t pv = 0;
                for (int y = 0; y < Bc; ++y) {
                    pv += S[(tx * Bc) + y] * V_j[(y * d) + x];
                }
                // update O online
                O[qkv_offset + (i * TILE_SIZE) + (tx * d) + x] = (
                    (1 / row_l_new)
                    * (
                        (row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (i * TILE_SIZE) + (tx * d) + x])
                        + (__expf(row_m - row_m_new) * pv)
                        )
                    );
            }
            m[lm_offset + (i * Br) + tx] = row_m_new;
            l[lm_offset + (i * Br) + tx] = row_l_new;
        }
        __syncthreads(); // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

torch::Tensor toy_flash_attn_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    // impl for MHA and Q, K, V have the same shape {B, nh, N, d}
    TORCH_CHECK(Q.is_cuda(), "Query tensor must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "Key tensor must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "Value tensor must be a CUDA tensor");

    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();


    int B = Q.size(0);     // batch size
    int nh = Q.size(1);    // number of heads
    int N = Q.size(2);     // sequence length
    int d = Q.size(3);     // head dim

    // reshape Q, k, V into {B, nh, Tc, Bc, d} logically
    // int Bc = 32;
    // int Br = 32;
    int Bc = 1;
    int Br = 1;
    int Tr = ceil(static_cast<float>(N) / Br);
    int Tc = ceil(static_cast<float>(N) / Bc);


    auto O = torch::zeros_like(Q, Q.options());
    auto l = torch::zeros({ B, nh, N }, Q.options());
    auto m = torch::full({ B, nh, N }, -INFINITY, Q.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        Q.scalar_type(),
        "toy_flash_attn_cuda",
        ([&] {
            scalar_t softmax_scale = 1.0f / sqrtf(static_cast<scalar_t>(d));

            // share mem with [Q {Bc, d}, K {Bc, d}, V {Bc, d}, S {Bc, Bc}]
            int smem_size = (3 * Bc * d * sizeof(scalar_t)) + (Bc * Bc * sizeof(scalar_t));

            // launch B * nh blocks
            dim3 grid_dim(B, nh);
            // each block has Bc threads, and handle sub Q, K V chunk with {Tc, Bc, d} and {Tr, Br, d} respectively
            dim3 block_dim(Bc);

            int max_sram_size;
            cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
            flash_attn_kernel_v1 << <grid_dim, block_dim, smem_size >> > (
                Q.data_ptr<scalar_t>(),
                K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                N, d, Tc, Tr, Bc, Br, softmax_scale,
                l.data_ptr<scalar_t>(),
                m.data_ptr<scalar_t>(),
                O.data_ptr<scalar_t>()
                );
         })
    );

    return O;
}