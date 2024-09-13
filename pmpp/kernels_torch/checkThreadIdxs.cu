#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(col) TORCH_CHECK(col.device().is_cuda(), #col " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(col) TORCH_CHECK(col.is_contiguous(), #col " must be contiguous")
#define CHECK_INPUT(col) CHECK_CUDA(col); CHECK_CONTIGUOUS(col)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}


__global__ void checkThreadIdxs(float *input, unsigned int *blockCounter, int size) {
    __shared__ unsigned int bid_s;
    if (threadIdx.x == 0) {
        bid_s = atomicAdd(blockCounter, 1);
    }
    int num_values = 3;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * num_values;
    if (idx < size) {
        input[idx] = blockDim.x;
        input[idx + 1] = bid_s;
        input[idx + 2] = threadIdx.x;
    }
}

torch::Tensor check_thread_idxs(torch::Tensor input) {
    unsigned int *blockCounter;
    blockCounter = (unsigned int *)malloc(sizeof(unsigned int));
    *blockCounter = 0;
    unsigned int *d_blockCounter;
    cudaMalloc(&d_blockCounter, sizeof(unsigned int));
    cudaMemcpy(d_blockCounter, blockCounter, sizeof(unsigned int), cudaMemcpyHostToDevice);
    CHECK_INPUT(input);
    const auto size = input.size(0);

    const dim3 blockSize(256);
    const dim3 gridSize(cdiv(size, blockSize.x * 3));

    checkThreadIdxs<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        d_blockCounter,
        size
    );

    return input;
}
