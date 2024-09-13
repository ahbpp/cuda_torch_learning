#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(col) TORCH_CHECK(col.device().is_cuda(), #col " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(col) TORCH_CHECK(col.is_contiguous(), #col " must be contiguous")
#define CHECK_INPUT(col) CHECK_CUDA(col); CHECK_CONTIGUOUS(col)


inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
constexpr int TILE_SIZE = 16;


__global__ void matrixMulKernel(float *first, float *second, float *output, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += first[row * n + i] * second[i * k + col];
        }
        output[row * k + col] = sum;
    }
}

__global__ void tiledMatrixMulKernel(
    float *first, 
    float *second, 
    float *output, 
    int m, 
    int n, 
    int k) {

    __shared__ float firstTile[TILE_SIZE][TILE_SIZE];
    __shared__ float secondTile[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    // blockDim.y == blockDim.x  == TILE_SIZE
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float pValue = 0.0;
    for (int ph = 0; ph < (n + TILE_SIZE - 1) / TILE_SIZE; ++ph) {
        int firstX = ph * TILE_SIZE + tx;  
        // int firstY = by * TILE_SIZE + ty; //row
        firstTile[ty][tx] = (firstX < n && row < m) ? first[row * n + firstX] : 0.0f;

        
        // int secondX = bx * TILE_SIZE + tx; // col
        int secondY = ph * TILE_SIZE + ty;
        secondTile[ty][tx] = (col < k && secondY < n) ? second[secondY * k + col] : 0.0f;

        // !!!Version if column major layout is used !!!!
        // More efficient version for column major layout
        // column major layout is used for transposed matrices
        // Major rule: tx should not be multiplied by k 
        // Improves performance by 15-20% 
        // by default, the code is optimized for row major layout
        // int secondX = bx * TILE_SIZE + ty;
        // int secondY = ph * TILE_SIZE + tx;
        // if (secondX < k && secondY < n) {
        //     secondTile[tx][ty] = second[secondY * k + secondX];
        // } else {
        //     secondTile[tx][ty] = 0.0f;
        // }
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j) {
            pValue += firstTile[ty][j] * secondTile[j][tx];
        }
        __syncthreads();

    }
    if (row < m && col < k) {
        output[row * k + col] = pValue;
    }
}


torch::Tensor matmul(const torch::Tensor first, const torch::Tensor second) {
    CHECK_INPUT(first);
    CHECK_INPUT(second);
    TORCH_CHECK(first.size(1) == second.size(0), "Matrix dimensions must match!");

    const auto m = first.size(0);
    const auto n = first.size(1);
    const auto k = second.size(1);

    auto output = torch::empty({m, k}, first.options());

    const dim3 blockSize(16, 16);
    const dim3 gridSize(cdiv(k, blockSize.x), cdiv(m, blockSize.y));

    matrixMulKernel<<<gridSize, blockSize, 0, torch::cuda::getCurrentCUDAStream()>>>(
        first.data_ptr<float>(),
        second.data_ptr<float>(),
        output.data_ptr<float>(),
        m, n, k
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}


torch::Tensor tiled_matmul(const torch::Tensor first, const torch::Tensor second) {
    CHECK_INPUT(first);
    CHECK_INPUT(second);
    TORCH_CHECK(first.size(1) == second.size(0), "Matrix dimensions must match!");

    const auto m = first.size(0);
    const auto n = first.size(1);
    const auto k = second.size(1);

    auto output = torch::empty({m, k}, first.options());

    const dim3 blockSize(TILE_SIZE, TILE_SIZE);
    const dim3 gridSize(cdiv(k, blockSize.x), cdiv(m, blockSize.y));

    tiledMatrixMulKernel<<<gridSize, blockSize, 0, torch::cuda::getCurrentCUDAStream()>>>(
        first.data_ptr<float>(),
        second.data_ptr<float>(),
        output.data_ptr<float>(),
        m, n, k
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
