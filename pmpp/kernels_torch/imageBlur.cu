#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(col) TORCH_CHECK(col.device().is_cuda(), #col " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(col) TORCH_CHECK(col.is_contiguous(), #col " must be contiguous")
#define CHEK_TYPE(col) TORCH_CHECK(col.dtype() == torch::kByte, #col  " must must be of type Byte")
#define CHECK_INPUT(col) CHECK_CUDA(col); CHECK_CONTIGUOUS(col); CHEK_TYPE(col)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}


__global__ void blur_kernel(unsigned char *output, unsigned char *input, int width, int height, int kernelRadius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;

    int baseOffset = channel * height * width;
    if (col < width && row < height) {
        unsigned int pixelValue = 0;
        unsigned int pixelCount = 0;

        for (int i = -kernelRadius; i <= kernelRadius; i++) {
            int currY = row + i;
            for (int j = -kernelRadius; j <= kernelRadius; j++) {
                int currX = col + j;
                if (currY >= 0 && currY < height && currX >= 0 && currX < width) {
                    pixelValue += input[baseOffset + currY * width + currX];
                    ++pixelCount;
                }
            }
        }

        output[baseOffset + row * width + col] = (unsigned char)(pixelValue / pixelCount);
    }
}


torch::Tensor blur_filter(torch::Tensor image, int radius) {
    // assert(image.is_contiguous() && "Input tensor must be contiguous");
    // assert(image.device().is_cuda() && "Input tensor must be on CUDA device");
    // assert(image.dtype() == torch::kByte && "Input tensor must be of type Byte");
    CHECK_INPUT(image);
    assert(radius > 0 && "Radius must be greater than 0");

    const auto channels = image.size(0);
    const auto height = image.size(1);
    const auto width = image.size(2);

    auto output = torch::empty_like(image);

    const dim3 blockSize(16, 16, channels);
    const dim3 gridSize(cdiv(width, blockSize.x), cdiv(height, blockSize.y));

    blur_kernel<<<gridSize, blockSize, 0, torch::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width, height, radius
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
