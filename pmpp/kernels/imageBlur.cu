#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

// #define CHECK_CUDA(image) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}


__global__ void blur_kernel(unsigned char *image, unsigned char *output, int width, int height, int kernelRadius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;

    int baseOffset = channel * height * width;
    if (x < width && y < height) {
        int pixelValue = 0;
        int pixelCount = 0;

        for (int i = -kernelRadius; i <= kernelRadius; i++) {
            for (int j = -kernelRadius; j <= kernelRadius; j++) {
                int currX = x + i;
                int currY = y + j;
                if (currX >= 0 && currX < width && currY >= 0 && currY < height) {
                    pixelValue += image[baseOffset + currY * width + currX];
                    ++count;
                }
            }
        }

        output[baseOffset + y * width + x] = pixelValue / count;
    }
}


torch::Tensor blur_filter(torch::Tensor image, int radius) {
    assert(image.is_contiguous() && "Input tensor must be contiguous");
    assert(image.device().is_cuda() && "Input tensor must be on CUDA device");
    assert(image.dtype() == torch::kByte && "Input tensor must be of type Byte");
    assert(radius > 0 && "Radius must be greater than 0");

    const auto channels = image.size(0);
    const auto height = image.size(1);
    const auto width = image.size(2);

    auto output = torch::empty_like(image);

    const dim3 blockSize(16, 16, channels);
    const dim3 gridSize(cdiv(width, blockSize.x), cdiv(height, blockSize.y), 1);

    blur_kernel<<<gridSize, blockSize>>>(
        image.data_ptr<unsigned char>(),
        output.data_ptr<unsigned char>(),
        width, height, radius
    );

}