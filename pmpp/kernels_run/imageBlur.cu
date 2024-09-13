#include <stdio.h>
#include <math.h>
#include <stdlib.h>


void read_image(const char *filename, unsigned char **img_data, int width, int height) {
    FILE *file;

    // Open file for reading in binary mode
    file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open file %s for reading\n", filename);
        exit(1);
    }

    // Allocate memory for the image
    *img_data = (unsigned char *)malloc(width * height * 3);
    if (*img_data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        exit(1);
    }

    // Read image data from file
    if (fread(*img_data, sizeof(unsigned char), width * height * 3, file) != width * height * 3) {
        fprintf(stderr, "Error reading image data from file\n");
        fclose(file);
        free(*img_data);
        exit(1);
    }

    // Close the file
    fclose(file);
}



void write_image(const char *filename, unsigned char *img_data, int width, int height) {
    FILE *file;

    // Open file for writing in binary mode
    file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open file %s for writing\n", filename);
        exit(1);
    }

    // Write image data to file
    fwrite(img_data, sizeof(unsigned char), width * height * 3, file);

    // Close the file
    fclose(file);
}

__global__ void blur(unsigned char *image, unsigned char *output, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernelRadius = kernelSize / 2;


    if (x < width && y < height) {
        int red = 0;
        int green = 0;
        int blue = 0;
        int count = 0;


        for (int i = -kernelRadius; i <= kernelRadius; i++) {
            for (int j = -kernelRadius; j <= kernelRadius; j++) {
                int x_new = x + i;
                int y_new = y + j;
                if (x_new >= 0 && x_new < width && y_new >= 0 && y_new < height) {
                    red += image[(y_new * width + x_new) * 3];
                    green += image[(y_new * width + x_new) * 3 + 1];
                    blue += image[(y_new * width + x_new) * 3 + 2];
                    ++count;
                }
            }
        }

        output[(y * width + x) * 3] = red / count;
        output[(y * width + x) * 3 + 1] = green / count;
        output[(y * width + x) * 3 + 2] = blue / count;
    }
}

int main() {
    const char *input_file = "jimmy.rgb";
    const char *output_file = "jimmy_blurred.rgb";
    int width = 1140;
    int height = 711;
    unsigned char *image = NULL;
    read_image(input_file, &image, width, height);
    printf("First pixel value: %d %d %d\n", image[0], image[1], image[2]);

    unsigned char *output = (unsigned char *)malloc(width * height * 3);

    unsigned char *d_image, *d_output;
    cudaMalloc((void **)&d_image, width * height * 3);
    cudaMalloc((void **)&d_output, width * height * 3);

    cudaMemcpy(d_image, image, width * height * 3, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 numBlocks(ceil((float)width / blockSize.x), ceil((float)height / blockSize.y));
    blur<<<numBlocks, blockSize>>>(d_image, d_output, width, height, 5);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        // Clean up if necessary
        cudaFree(d_image);
        cudaFree(d_output);
        free(image);
        free(output);
        return -1;
    }

    cudaMemcpy(output, d_output, width * height * 3, cudaMemcpyDeviceToHost);
    printf("Output first pixel value: %d %d %d\n", output[100], output[100], output[100]);

    write_image(output_file, output, width, height);

    free(image);
    free(output);
    cudaFree(d_image);
    cudaFree(d_output);
    return 0;
}
