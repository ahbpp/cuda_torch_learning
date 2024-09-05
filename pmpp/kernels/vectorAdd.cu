#include <stdio.h>
#include <math.h>
#include <sys/time.h>



__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 10000000;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int size = n * sizeof(float);

    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    for (int i = 0; i < n; i++) {
        a[i] = i + (float)rand() / RAND_MAX;
        b[i] = i + (float)rand() / RAND_MAX;
    }

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = ceil((float)n / blockSize);
    int numIterations = 512;
    // start time in milliseconds
    timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < numIterations; i++) {
        vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    printf("GPU time: %ld\n", end.tv_sec * 1000 + end.tv_usec / 1000 - start.tv_sec * 1000 - start.tv_usec / 1000);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // time_t start, end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < numIterations; i++) {
        vectorAddCPU(a, b, c, n);
    }
    gettimeofday(&end, NULL);
    printf("CPU time: %ld\n", end.tv_sec * 1000 + end.tv_usec / 1000 - start.tv_sec * 1000 - start.tv_usec / 1000);

    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", a[i], b[i], c[i]);
    }

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
