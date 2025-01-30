#include "libs/cudarender.h"

__global__ void fillBuffer(uchar4* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        buffer[idx] = make_uchar4(x % 256, y % 256, 128, 255);
    }
}

void updateBuffer() {
    int nextPBO = (pboIndex + 1) % 2;  // Swap buffer
    uchar4* d_buffer;
    size_t buffer_size;

    cudaGraphicsMapResources(1, &cuda_pbo_resource[nextPBO]);
    cudaGraphicsResourceGetMappedPointer((void**)&d_buffer, &buffer_size, cuda_pbo_resource[nextPBO]);

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);
    fillBuffer<<<gridSize, blockSize>>>(d_buffer, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    
	cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }


    cudaGraphicsUnmapResources(1, &cuda_pbo_resource[nextPBO]);

    pboIndex = nextPBO;
}

void render() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[pboIndex]);
    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL Error: " << err << std::endl;
    }
}