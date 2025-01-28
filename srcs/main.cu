#include "../libs/cudarender.h"

__global__ void blur(unsigned char *image, unsigned char *result, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            int pixel_sum = 0;
            int count = 0;

            for (int j = -10; j <= 10; j++) {
                for (int i = -10; i <= 10; i++) {
                    int nx = x + i;
                    int ny = y + j;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        pixel_sum += image[(ny * width + nx) * channels + c];
                        count++;
                    }
                }
            }

            result[(y * width + x) * channels + c] = pixel_sum / count;
        }
    }
}

int main() {
    int width, height, channels;

    unsigned char *img = stbi_load("input/image.jpg", &width, &height, &channels, 3);
    if (img == NULL) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    unsigned char *result = new unsigned char[width * height * channels];

    unsigned char *d_image, *d_result;
    cudaMalloc(&d_image, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_result, width * height * channels * sizeof(unsigned char));
    cudaMemcpy(d_image, img, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    blur<<<gridDim, blockDim>>>(d_image, d_result, width, height, channels);

    cudaMemcpy(result, d_result, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_jpg("output/output.jpg", width, height, channels, result, 100);

    stbi_image_free(img);
    delete[] result;
    cudaFree(d_image);
    cudaFree(d_result);

    return 0;
}