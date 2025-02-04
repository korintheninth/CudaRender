#include "libs/cudarender.h"

extern GLuint pbo[2];
extern cudaGraphicsResource* cuda_pbo_resource[2]; 
extern float3 *d_vertices;
extern int *d_indices;
extern int numTriangles;
extern float *depthBuffer;

int pboIndex = 0;

void updateBuffer(int width, int height) {
    int nextPBO = (pboIndex + 1) % 2;
    uchar4* d_buffer;
    size_t buffer_size;

    cudaGraphicsMapResources(1, &cuda_pbo_resource[nextPBO]);
    cudaGraphicsResourceGetMappedPointer((void**)&d_buffer, &buffer_size, cuda_pbo_resource[nextPBO]);

    render(d_buffer, width, height, d_vertices, d_indices, numTriangles, depthBuffer);
	
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaGraphicsUnmapResources(1, &cuda_pbo_resource[nextPBO]);

    pboIndex = nextPBO;
}

void writeToWindow(int width, int height) {
    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[pboIndex]);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL Error: " << err << std::endl;
    }
}

void updateContent(int width, int height, GLFWwindow* window) {
    if (width == 0 || height == 0)
        return;
    updateBuffer(width, height);
    writeToWindow(width, height);
    glfwSwapBuffers(window);
}