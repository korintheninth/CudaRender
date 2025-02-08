#include "libs/cudarender.h"

GLuint pbo[2];
cudaGraphicsResource* cuda_pbo_resource[2]; 

int width = 1250;
int height = 720;

float3 *d_vertices;
int *d_indices;
int numTriangles;
float *depthBuffer;

triangle *d_triangles;
tile *d_tiles;

int main() {
    GLFWwindow* window = openWindow(width, height, "CUDA OpenGL Interop", NULL);
    if (!window) {
        return -1;
    }
    glfwMaximizeWindow(window);
    glfwGetWindowSize(window, &width, &height);
    cudaMalloc(&depthBuffer, width * height * sizeof(float));

    createPBOs(width, height);
    glfwSetFramebufferSizeCallback(window, updateBuffersize);

    std::vector<int> indices;
    std::vector<float3> vertices;
    int numVertices;
    int numIndices;

    if (!LoadModel("objs/monkey.obj", indices, vertices, &numIndices, &numVertices)) {
        return -1;
    }
    numTriangles = numIndices / 3;
    cudaMalloc(&d_triangles, numTriangles * sizeof(triangle));
    cudaMalloc(&d_tiles, ((width * height) / (TILE_SIZE * TILE_SIZE)) * sizeof(tile));

    cudaMalloc(&d_vertices, vertices.size() * sizeof(float3));
    cudaMemcpy(d_vertices, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_indices, indices.size() * sizeof(int));
    cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    
	while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        updateContent(width, height, window);
    }

    for (int i = 0; i < 2; i++) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource[i]);
        glDeleteBuffers(1, &pbo[i]);
    }

    cudaFree(d_indices);
    cudaFree(d_vertices);
    cudaFree(depthBuffer);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
