#include "libs/cudarender.h"

GLuint pbo[2];
cudaGraphicsResource* cuda_pbo_resource[2]; 

int width = 1250;
int height = 720;

triangleData data;

triangle    *d_triangles;
tile        *d_tiles;

controls   camera;

int main() {
    camera.lastMousePos = {0.0f, 0.0f};
    camera.rotation = {0.0f, 0.0f, 0.0f};
    camera.position = {0.0f, 0.0f, 0.0f};
    camera.scale = {1.0f, 1.0f, 1.0f};
    camera.mouseLeft = false;

    GLFWwindow* window = openWindow(width, height, "CUDA OpenGL Interop", NULL);
    if (!window) {
        return -1;
    }
    glfwMaximizeWindow(window);
    glfwGetWindowSize(window, &width, &height);
    cudaMalloc(&data.depthBuffer, width * height * sizeof(float));

    createPBOs(width, height);
    glfwSetFramebufferSizeCallback(window, updateBuffersize);

    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);

    std::vector<int> indices;
    std::vector<float3> vertices;
    int numVertices;
    int numIndices;

    if (!LoadModel("objs/monkey.obj", indices, vertices, &numIndices, &numVertices)) {
        return -1;
    }
    data.numTriangles = numIndices / 3;
    cudaMalloc(&d_triangles, data.numTriangles * sizeof(triangle));
    cudaMalloc(&d_tiles, ((width * height) / (TILE_SIZE * TILE_SIZE)) * sizeof(tile));

    cudaMalloc(&data.vertices, vertices.size() * sizeof(float3));
    cudaMemcpy(data.vertices, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice);
    
    cudaMalloc(&data.indices, indices.size() * sizeof(int));
    cudaMemcpy(data.indices, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    
	while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        updateContent(width, height, window);
    }

    for (int i = 0; i < 2; i++) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource[i]);
        glDeleteBuffers(1, &pbo[i]);
    }

    cudaFree(data.indices);
    cudaFree(data.vertices);
    cudaFree(data.depthBuffer);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
