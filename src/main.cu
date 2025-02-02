#include "libs/cudarender.h"

GLuint pbo[2];
cudaGraphicsResource* cuda_pbo_resource[2]; 
int width = 1250;
int height = 720;

int main() {
    GLFWwindow* window = openWindow(1250, 720, "CUDA OpenGL Interop", NULL);
    if (!window) {
        return -1;
    }

    createPBOs(width, height);
    glfwSetFramebufferSizeCallback(window, updateBuffersize);
    
	while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        updateContent(width, height, window);
    }

    for (int i = 0; i < 2; i++) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource[i]);
        glDeleteBuffers(1, &pbo[i]);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
