#include "libs/cudarender.h"


int main() {
    GLFWwindow* window = openWindow(WIDTH, HEIGHT, "CUDA OpenGL Interop", NULL);
    if (!window) {
        return -1;
    }

    createPBOs();
    
	while (!glfwWindowShouldClose(window)) {
        updateBuffer();
        render();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    for (int i = 0; i < 2; i++) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource[i]);
        glDeleteBuffers(1, &pbo[i]);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
