#include "libs/cudarender.h"

GLFWwindow* openWindow(int width, int height, const char* title, GLFWmonitor* monitor) {
	if (!glfwInit()) {
		printf("Failed to initialize GLFW\n");
		return NULL;
	}

	GLFWwindow* window = glfwCreateWindow(width, height, title, NULL, NULL);
	if (!window) {
		printf("Failed to create window\n");
		glfwTerminate();
		return NULL;
	}

	glfwMakeContextCurrent(window);
    
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return NULL;
    }

	return window;
}

void closeWindow(GLFWwindow* window) {
	glfwDestroyWindow(window);
	glfwTerminate();
}