#ifndef CUDA_RENDER_H
# define CUDA_RENDER_H
#include "libs/external/glew-2.1.0/include/GL/glew.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include "external/glfw/glfw-3.3.8.bin.WIN64/include/GLFW/glfw3.h"

GLFWwindow* openWindow(int width, int height, const char* title, GLFWmonitor* monitor);
void		closeWindow(GLFWwindow* window);
void createPBOs(int width, int height);
void updatePBOsize(GLFWwindow *window, int width, int height);
__global__ void fillBuffer(uchar4* buffer, int width, int height);
void updateBuffer(int width, int height);
void render(int width, int height);

#endif