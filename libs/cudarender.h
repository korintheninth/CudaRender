#ifndef CUDA_RENDER_H
# define CUDA_RENDER_H
#include "libs/external/glew-2.1.0/include/GL/glew.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include "external/glfw/glfw-3.3.8.bin.WIN64/include/GLFW/glfw3.h"

#define WIDTH 800
#define HEIGHT 600

GLuint pbo[2];
cudaGraphicsResource* cuda_pbo_resource[2]; 
int pboIndex = 0;

GLFWwindow* openWindow(int width, int height, const char* title, GLFWmonitor* monitor);
void		closeWindow(GLFWwindow* window);
void createPBOs();
__global__ void fillBuffer(uchar4* buffer, int width, int height);
void updateBuffer();
void render();

#endif