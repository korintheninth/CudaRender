#ifndef CUDA_RENDER_H
# define CUDA_RENDER_H
#include "libs/external/glew-2.1.0/include/GL/glew.h"
#include "external/glfw/glfw-3.3.8.bin.WIN64/include/GLFW/glfw3.h"
#include "external/Assimp/include/assimp/Importer.hpp"
#include "external/Assimp/include/assimp/scene.h"
#include "external/Assimp/include/assimp/postprocess.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <math_constants.h>
#include <iostream>
#include <vector>

GLFWwindow* openWindow(int width, int height, const char* title, GLFWmonitor* monitor);
void closeWindow(GLFWwindow* window);
void createPBOs(int width, int height);
void updateBuffersize(GLFWwindow *window, int width, int height);
void updateContent(int width, int height, GLFWwindow* window);
void render(uchar4 *d_buffer, int width, int height, float3 *d_vertices, int *d_indices, int numIndices, float *depthBuffer);
bool LoadModel(const std::string& fileDir, std::vector<int>& indices, std::vector<float3>& vertices, int* numIndices, int* numVertices);

#endif