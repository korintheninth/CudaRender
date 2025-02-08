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

#define TILE_SIZE 4
#define MAX_TRIANGLES_PER_TILE 1000

typedef struct triangle {
	float3 a;
	float3 b;
	float3 c;
	float3 normal;
	int xmin;
	int xmax;
	int ymin;
	int ymax;
} triangle;

typedef struct tile {
	int numTriangles;
	size_t triangles[MAX_TRIANGLES_PER_TILE];
} tile;


GLFWwindow* openWindow(int width, int height, const char* title, GLFWmonitor* monitor);
void closeWindow(GLFWwindow* window);
void createPBOs(int width, int height);
void updateBuffersize(GLFWwindow *window, int width, int height);
void updateContent(int width, int height, GLFWwindow* window);
void render(uchar4 *d_buffer, int width, int height, float3 *d_vertices, int *d_indices,
	int numTriangles, float *depthBuffer, float3 *d_screenVertices, float3 *d_normals);
bool LoadModel(const std::string& fileDir, std::vector<int>& indices, std::vector<float3>& vertices, int* numIndices, int* numVertices);
__device__ float atomicMinFloat(float* address, float val);
__device__ void matMult(float mat1[4][4], float mat2[4][4], float result[4][4]);
__device__ void matVecMult(float mat[4][4], float vec[4], float result[4]);
__device__ float3 modelToWorld(float3 model, float3 position, int width, int height);
__device__ void projectionMatrix(float fov, float aspect, float near, float far, float mat[4][4]);
__device__ float3 viewportTransformation(float3 world, int width, int height);
__device__ float pointDepth(float3 a, float3 b, float3 c, int x, int y);
__device__ float3 calculateSurfaceNormal(float3 a, float3 b, float3 c);
__device__ float edgeFunction(float3 a, float3 b, float x, float y);
__global__ void initBuffer(float *Buffer, int size, float value);
__global__ void initTiles(int width, int height, tile *tiles);
__device__ float3 normalize(float3 v);
__device__ float dot(float3 a, float3 b);
__device__ bool insideTriangle(triangle t, int3 p);
void newRender(uchar4 *d_buffer, int width, int height, float3 *d_vertices, int *d_indices,
		int numTriangles, float *depthBuffer, triangle *d_triangles, tile *d_tiles);

#endif