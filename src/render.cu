#include "libs/cudarender.h"
#define BLOCK_SIZE 256

__device__ float atomicMinFloat(float* address, float val);
__device__ void matMult(float mat1[4][4], float mat2[4][4], float result[4][4]);
__device__ void matVecMult(float mat[4][4], float vec[4], float result[4]);
__device__ float3 modelToWorld(float3 model, float3 position, int width, int height);
__device__ void projectionMatrix(float fov, float aspect, float near, float far, float mat[4][4]);
__device__ float3 viewportTransformation(float3 world, int width, int height);
__global__ void fill(int xmin, int xmax, int ymin, int ymax, float3 a, float3 b, float3 c, float3 normal, uchar4* buffer, int width, int height, float *depthBuffer);
__global__ void fillBuffer(uchar4* buffer, int width, int height, float3 *vertices, int *indices, float *depthBuffer, int numIndices);
__device__ float pointDepth(float3 a, float3 b, float3 c, int x, int y);
__global__ void initDepthBuffer(float *depthBuffer, int size, float value);
//__device__ void connectVertices(int x1, int y1, int x2, int y2, uchar4* buffer, int width, int height);
void render(uchar4 *d_buffer, int width, int height);

__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__device__ void matMult(float mat1[4][4], float mat2[4][4], float result[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result[i][j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}

__device__ void matVecMult(float mat[4][4], float vec[4], float result[4]) {
    for (int i = 0; i < 4; i++) {
        result[i] = 0.0f;
        for (int j = 0; j < 4; j++) {
            result[i] += mat[i][j] * vec[j];
        }
    }
}

__device__ float3 modelToWorld(float3 model, float3 position, int width, int height) {
    float3 world;

    world.x = model.x + position.x;
    world.y = model.y + position.y;
    world.z = model.z + position.z;
    return world;
}

__device__ void projectionMatrix(float fov, float aspect, float near, float far, float mat[4][4]) {
    float s = 1.0f / tan((fov / 2.0f) * (CUDART_PI / 180.0f));
    float clip1 = -far/(far - near);
    float clip2 = -(far * near)/(far - near);
    mat[0][0] = s / aspect; mat[0][1] = 0.0f; mat[0][2] = 0.0f;  mat[0][3] = 0.0f;
    mat[1][0] = 0.0f;      mat[1][1] = s;    mat[1][2] = 0.0f;  mat[1][3] = 0.0f;
    mat[2][0] = 0.0f;      mat[2][1] = 0.0f; mat[2][2] = clip1; mat[2][3] = -1.0f;
    mat[3][0] = 0.0f;      mat[3][1] = 0.0f; mat[3][2] = clip2; mat[3][3] = 0.0f;
}

__device__ float3 viewportTransformation(float3 world, int width, int height) {
    float3 screen;
    screen.x = (world.x + 1.0f) * width / 2.0f;
    screen.y = (world.y + 1.0f) * height / 2.0f;
    screen.z = world.z;
    return screen;
}

__device__ float pointDepth(float3 a, float3 b, float3 c, int x, int y) {
    float3 ab = {b.x - a.x, b.y - a.y, b.z - a.z};
    float3 ac = {c.x - a.x, c.y - a.y, c.z - a.z};
    float3 bc = {c.x - b.x, c.y - b.y, c.z - b.z};
    float3 bp = {x - b.x, y - b.y, 0};
    float3 cp = {x - c.x, y - c.y, 0};

    float areaABC = 0.5f * (ab.x * ac.y - ab.y * ac.x);
    if (areaABC == 0)
        return 100.0f;
    float alpha = 0.5f * (bc.x * cp.y - bc.y * cp.x) / areaABC;
    float beta = 0.5f * (ac.x * bp.y - ac.y * bp.x) / areaABC;
    float gamma = 1.0f - alpha - beta;

    return alpha * a.z + beta * b.z + gamma * c.z;
}

__device__ float3 calculateSurfaceNormal(float3 a, float3 b, float3 c) {
    float3 ab = {b.x - a.x, b.y - a.y, b.z - a.z};
    float3 ac = {c.x - a.x, c.y - a.y, c.z - a.z};

    float3 normal = {ab.y * ac.z - ab.z * ac.y, 
                     ab.z * ac.x - ab.x * ac.z, 
                     ab.x * ac.y - ab.y * ac.x};

    float length = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    
    if (length > 0.0f) {
        normal.x /= length;
        normal.y /= length;
        normal.z /= length;
    }

    return normal;
}

__global__ void fillBuffer(uchar4* buffer, int width, int height, float3 *vertices, int *indices, float *depthBuffer, int numIndices) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx >= numIndices)
        return;

    float3 position = {0.0f, 0.0f, 40.0f};

    float3 a = modelToWorld(vertices[indices[globalIdx * 3]], position, width, height);
    float3 b = modelToWorld(vertices[indices[globalIdx * 3 + 1]], position, width, height);
    float3 c = modelToWorld(vertices[indices[globalIdx * 3 + 2]], position, width, height);
    float3 normal = calculateSurfaceNormal(a, b, c);

    float a4[4] = {a.x, a.y, a.z, 1};
    float b4[4] = {b.x, b.y, b.z, 1};
    float c4[4] = {c.x, c.y, c.z, 1};

    float a4p[4], b4p[4], c4p[4];
    float projection[4][4];
    projectionMatrix(90.0f, (float)width / (float)height, 0.1f, 100.0f, projection);
    matVecMult(projection, a4, a4p);
    matVecMult(projection, b4, b4p);
    matVecMult(projection, c4, c4p);

    a.x = a4p[0] / a4p[3];
    a.y = a4p[1] / a4p[3];
    a.z = a4p[2] / a4p[3];

    b.x = b4p[0] / b4p[3];
    b.y = b4p[1] / b4p[3];
    b.z = b4p[2] / b4p[3];

    c.x = c4p[0] / c4p[3];
    c.y = c4p[1] / c4p[3];
    c.z = c4p[2] / c4p[3];


    a = viewportTransformation(a, width, height);
    b = viewportTransformation(b, width, height);
    c = viewportTransformation(c, width, height);

    int xmin = max(0, (int)floorf(min(a.x, min(b.x, c.x))));
    int xmax = min(width - 1, (int)ceilf(max(a.x, max(b.x, c.x))));
    int ymin = max(0, (int)floorf(min(a.y, min(b.y, c.y))));
    int ymax = min(height - 1, (int)ceilf(max(a.y, max(b.y, c.y))));


    int xrange = xmax - xmin + 1;
    int yrange = ymax - ymin + 1;
    int threadCount = min(BLOCK_SIZE, xrange * yrange);
    int blockCount = max(1 ,((xrange * yrange) + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    fill<<<blockCount, threadCount>>>(xmin, xmax, ymin, ymax, a, b, c, normal, buffer, width, height, depthBuffer);
}

__global__ void fill(int xmin, int xmax, int ymin, int ymax, float3 a, float3 b, float3 c, float3 normal, uchar4* buffer, int width, int height, float *depthBuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (xmax - xmin + 1) * (ymax - ymin + 1))
        return;
    int widthRange = xmax - xmin + 1;
    int x = idx % widthRange + xmin;
    int y = idx / widthRange + ymin;

    if (x >= width || y >= height)
        return;

    int3 p = make_int3(x, y, 0);

    if (pointDepth(a, b, c, p.x, p.y) > depthBuffer[p.y * width + p.x])
        return;

    float cross1 = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
    float cross2 = (c.x - b.x) * (p.y - b.y) - (c.y - b.y) * (p.x - b.x);
    float cross3 = (a.x - c.x) * (p.y - c.y) - (a.y - c.y) * (p.x - c.x);

    if ((cross1 >= 0 && cross2 >= 0 && cross3 >= 0) || (cross1 <= 0 && cross2 <= 0 && cross3 <= 0))
        if (p.x >= 0 && p.x < width && p.y >= 0 && p.y < height)
        {
            float depth = pointDepth(a, b, c, p.x, p.y);
            atomicMinFloat(&depthBuffer[p.y * width + p.x], depth);
            float3 lightColor = {255.0f, 255.0f, 255.0f};
            float3 lightDirection = {0.0f, 0.0f, -1.0f};
            float cost = -normal.z;
            if (cost < 0)
                cost = 0;
            buffer[p.y * width + p.x] = make_uchar4(lightColor.x * cost, lightColor.y * cost, lightColor.z * cost, 255);
        }
}

__global__ void initDepthBuffer(float *depthBuffer, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        depthBuffer[idx] = value;
    }
}
void render(uchar4 *d_buffer, int width, int height, float3 *d_vertices, int *d_indices, int numIndices, float *depthBuffer) {
    int threadCount = min(BLOCK_SIZE, numIndices);
    int blockCount = max(1, (numIndices + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int numPixels = width * height;

    initDepthBuffer<<<(numPixels + 255)/256, 256>>>(depthBuffer, numPixels, 100.0f);
    cudaDeviceSynchronize();
    fillBuffer<<<blockCount, threadCount>>>(d_buffer, width, height, d_vertices, d_indices, depthBuffer, numIndices);
    cudaDeviceSynchronize();
    cudaGetLastError();
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}

//__device__ void connectVertices(int x1, int y1, int x2, int y2, uchar4* buffer, int width, int height) {
//    int dx = abs(x2 - x1), dy = abs(y2 - y1);
//    int sx = (x1 < x2) ? 1 : -1;
//    int sy = (y1 < y2) ? 1 : -1;
//    int err = dx - dy;
//
//    while (true) {
//        int index = y1 * width + x1;
//        if (index >= 0 && index < width * height)
//            buffer[index] = make_uchar4(255, 255, 255, 255);
//
//        if (x1 == x2 && y1 == y2) break;
//
//        int e2 = 2 * err;
//        if (e2 > -dy) { err -= dy; x1 += sx; }
//        if (e2 < dx) { err += dx; y1 += sy; }
//    }
//}