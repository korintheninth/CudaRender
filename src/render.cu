#include "libs/cudarender.h"
#define BLOCK_SIZE 256

__global__ void fill(int xmin, int xmax, int ymin, int ymax, float3 a, float3 b, float3 c, float3 normal, uchar4* buffer, int width, int height, float *depthBuffer);
__device__ void connectVertices(int x1, int y1, int x2, int y2, uchar4* buffer, int width, int height);


__global__ void getFragments(int xmin, int xmax, int ymin, int ymax, float edge0, float edge1, float edge2,
    float dedgex0, float dedgey0, float dedgex1, float dedgey1, float dedgex2, float dedgey2,
    float3 a, float3 b, float3 c, float *depthBuffer, int width, int height, uchar4 *buffer, float3 normal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    int x = xmin + idx;
    int y = ymin + idy;
    if (x > xmax || y > ymax)
        return;
    
    float depth = pointDepth(a, b, c, x, y);
    if (depth < 0.0f)
        return;
    if (depth > depthBuffer[y * width + x])
        return;
    
    float edge0p = edge0 + dedgex0 * idx + dedgey0 * idy;
    float edge1p = edge1 + dedgex1 * idx + dedgey1 * idy;
    float edge2p = edge2 + dedgex2 * idx + dedgey2 * idy;

    if (edge0p <= 0 && edge1p <= 0 && edge2p <= 0) {
        atomicMinFloat(&depthBuffer[y * width + x], depth);
        float3 lightColor = {255.0f, 255.0f, 255.0f};
        float cost = -normal.z;
        if (cost < 0)
            cost = 0;
        buffer[y * width + x] = make_uchar4(lightColor.x * cost, lightColor.y * cost, lightColor.z * cost, 255);
    }
}

__global__ void rasteriseTriangle(float3 *vertices, int *indices, float *depthBuffer, int width, int height, uchar4 *buffer, int numTriangles, float3 *normals) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx >= numTriangles)
        return;

    float3 a = vertices[indices[globalIdx * 3]];
    float3 b = vertices[indices[globalIdx * 3 + 1]];
    float3 c = vertices[indices[globalIdx * 3 + 2]];

    int xmin = max(0, (int)floorf(fminf(a.x, fminf(b.x, c.x))));
    int xmax = min(width - 1, (int)ceilf(fmaxf(a.x, fmaxf(b.x, c.x))));
    int ymin = max(0, (int)floorf(fminf(a.y, fminf(b.y, c.y))));
    int ymax = min(height - 1, (int)ceilf(fmaxf(a.y, fmaxf(b.y, c.y))));
    

    int xrange = xmax - xmin + 1;
    int yrange = ymax - ymin + 1;

    float edge0 = edgeFunction(b, c, xmin, ymin);
    float edge1 = edgeFunction(c, a, xmin, ymin);
    float edge2 = edgeFunction(a, b, xmin, ymin);


    float dedgex0 = c.y - b.y; float dedgey0 = b.x - c.x;
    float dedgex1 = a.y - c.y; float dedgey1 = c.x - a.x;
    float dedgex2 = b.y - a.y; float dedgey2 = a.x - b.x;

    float3 normal = normals[globalIdx];

    
    int threadx = min(16, xrange);
    int thready = min(16, yrange);

    int blockx = max(1, (xrange + 15) / 16);
    int blocky = max(1, (yrange + 15) / 16);

    dim3 threads(threadx, thready);
    dim3 blocks(blockx, blocky);

    getFragments<<<blocks, threads>>>(xmin, xmax, ymin, ymax, edge0, edge1, edge2, dedgex0, dedgey0, dedgex1, dedgey1, dedgex2, dedgey2,
         a, b, c, depthBuffer, width, height, buffer, normal);
}

__global__ void fillBuffer(int width, int height, float3 *vertices, int *indices, int numTriangles, float3 *screenVertices, float3 *normals) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx >= numTriangles)
        return;

    float3 position = {0.0f, 0.0f, 40.0f};

    float3 a = modelToWorld(vertices[indices[globalIdx * 3]], position, width, height);
    float3 b = modelToWorld(vertices[indices[globalIdx * 3 + 1]], position, width, height);
    float3 c = modelToWorld(vertices[indices[globalIdx * 3 + 2]], position, width, height);

    float3 normal = calculateSurfaceNormal(a, b, c);
    normals[globalIdx] = normal;

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

    screenVertices[indices[globalIdx * 3]] = a;
    screenVertices[indices[globalIdx * 3 + 1]] = b;
    screenVertices[indices[globalIdx * 3 + 2]] = c;

    //connectVertices(a.x, a.y, b.x, b.y, buffer, width, height);

    //int xmin = max(0, (int)floorf(min(a.x, min(b.x, c.x))));
    //int xmax = min(width - 1, (int)ceilf(max(a.x, max(b.x, c.x))));
    //int ymin = max(0, (int)floorf(min(a.y, min(b.y, c.y))));
    //int ymax = min(height - 1, (int)ceilf(max(a.y, max(b.y, c.y))));
//
//
    //int xrange = xmax - xmin + 1;
    //int yrange = ymax - ymin + 1;
    //int threadCount = min(BLOCK_SIZE, xrange * yrange);
    //int blockCount = max(1 ,((xrange * yrange) + BLOCK_SIZE - 1) / BLOCK_SIZE);
    //
    //fill<<<blockCount, threadCount>>>(xmin, xmax, ymin, ymax, a, b, c, normal, buffer, width, height, depthBuffer);
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

void render(uchar4 *d_buffer, int width, int height, float3 *d_vertices, int *d_indices, int numTriangles, float *depthBuffer, float3 *d_screenVertices, float3 *d_normals) {
    int threadCount = min(BLOCK_SIZE, numTriangles);
    int blockCount = max(1, (numTriangles + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int numPixels = width * height;

    cudaDeviceSynchronize();
    initBuffer<<<(numPixels + 255)/256, 256>>>(depthBuffer, numPixels, 100.0f);
    cudaDeviceSynchronize();
    fillBuffer<<<blockCount, threadCount>>>(width, height, d_vertices, d_indices, numTriangles, d_screenVertices, d_normals);
    cudaDeviceSynchronize();
    rasteriseTriangle<<<blockCount, threadCount>>>(d_screenVertices, d_indices, depthBuffer, width, height, d_buffer, numTriangles, d_normals);
    cudaDeviceSynchronize();
    cudaGetLastError();
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}

__device__ void connectVertices(int x1, int y1, int x2, int y2, uchar4* buffer, int width, int height) {
    int dx = abs(x2 - x1), dy = abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        int index = y1 * width + x1;
        if (index >= 0 && index < width * height)
            buffer[index] = make_uchar4(255, 255, 255, 255);

        if (x1 == x2 && y1 == y2) break;

        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 < dx) { err += dx; y1 += sy; }
    }
}