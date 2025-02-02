#include "libs/cudarender.h"
#define BLOCK_SIZE 256

float3 vertices[] = {
    {-0.5f, -0.5f, 0.0f}, // Bottom-left
    { 0.5f, -0.5f, 0.0f}, // Bottom-right
    { 0.5f,  0.5f, 0.0f}, // Top-right
    {-0.5f,  0.5f, 0.0f}  // Top-left
};
int indices[] = {
    0, 1, 2, // First triangle
    2, 3, 0  // Second triangle
};

__global__ void fill(int xmin, int xmax, int ymin, int ymax, int3 a, int3 b, int3 c, uchar4* buffer, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (xmax - xmin + 1) * (ymax - ymin + 1))
        return;
    int widthRange = xmax - xmin + 1;
    int x = idx % widthRange + xmin;
    int y = idx / widthRange + ymin;

    if (x >= width || y >= height)
        return;

    int3 p = make_int3(x, y, 0);

    int3 ab = make_int3(b.x - a.x, b.y - a.y, 0);
    int3 bc = make_int3(c.x - b.x, c.y - b.y, 0);
    int3 ca = make_int3(a.x - c.x, a.y - c.y, 0);

    int3 ap = make_int3(p.x - a.x, p.y - a.y, 0);
    int3 bp = make_int3(p.x - b.x, p.y - b.y, 0);
    int3 cp = make_int3(p.x - c.x, p.y - c.y, 0);

    int cross1 = ab.x * ap.y - ab.y * ap.x;
    int cross2 = bc.x * bp.y - bc.y * bp.x;
    int cross3 = ca.x * cp.y - ca.y * cp.x;

    if ((cross1 >= 0 && cross2 >= 0 && cross3 >= 0))
        if (p.x >= 0 && p.x < width && p.y >= 0 && p.y < height)
            buffer[p.y * width + p.x] = make_uchar4(255, 128 + (a.x * 255), 255, 255);
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

__global__ void fillBuffer(uchar4* buffer, int width, int height, float3 *vertices, int *indices) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    int3 a = make_int3(__float2int_rn((vertices[indices[globalIdx * 3]].x + 1.0) / 2.0 * (width - 1)),
                        __float2int_rn((vertices[indices[globalIdx * 3]].y + 1.0) / 2.0 * (height - 1)),
                        0);
    int3 b = make_int3(__float2int_rn((vertices[indices[globalIdx * 3 + 1]].x + 1.0) / 2.0 * (width - 1)),
                        __float2int_rn((vertices[indices[globalIdx * 3 + 1]].y + 1.0) / 2.0 * (height - 1)),
                        0);
    int3 c = make_int3(__float2int_rn((vertices[indices[globalIdx * 3 + 2]].x + 1.0) / 2.0 * (width - 1)),
                        __float2int_rn((vertices[indices[globalIdx * 3 + 2]].y + 1.0) / 2.0 * (height - 1)),
                        0);

    int xmin = max(0,min(a.x, min(b.x, c.x)));
    int xmax = min(width - 1, max(a.x, max(b.x, c.x)));
    int ymin = max(0,min(a.y, min(b.y, c.y)));
    int ymax = min(height - 1, max(a.y, max(b.y, c.y)));

    int xrange = xmax - xmin + 1;
    int yrange = ymax - ymin + 1;
    int threadCount = min(BLOCK_SIZE, xrange * yrange);
    int blockCount = max(1 ,((xrange * yrange) + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    fill<<<blockCount, threadCount>>>(xmin, xmax, ymin, ymax, a, b, c, buffer, width, height);
}

//__device__ float3 modelToWorld(float3 model, float3 position, float3 projection, float3 view) {
//}

void render(uchar4 *d_buffer, int width, int height)
{
    int *d_indices;
    float3 *d_vertices;
    cudaMalloc(&d_vertices, sizeof(vertices));
    cudaMemcpy(d_vertices, vertices, sizeof(vertices), cudaMemcpyHostToDevice);
    cudaMalloc(&d_indices, sizeof(indices));
    cudaMemcpy(d_indices, indices, sizeof(indices), cudaMemcpyHostToDevice);
    
    int numIndices = sizeof(indices)/(sizeof(int) * 3);
    fillBuffer<<<max(1, numIndices/1024), min(1024, numIndices)>>>(d_buffer, width, height, d_vertices, d_indices);
}
