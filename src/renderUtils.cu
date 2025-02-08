#include "libs/cudarender.h"

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
    screen.y = height - (world.y + 1.0f) * height / 2.0f;
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

__device__ float edgeFunction(float3 a, float3 b, float x, float y) {
    return (x - a.x) * (b.y - a.y) - (y - a.y) * (b.x - a.x);
}

__device__ bool insideTriangle(triangle t, int3 p) {
    float3 a = t.a;
    float3 b = t.b;
    float3 c = t.c;
    
    float w0 = edgeFunction(b, c, p.x, p.y);
    float w1 = edgeFunction(c, a, p.x, p.y);
    float w2 = edgeFunction(a, b, p.x, p.y);
    
    float orientation = w0;
    if (orientation != 0) {
        if (orientation < 0) {
            w0 = -w0;
            w1 = -w1;
            w2 = -w2;
        }
        return w0 >= 0 && w1 >= 0 && w2 >= 0;
    } else {
        return w0 == 0 && w1 == 0 && w2 == 0;
    }
}

__global__ void initBuffer(float *Buffer, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        Buffer[idx] = value;
    }
}

__global__ void initTiles(int width, int height, tile *tiles) {
    int tilesWidth = (width + TILE_SIZE - 1) / TILE_SIZE;
    int tilesHeight = (height + TILE_SIZE - 1) / TILE_SIZE;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < tilesWidth * tilesHeight) {
        tiles[idx].numTriangles = 0;
    }
}

__device__ float3 normalize(float3 v) {
	float length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	if (length > 0.0f) {
		v.x /= length;
		v.y /= length;
		v.z /= length;
	}
	return v;
}

__device__ float dot(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}