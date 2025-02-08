#include "libs/cudarender.h"


__global__ void castTriangles(int width, int height, float3 *vertices, int *indices, int numTriangles, triangle *triangles, controls camera) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx >= numTriangles)
        return;

    float3 position = {0.0f, 0.0f, 10.0f};

    float3 a = modelToWorld(vertices[indices[globalIdx * 3]], position, camera.rotation);
    float3 b = modelToWorld(vertices[indices[globalIdx * 3 + 1]], position, camera.rotation);
    float3 c = modelToWorld(vertices[indices[globalIdx * 3 + 2]], position, camera.rotation);
    float3 normal = calculateSurfaceNormal(a, b, c);

    float a4[4] = {a.x, a.y, a.z, 1};
    float b4[4] = {b.x, b.y, b.z, 1};
    float c4[4] = {c.x, c.y, c.z, 1};

    float a4p[4], b4p[4], c4p[4];
    float projection[4][4];
    //float rotation[4][4];
    //float result[4][4];
    projectionMatrix(100.0f, (float)width / (float)height, 0.1f, 100.0f, projection);
    //rotationMatrix(camera.rotation, rotation);
    //matMult(rotation, projection, result);
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

	triangle t;
	t.a = a;
	t.b = b;
	t.c = c;
	t.normal = normal;
    t.xmin = max(0, (int)floorf(fminf(a.x, fminf(b.x, c.x))));
    t.xmax = min(width - 1, (int)ceilf(fmaxf(a.x, fmaxf(b.x, c.x))));
    t.ymin = max(0, (int)floorf(fminf(a.y, fminf(b.y, c.y))));
    t.ymax = min(height - 1, (int)ceilf(fmaxf(a.y, fmaxf(b.y, c.y))));

	triangles[globalIdx] = t;
}

__global__ void assignToTiles(int width, int height, int numTriangles, triangle *triangles, tile *tiles) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx >= numTriangles)
        return;
        
    triangle t = triangles[globalIdx];
    
    int startTileX = t.xmin / TILE_SIZE;
    int endTileX = (t.xmax + TILE_SIZE - 1) / TILE_SIZE;
    int startTileY = t.ymin / TILE_SIZE;
    int endTileY = (t.ymax + TILE_SIZE - 1) / TILE_SIZE;
    
    int tilesWidth = (width + TILE_SIZE - 1) / TILE_SIZE;
    int tilesHeight = (height + TILE_SIZE - 1) / TILE_SIZE;
    
    startTileX = max(0, startTileX);
    endTileX = min(tilesWidth - 1, endTileX);
    startTileY = max(0, startTileY);
    endTileY = min(tilesHeight - 1, endTileY);
    
    for (int tileY = startTileY; tileY <= endTileY; tileY++) {
        for (int tileX = startTileX; tileX <= endTileX; tileX++) {
            tile* currentTile = &tiles[tileY * tilesWidth + tileX];
            int idx = atomicAdd(&currentTile->numTriangles, 1);
            if (idx < MAX_TRIANGLES_PER_TILE) {
                currentTile->triangles[idx] = globalIdx;
            }
            else {
                printf("Warning: Tile (%d,%d) exceeded MAX_TRIANGLES_PER_TILE: %d\n", tileX, tileY, idx);
            }
        }
    }
}

__global__ void renderTile(int width, int height, tile *tiles, uchar4 *buffer, float *depthBuffer, triangle *triangles) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	tile *currentTile = &tiles[(y / TILE_SIZE) * ((width + TILE_SIZE - 1) / TILE_SIZE) + (x / TILE_SIZE)];
	if (currentTile->numTriangles == 0)
		return;

	for (int i = 0; i < currentTile->numTriangles; i++) {
		triangle t = triangles[currentTile->triangles[i]];
		if (!insideTriangle(t, make_int3(x, y, 0)))
			continue;

		if (x >= t.xmin && x <= t.xmax && y >= t.ymin && y <= t.ymax) {
			float depth = pointDepth(t.a, t.b, t.c, x, y);
			if (depth < depthBuffer[y * width + x]) {
            	atomicMinFloat(&depthBuffer[y * width + x], depth);
				float3 color = {0.0f, 0.0f, 1.0f};
				float intensity = fmaxf(0.0f, dot(normalize(t.normal), normalize({0.0f, 0.0f, 1.0f})));
				color.x *= intensity;
				color.y *= intensity;
				color.z *= intensity;
				buffer[y * width + x] = make_uchar4(255 * color.x, 255 * color.y, 255 * color.z, 255);
			}
		}
	}
}

void newRender(uchar4 *d_buffer, int width, int height, triangleData data, triangle *d_triangles, tile *d_tiles, controls camera) {
    int numPixels = width * height;

    int tilesWidth = (width + TILE_SIZE - 1) / TILE_SIZE;
    int tilesHeight = (height + TILE_SIZE - 1) / TILE_SIZE;
    int numTiles = tilesWidth * tilesHeight;
    int tileThreads = min(256, numTiles);
    int tileBlocks = max(1, (numTiles + 255) / 256);

    dim3 bufferBlockSize(16, 16);
    dim3 bufferGridSize((width + 15) / 16, (height + 15) / 16);

    clearBuffer<<<bufferGridSize, bufferBlockSize>>>(d_buffer, width, height);

    initTiles<<<tileBlocks, tileThreads>>>(width, height, d_tiles);
    initBuffer<<<(numPixels + 255)/256, 256>>>(data.depthBuffer, numPixels, 100.0f);
    cudaDeviceSynchronize();
    
	castTriangles<<<max(1, (data.numTriangles + 255) / 256), min(256, data.numTriangles)>>>(width, height, data.vertices, data.indices, data.numTriangles, d_triangles, camera);
    cudaDeviceSynchronize();
    
	assignToTiles<<<max(1, (data.numTriangles + 255) / 256), min(256, data.numTriangles)>>>(width, height, data.numTriangles, d_triangles, d_tiles);
    cudaDeviceSynchronize();
	
	dim3 renderBlockSize(TILE_SIZE, TILE_SIZE);
	dim3 renderGridSize((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
	
	renderTile<<<renderGridSize, renderBlockSize>>>(width, height, d_tiles, d_buffer, data.depthBuffer, d_triangles);
	cudaDeviceSynchronize();
	
	cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}