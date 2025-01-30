#include "libs/cudarender.h"

extern GLuint pbo[2];
extern cudaGraphicsResource* cuda_pbo_resource[2];
extern int width;
extern int height;

void createPBOs(int width, int height) {
    glGenBuffers(2, pbo);
    for (int i = 0; i < 2; i++) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[i]);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);
        cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource[i], pbo[i], cudaGraphicsMapFlagsWriteDiscard);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void updatePBOsize(GLFWwindow *window, int newwidth, int newheight) {
	width = newwidth;
	height = newheight;
	cudaGraphicsUnregisterResource(cuda_pbo_resource[0]);
	cudaGraphicsUnregisterResource(cuda_pbo_resource[1]);
	createPBOs(width, height);
}
