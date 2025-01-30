#include "libs/cudarender.h"

void createPBOs() {
    glGenBuffers(2, pbo);
    for (int i = 0; i < 2; i++) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[i]);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 4, nullptr, GL_DYNAMIC_DRAW);
        cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource[i], pbo[i], cudaGraphicsMapFlagsWriteDiscard);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}
