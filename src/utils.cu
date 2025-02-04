#include "libs/cudarender.h"

extern GLuint pbo[2];
extern cudaGraphicsResource* cuda_pbo_resource[2];
extern int width;
extern int height;
extern float *depthBuffer;

void createPBOs(int width, int height) {
    glGenBuffers(2, pbo);
    for (int i = 0; i < 2; i++) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[i]);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);
        cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource[i], pbo[i], cudaGraphicsMapFlagsWriteDiscard);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void updateBuffersize(GLFWwindow *window, int newwidth, int newheight) {
    if (newwidth == 0 || newheight == 0)
        return;
    width = newwidth;
    height = newheight;
    cudaFree(depthBuffer);
    cudaMalloc(&depthBuffer, width * height * sizeof(float));

    for (int i = 0; i < 2; i++) {
        cudaError_t err = cudaGraphicsUnregisterResource(cuda_pbo_resource[i]);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error (Unregistering PBO): " << cudaGetErrorString(err) << std::endl;
        }
    }
    createPBOs(width, height);
    updateContent(width, height, window);
}

bool LoadModel(const std::string& fileDir, std::vector<int>& indices, std::vector<float3>& vertices, int* numIndices, int* numVertices) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(fileDir, aiProcess_Triangulate);

    if (!scene || !scene->mRootNode || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) {
        std::cerr << "Assimp Error: " << importer.GetErrorString() << std::endl;
        return false;
    }

    const aiMesh* mesh = scene->mMeshes[0];

    vertices.resize(mesh->mNumVertices);
    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        vertices[i] = { mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z };
    }

    indices.resize(mesh->mNumFaces * 3);
    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        const aiFace& face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
            indices[i * 3 + j] = face.mIndices[j];
        }
    }

    *numVertices = static_cast<int>(vertices.size());
    *numIndices = static_cast<int>(indices.size());

    return true;
}

