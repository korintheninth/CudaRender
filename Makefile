CC = nvcc

CFLAGS = -arch=sm_86
DEBUGFLAGS = -g -G
RELEASEFLAGS =

CUDA_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
CUDAFLAGS = -I$(CUDA_PATH)/include
GLFWFLAGS = -I$(GLFW_PATH)/glfw/glfw-3.3.8.bin.WIN64/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -L$(GLFW_PATH)/glfw/glfw-3.3.8.bin.WIN64/lib-vc2022
LIBS = -lcuda -lcudart -lglfw3

SRCDIR = srcs
SRCS = main.cu

HEADERS = libs/cudarender.h libs/external/includes/GLFW/glfw3.h
BUILD_DIR = build
DEBUG_DIR = $(BUILD_DIR)/Debug

RELEASE_DIR = $(BUILD_DIR)/Release
TARGET_DEBUG = $(DEBUG_DIR)/CudaRender.exe
TARGET_RELEASE = $(RELEASE_DIR)/CudaRender.exe

DEBUG_OBJS = $(addprefix $(DEBUG_DIR)/, $(SRCS:.cu=.obj))
RELEASE_OBJS = $(addprefix $(RELEASE_DIR)/, $(SRCS:.cu=.obj))

all: debug

debug: CFLAGS += $(DEBUGFLAGS)
debug: $(TARGET_DEBUG)

release: CFLAGS += $(RELEASEFLAGS)
release: $(TARGET_RELEASE)

$(TARGET_DEBUG): $(DEBUG_OBJS)
	mkdir -p $(DEBUG_DIR)
	$(CC) $(CFLAGS) $(CUDAFLAGS) $(DEBUG_OBJS) -o $(TARGET_DEBUG) $(LDFLAGS) $(LIBS)

$(TARGET_RELEASE): $(RELEASE_OBJS)
	mkdir -p $(RELEASE_DIR)
	$(CC) $(CFLAGS) $(CUDAFLAGS) $(RELEASE_OBJS) -o $(TARGET_RELEASE) $(LDFLAGS) $(LIBS)

$(DEBUG_DIR)/%.obj: $(SRCDIR)/%.cu $(HEADERS)
	mkdir -p $(DEBUG_DIR)
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c $< -o $@

$(RELEASE_DIR)/%.obj: $(SRCDIR)/%.cu $(HEADERS)
	mkdir -p $(RELEASE_DIR)
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c $< -o $@

clean:
	rm -f *.pdb
	rm -f $(DEBUG_DIR)/*.pdb
	rm -f $(RELEASE_DIR)/*.pdb
	rm -f $(DEBUG_OBJS) $(RELEASE_OBJS)

fclean: clean
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET_DEBUG) $(TARGET_RELEASE)

re: fclean all

.PHONY: all debug release clean fclean re