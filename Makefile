CC = nvcc

CFLAGS = -rdc=true -arch=sm_86 -I. -Xcompiler \"/MD\" --cudart=shared
DEBUGFLAGS = -g -G
RELEASEFLAGS = -O3

GLEWDIR = libs/external/glew-2.1.0
GLEWFLAGS = -I$(GLEWDIR)/include -L$(GLEWDIR)/lib/Release/x64

GLFWDIR = libs/external/glfw/glfw-3.3.8.bin.WIN64
GLFWFLAGS = -I$(GLFWDIR)/include -L$(GLFWDIR)/lib-vc2022

ASSIMPDIR = libs/external/assimp
ASSIMPFLAGS = -I$(ASSIMPDIR)/include -L$(ASSIMPDIR)/lib/x64

RELEASETOTALFLAGS = $(CFLAGS) $(RELEASEFLAGS) $(GLFWFLAGS) $(GLEWFLAGS) $(ASSIMPFLAGS)
DEBUGTOTALFLAGS = $(CFLAGS) $(DEBUGFLAGS) $(GLFWFLAGS) $(GLEWFLAGS) $(ASSIMPFLAGS)

LIBS = -lcuda -lcudart -lglew32 -lglfw3 -lopengl32 -lgdi32 -luser32 -lshell32 -lassimp

SRCDIR = src
SRCS = main.cu windowManager.cu renderManager.cu utils.cu newRender.cu renderUtils.cu

HEADERS = libs/cudarender.h

BUILD_DIR = build
DEBUG_DIR = $(BUILD_DIR)/Debug
RELEASE_DIR = $(BUILD_DIR)/Release

TARGET_DEBUG = $(DEBUG_DIR)/CudaRender.exe
TARGET_RELEASE = $(RELEASE_DIR)/CudaRender.exe

DEBUG_OBJS = $(patsubst %, $(DEBUG_DIR)/%, $(SRCS:.cu=.obj))
RELEASE_OBJS = $(patsubst %, $(RELEASE_DIR)/%, $(SRCS:.cu=.obj))

all: debug

debug: CFLAGS += $(DEBUGFLAGS)
debug: $(TARGET_DEBUG)

release: CFLAGS += $(RELEASEFLAGS)
release: $(TARGET_RELEASE)

$(TARGET_DEBUG): $(DEBUG_OBJS)
	mkdir -p $(DEBUG_DIR)
	$(CC) $(DEBUGTOTALFLAGS)  -o $(TARGET_DEBUG) $(DEBUG_OBJS) $(LDFLAGS) $(LIBS)

$(TARGET_RELEASE): $(RELEASE_OBJS)
	mkdir -p $(RELEASE_DIR)
	$(CC) $(RELEASETOTALFLAGS) -o $(TARGET_RELEASE) $(RELEASE_OBJS) $(LDFLAGS) $(LIBS)

$(DEBUG_DIR)/%.obj: $(SRCDIR)/%.cu $(HEADERS)
	mkdir -p $(DEBUG_DIR)
	$(CC) $(DEBUGTOTALFLAGS) -c $< -o $@

$(RELEASE_DIR)/%.obj: $(SRCDIR)/%.cu $(HEADERS)
	mkdir -p $(RELEASE_DIR)
	$(CC) $(RELEASETOTALFLAGS) -c $< -o $@

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
