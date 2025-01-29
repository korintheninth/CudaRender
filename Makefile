CC = nvcc

CFLAGS = -arch=sm_86 -I.
DEBUGFLAGS = -g -G

LIBS = -lcuda -lcudart

SRCDIR = src
SRCS = main.cu

HEADERS = libs/cudarender.h

BUILD_DIR = build
DEBUG_DIR = $(BUILD_DIR)/Debug
RELEASE_DIR = $(BUILD_DIR)/Release

TARGET_DEBUG = $(DEBUG_DIR)/CudaRender.exe
TARGET_RELEASE = $(RELEASE_DIR)/CudaRender.exe

DEBUG_OBJS = $(DEBUG_DIR)/$(SRCS:.cu=.obj)
RELEASE_OBJS = $(RELEASE_DIR)/$(SRCS:.cu=.obj)

all: debug

debug: CFLAGS += $(DEBUGFLAGS)
debug: $(TARGET_DEBUG)

release: CFLAGS += $(RELEASEFLAGS)
release: $(TARGET_RELEASE)

$(TARGET_DEBUG): $(DEBUG_OBJS)
	mkdir -p $(DEBUG_DIR)
	$(CC) $(CFLAGS) $(CUDAFLAGS) -o $(TARGET_DEBUG) $(DEBUG_OBJS) $(LDFLAGS) $(LIBS)

$(TARGET_RELEASE): $(RELEASE_OBJS)
	mkdir -p $(RELEASE_DIR)
	$(CC) $(CFLAGS) $(CUDAFLAGS) -o $(TARGET_RELEASE) $(RELEASE_OBJS) $(LDFLAGS) $(LIBS)

$(DEBUG_DIR)/%.obj: $(SRCDIR)/%.cu $(HEADERS)
	mkdir -p $(DEBUG_DIR)
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c $< -o $@

$(RELEASE_DIR)/%.obj: $(SRCDIR)%.cu $(HEADERS)
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
