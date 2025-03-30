# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -g -I./include  # -I./include includes the header files from the include directory
NVCCFLAGS = -Xcompiler "$(CFLAGS)"  # Pass CFLAGS and include directory to NVCC via -Xcompiler
LDFLAGS = -lm -lcudart -lcuda  # Linker flags, adding -lm for math library and CUDA libraries

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
INCLUDE_DIR = include
DATA_DIR = data
PROFILE_DIR = profile

# File extensions
C_SRC_EXT = .c
CU_SRC_EXT = .cu
OBJ_EXT = .o

# Source files and object files
C_SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
CU_SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES = $(C_SRC_FILES:$(SRC_DIR)/%$(C_SRC_EXT)=$(BUILD_DIR)/%.o) $(CU_SRC_FILES:$(SRC_DIR)/%$(CU_SRC_EXT)=$(BUILD_DIR)/%.o)

# Executable name
EXEC_NAME = mnist

# Final executable
EXEC = $(BIN_DIR)/$(EXEC_NAME)

# Targets
all: clean build run profile

build: $(EXEC)

$(EXEC): $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(OBJ_FILES) -o $@ $(LDFLAGS)

# Rule to compile C source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%$(C_SRC_EXT)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to compile CUDA source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%$(CU_SRC_EXT)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Target for cleaning the build and bin directories
clean:
	rm -rf $(BUILD_DIR)/*.o $(BIN_DIR)/$(EXEC_NAME) $(PROFILE_DIR)/gmon.out $(PROFILE_DIR)/analysis.txt $(PROFILE_DIR)/perf.data

# Target to run the executable
run: $(EXEC)
	$(EXEC)

# Target to profile the executable
profile: $(EXEC)
	@mkdir -p $(PROFILE_DIR)
	nsys profile \
		-o $(PROFILE_DIR)/minimal_profile \
		-f true \
		--trace=cuda,osrt \
		--sample=process-tree \
		$(EXEC)