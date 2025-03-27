# Compiler settings
CC = gcc
CFLAGS = -Wall -g -I./include  # -I./include includes the header files from the include directory
LDFLAGS = -lm  # Linker flags, adding -lm for math library

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
INCLUDE_DIR = include
DATA_DIR = data
PROFILE_DIR = profile

# File extensions
SRC_EXT = .c
OBJ_EXT = .o

# Source files and object files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(SRC_FILES:$(SRC_DIR)/%$(SRC_EXT)=$(BUILD_DIR)/%.o)

# Executable name
EXEC_NAME = mnist

# Final executable
EXEC = $(BIN_DIR)/$(EXEC_NAME)

# Targets
all: clean build run profile

build: $(EXEC)

$(EXEC): $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	$(CC) $(OBJ_FILES) -o $@ $(LDFLAGS)

# Rule to compile source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%$(SRC_EXT)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Target for cleaning the build and bin directories
clean:
	rm -rf $(BUILD_DIR)/*.o $(BIN_DIR)/nn $(PROFILE_DIR)/gmon.out $(PROFILE_DIR)/analysis.txt $(PROFILE_DIR)/perf.data

# Target to run the executable
run: $(EXEC)
	$(EXEC)

# Target for profiling with gprof and generating a graph
gprof: $(EXEC)
	@mkdir -p $(PROFILE_DIR) $(PROFILE_DIR)/graphs
	$(CC) $(CFLAGS) -pg $(SRC_FILES) -o $(EXEC) $(LDFLAGS)
	$(EXEC)
	# Move this gmon.out to the profile dir
	mv gmon.out $(PROFILE_DIR)/
	gprof $(EXEC) $(PROFILE_DIR)/gmon.out > $(PROFILE_DIR)/analysis.txt
	gprof2dot -f prof $(PROFILE_DIR)/analysis.txt | dot -Tpng -o $(PROFILE_DIR)/graphs/gprof_analysis.png

# Target for performance profiling using perf
perf: $(EXEC)
	@mkdir -p $(PROFILE_DIR)
	# Record perf data with call graph
	perf record -g -o $(PROFILE_DIR)/perf.data -- $(EXEC)
	# Generate a perf report
	perf report -i $(PROFILE_DIR)/perf.data > $(PROFILE_DIR)/perf_report.txt
	# Generate a perf graph
	perf script -i $(PROFILE_DIR)/perf.data | c++filt | gprof2dot -f perf | dot -Tpng -o $(PROFILE_DIR)/graphs/perf_graph.png
	# Clean up because idk where this gmon.out keeps coming from
	@rm -f gmon.out

# Target to run both gprof and perf profiling
profile: gprof perf
	@echo "Profiling complete. Gprof analysis: $(PROFILE_DIR)/analysis.txt, Perf report: $(PROFILE_DIR)/perf_report.txt"
