# Use NVCC;
CXX=nvcc
FLAGS = -std=c++11 -O3 -arch=sm_70

# Use Clang;
CXX=$(CLANG_DIR)/clang++
FLAGS = --cuda-gpu-arch=sm_70 -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -std=c++11 -O3

BIN_FOLDER=bin
FILES=main.cu benchmark.cu b1.cu b5.cu b6.cu b7.cu b8.cu b10.cu

.PHONY: full all b1 b5 b6 b7 b8 b10

full:
	$(CXX) $(FILES) $(FLAGS) -o $(BIN_FOLDER)/b;

all: \
    full \
	b1 \
	b5 \
	b6 \
	b7 \
	b8 \
	b10

b1: b1*
	$(CXX) old/b1_default.cu $(FLAGS) -o $(BIN_FOLDER)/b1_default;
	$(CXX) old/b1_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b1_sync

b5: b5*
	$(CXX) old/b5_default.cu $(FLAGS) -o $(BIN_FOLDER)/b5_default;
	$(CXX) old/b5_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b5_sync

b6: b6*
	$(CXX) old/b6_default.cu $(FLAGS) -o $(BIN_FOLDER)/b6_default;
	$(CXX) old/b6_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b6_sync

b7: b7*
	$(CXX) old/b7_default.cu $(FLAGS) -o $(BIN_FOLDER)/b7_default;
	$(CXX) old/b7_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b7_sync

b8: b8*
	$(CXX) old/b8_default.cu $(FLAGS) -o $(BIN_FOLDER)/b8_default;
	$(CXX) old/b8_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b8_sync

b10: b10*
	$(CXX) old/b10_default.cu $(FLAGS) -o $(BIN_FOLDER)/b10_default;
	$(CXX) old/b10_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b10_sync
