FLAGS = -std=c++11 -O3 -arch=sm_50
BIN_FOLDER=bin
FILES=main.cu benchmark.cu b1.cu b5.cu b6.cu b7.cu b8.cu b10.cu

.PHONY: full all b1 b5 b6 b7 b8 b10

full:
	nvcc $(FILES) $(FLAGS) -o $(BIN_FOLDER)/b;

all: \
    full \
	b1 \
	b5 \
	b6 \
	b7 \
	b8 \
	b10

b1: b1*
	nvcc old/b1_default.cu $(FLAGS) -o $(BIN_FOLDER)/b1_default;
	nvcc old/b1_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b1_sync

b5: b5*
	nvcc old/b5_default.cu $(FLAGS) -o $(BIN_FOLDER)/b5_default;
	nvcc old/b5_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b5_sync

b6: b6*
	nvcc old/b6_default.cu $(FLAGS) -o $(BIN_FOLDER)/b6_default;
	nvcc old/b6_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b6_sync

b7: b7*
	nvcc old/b7_default.cu $(FLAGS) -o $(BIN_FOLDER)/b7_default;
	nvcc old/b7_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b7_sync

b8: b8*
	nvcc old/b8_default.cu $(FLAGS) -o $(BIN_FOLDER)/b8_default;
	nvcc old/b8_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b8_sync

b10: b10*
	nvcc old/b10_default.cu $(FLAGS) -o $(BIN_FOLDER)/b10_default;
	nvcc old/b10_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b10_sync
