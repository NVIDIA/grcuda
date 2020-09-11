FLAGS = -std=c++11 -O3 -arch=sm_50
BIN_FOLDER=bin

.PHONY: all b1 b5 b6 b7 b8 b10
all: \
	b1 \
	b5 \
	b6 \
	b7 \
	b8 \
	b10

b1: b1*
	nvcc b1_default.cu $(FLAGS) -o $(BIN_FOLDER)/b1_default;
	nvcc b1_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b1_sync

b5: b5*
	nvcc b5_default.cu $(FLAGS) -o $(BIN_FOLDER)/b5_default;
	nvcc b5_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b5_sync

b6: b6*
	nvcc b6_default.cu $(FLAGS) -o $(BIN_FOLDER)/b6_default;
	nvcc b6_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b6_sync

b7: b7*
	nvcc b7_default.cu $(FLAGS) -o $(BIN_FOLDER)/b7_default;
	nvcc b7_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b7_sync

b8: b8*
	nvcc b8_default.cu $(FLAGS) -o $(BIN_FOLDER)/b8_default;
	nvcc b8_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b8_sync

b10: b10*
	nvcc b10_default.cu $(FLAGS) -o $(BIN_FOLDER)/b10_default;
	nvcc b10_sync.cu $(FLAGS) -o $(BIN_FOLDER)/b10_sync
