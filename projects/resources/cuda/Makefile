# Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NECSTLab nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#  * Neither the name of Politecnico di Milano nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
