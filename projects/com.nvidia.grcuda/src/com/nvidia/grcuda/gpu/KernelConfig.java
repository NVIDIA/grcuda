/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.nvidia.grcuda.gpu;

import java.util.Arrays;
import java.util.Objects;

import com.oracle.truffle.api.CompilerAsserts;

public final class KernelConfig {
    private final Dim3 gridSize;
    private final Dim3 blockSize;
    private final int dynamicSharedMemoryBytes;

    public KernelConfig(int numBlocks, int numThreadsPerBlock) {
        gridSize = new Dim3(numBlocks);
        blockSize = new Dim3(numThreadsPerBlock);
        dynamicSharedMemoryBytes = 0;
    }

    public KernelConfig(Dim3 gridSize, Dim3 blockSize) {
        this.gridSize = gridSize;
        this.blockSize = blockSize;
        this.dynamicSharedMemoryBytes = 0;
    }

    public KernelConfig(Dim3 gridSize, Dim3 blockSize, int sharedMemoryBytes) {
        this.gridSize = gridSize;
        this.blockSize = blockSize;
        this.dynamicSharedMemoryBytes = sharedMemoryBytes;
    }

    @Override
    public String toString() {
        return "KernelConfig(gridSize=" + gridSize + ", blockSize=" + blockSize +
                        ", sharedMemoryBytes=" + dynamicSharedMemoryBytes + ", stream=" + getStream() + ')';
    }

    public Dim3 getGridSize() {
        return gridSize;
    }

    public Dim3 getBlockSize() {
        return blockSize;
    }

    public int getDynamicSharedMemoryBytes() {
        return dynamicSharedMemoryBytes;
    }

    @SuppressWarnings("static-method")
    public int getStream() {
        return 0; // default stream
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        KernelConfig that = (KernelConfig) o;
        return dynamicSharedMemoryBytes == that.dynamicSharedMemoryBytes &&
                        getStream() == that.getStream() &&
                        gridSize.equals(that.gridSize) &&
                        blockSize.equals(that.blockSize);
    }

    @Override
    public int hashCode() {
        return Objects.hash(gridSize, blockSize, dynamicSharedMemoryBytes, getStream());
    }
}

final class Dim3 {
    private final int[] dims = new int[3];

    Dim3(int x) {
        dims[0] = x;
        dims[1] = 1;
        dims[2] = 1;
    }

    Dim3(int x, int y) {
        dims[0] = x;
        dims[1] = y;
        dims[2] = 1;
    }

    Dim3(int x, int y, int z) {
        dims[0] = x;
        dims[1] = y;
        dims[2] = z;
    }

    public int getX() {
        return dims[0];
    }

    public int getY() {
        return dims[1];
    }

    public int getZ() {
        return dims[2];
    }

    @Override
    public String toString() {
        CompilerAsserts.neverPartOfCompilation();
        return "(" + dims[0] + ", " + dims[1] + ", " + dims[2] + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Dim3 dim3 = (Dim3) o;
        return Arrays.equals(dims, dim3.dims);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(dims);
    }
}
