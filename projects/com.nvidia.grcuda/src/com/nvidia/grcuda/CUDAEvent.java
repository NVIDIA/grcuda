/*
 * Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
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
package com.nvidia.grcuda;

import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

import java.util.Objects;

@ExportLibrary(InteropLibrary.class)
public class CUDAEvent extends GPUPointer {

    private final long eventNumber;
    /**
     * Keep track of whether this event has been destroyed by {@link com.nvidia.grcuda.runtime.CUDARuntime#cudaEventDestroy}
     */
    private boolean isAlive = true;

    public CUDAEvent(long rawPointer, long eventNumber) {
        super(rawPointer);
        this.eventNumber = eventNumber;
    }

    public long getEventNumber() {
        return eventNumber;
    }

    public boolean isDefaultStream() { return false; }

    /**
     * Keep track of whether this event has been destroyed by {@link com.nvidia.grcuda.runtime.CUDARuntime#cudaEventDestroy}
     */
    public boolean isAlive() {
        return isAlive;
    }

    /**
     * Set the event as destroyed by the CUDA runtime;
     */
    public void setDead() {
        isAlive = false;
    }

    @Override
    public String toString() {
        return "CUDAEvent(eventNumber=" + this.eventNumber + "; address=0x" + Long.toHexString(this.getRawPointer()) + ")";
    }

    @ExportMessage
    public Object toDisplayString(boolean allowSideEffect) {
        return this.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CUDAEvent that = (CUDAEvent) o;
        return (eventNumber == that.eventNumber && this.getRawPointer() == that.getRawPointer());
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), eventNumber);
    }
}
