/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
package com.nvidia.grcuda.functions;

import com.nvidia.grcuda.GrCUDALanguage;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.array.AbstractArray;
import com.nvidia.grcuda.array.MultiDimDeviceArrayView;
import com.nvidia.grcuda.gpu.computation.ArrayCopyFunctionExecutionDefault;
import com.nvidia.grcuda.gpu.computation.ArrayCopyFunctionExecutionInitializer;
import com.nvidia.grcuda.gpu.computation.ArrayCopyFunctionExecutionMemcpy;
import com.nvidia.grcuda.gpu.computation.DeviceArrayCopyException;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;

@ExportLibrary(InteropLibrary.class)
public class DeviceArrayCopyFunction implements TruffleObject {

    public enum CopyDirection {
        FROM_POINTER,
        TO_POINTER
    }

    private final AbstractArray array;
    private final CopyDirection direction;

    public DeviceArrayCopyFunction(AbstractArray array, CopyDirection direction) {
        this.array = array;
        this.direction = direction;
    }

    @ExportMessage
    @SuppressWarnings("static-method")
    boolean isExecutable() {
        return true;
    }

    private static long extractPointer(Object valueObj, InteropLibrary access) throws UnsupportedMessageException {
        if (access.isPointer(valueObj)) {
            return access.asPointer(valueObj);
        } else {
            return access.asLong(valueObj);
        }
    }

    private static int extractNumber(Object valueObj, InteropLibrary access) throws UnsupportedTypeException {
        try {
            return access.asInt(valueObj);
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw UnsupportedTypeException.create(new Object[]{valueObj}, "integer expected for numElements");
        }
    }

    @ExportMessage
    Object execute(Object[] arguments,
                    @CachedLibrary(limit = "3") InteropLibrary pointerAccess,
                    @CachedLibrary(limit = "3") InteropLibrary numElementsAccess) throws UnsupportedTypeException, ArityException, IndexOutOfBoundsException, DeviceArrayCopyException {
        // Obtain the number of elements to copy;
        long numElements;
        if (arguments.length == 1) {
            numElements = array.getArraySize();
        } else if (arguments.length == 2) {
            numElements = extractNumber(arguments[1], numElementsAccess);
        } else {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(1, arguments.length);
        }
        // Obtain what kind of copy (pointer or array) should be executed.
        // By default, see if we can use the fast CUDA memcpy: we cannot use it if the source and target arrays
        // are stored with incompatible memory layouts.
        ArrayCopyFunctionExecutionInitializer dependencyInitializer = new ArrayCopyFunctionExecutionInitializer(array, arguments[0], direction);
        if (canUseMemcpy(arguments[0])) {
            try {
                // Try using the native pointer implementation;
                long pointer = extractPointer(arguments[0], pointerAccess);
                // Fast memcpy path;
                return new ArrayCopyFunctionExecutionMemcpy(array, direction, numElements, pointer, dependencyInitializer).schedule();
            } catch (UnsupportedMessageException e) {
                GrCUDALanguage.LOGGER.info("cannot extract a native pointer; falling back to slow copy");
            }
        }
        // Perform the slow memcpy, if no other option is available;
        return slowCopyPath(pointerAccess, arguments[0], numElements, dependencyInitializer);
    }

    private Object slowCopyPath(@CachedLibrary(limit = "3") InteropLibrary pointerAccess, Object otherArray,
                                long numElements, ArrayCopyFunctionExecutionInitializer dependencyInitializer) throws UnsupportedTypeException {
        // Slow array copy, suitable for generic arrays or incompatible memory layouts;
        if (pointerAccess.hasArrayElements(otherArray)) {
            return new ArrayCopyFunctionExecutionDefault(array, direction, numElements, pointerAccess, otherArray, dependencyInitializer).schedule();
        } else {
            // The target object is not an array;
            CompilerDirectives.transferToInterpreter();
            throw UnsupportedTypeException.create(new Object[]{otherArray}, "array or pointer expected for " + (direction.equals(CopyDirection.FROM_POINTER) ? "fromPointer" : "toPointer"));
        }
    }

    /**
     * We can use fast memcpy only if both arrays are stored with the same memory layout. This also
     * holds true if either the other array is not an Abstract array, but some other kind of native
     * memory; in this second case, the user has the responsibility of providing meaningful data to
     * copy.
     * 
     * @param other the other object involved in the copy
     * @return if we can use the fast CUDA memcpy, under the assumption that "other" contains an accessible pointer
     */
    public boolean canUseMemcpy(Object other) {
        if (other instanceof AbstractArray) {
            boolean coherentMemoryLayout = array.isColumnMajorFormat() == ((AbstractArray) other).isColumnMajorFormat();
            if (!coherentMemoryLayout) {
                GrCUDALanguage.LOGGER.warning("both source and destination arrays should be row-major or column-major; falling back to slow copy");
                return false;
            }
            if ((array instanceof MultiDimDeviceArrayView && array.isColumnMajorFormat()) ||
                            (other instanceof MultiDimDeviceArrayView && ((MultiDimDeviceArrayView) other).isColumnMajorFormat())) {
                GrCUDALanguage.LOGGER.warning("fast copy from/to column-major array views is not supported; falling back to slow copy");
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        return "DeviceArrayCopyFunction(deviceArray=" + array + ", direction=" + direction.name() + ")";
    }
}
