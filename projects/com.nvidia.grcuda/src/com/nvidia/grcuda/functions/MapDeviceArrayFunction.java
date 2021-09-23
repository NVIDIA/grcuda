/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
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
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
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
package com.nvidia.grcuda.functions;

import java.util.concurrent.ConcurrentHashMap;

import com.nvidia.grcuda.runtime.array.DeviceArray;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAException;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.GrCUDALanguage;
import com.nvidia.grcuda.NoneValue;
import com.nvidia.grcuda.TypeException;
import com.nvidia.grcuda.runtime.executioncontext.AbstractGrCUDAExecutionContext;
import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.Truffle;
import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.CachedContext;
import com.oracle.truffle.api.dsl.GenerateUncached;
import com.oracle.truffle.api.dsl.Specialization;
import com.oracle.truffle.api.frame.FrameDescriptor;
import com.oracle.truffle.api.frame.FrameSlot;
import com.oracle.truffle.api.frame.FrameSlotKind;
import com.oracle.truffle.api.frame.FrameSlotTypeException;
import com.oracle.truffle.api.frame.VirtualFrame;
import com.oracle.truffle.api.interop.ArityException;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.library.DynamicDispatchLibrary;
import com.oracle.truffle.api.library.ExportLibrary;
import com.oracle.truffle.api.library.ExportMessage;
import com.oracle.truffle.api.nodes.LoopNode;
import com.oracle.truffle.api.nodes.Node;
import com.oracle.truffle.api.nodes.RepeatingNode;
import com.oracle.truffle.api.nodes.RootNode;
import com.oracle.truffle.api.profiles.ConditionProfile;
import com.oracle.truffle.api.profiles.ValueProfile;

/**
 * This node is conceptually a simple memcpy operation, but it can take arbitrary Truffle objects as
 * input and uses {@link InteropLibrary} to access them. The target is a {@link DeviceArray}. The
 * actual work is done behind a {@link CallTarget} so that it can be compiled separately.
 */
@GenerateUncached
abstract class MapArrayNode extends Node {

    abstract Object execute(Object source, Type elementType, AbstractGrCUDAExecutionContext grCUDAExecutionContext);

    private static final FrameDescriptor DESCRIPTOR = new FrameDescriptor();
    private static final FrameSlot SIZE_SLOT = DESCRIPTOR.addFrameSlot("size", FrameSlotKind.Long);
    private static final FrameSlot INDEX_SLOT = DESCRIPTOR.addFrameSlot("index", FrameSlotKind.Long);
    private static final FrameSlot SOURCE_SLOT = DESCRIPTOR.addFrameSlot("source", FrameSlotKind.Object);
    private static final FrameSlot RESULT_SLOT = DESCRIPTOR.addFrameSlot("result", FrameSlotKind.Object);

    private static final class RepeatingLoopNode extends Node implements RepeatingNode {

        @Child private InteropLibrary interop;
        @Child private InteropLibrary elementInterop = InteropLibrary.getFactory().createDispatched(3);

        private final ValueProfile elementTypeProfile = ValueProfile.createIdentityProfile();
        private final ConditionProfile loopProfile = ConditionProfile.createCountingProfile();

        RepeatingLoopNode(InteropLibrary interop) {
            this.interop = interop;
        }

        public boolean executeRepeating(VirtualFrame frame) {
            try {
                long size = frame.getLong(SIZE_SLOT);
                long index = frame.getLong(INDEX_SLOT);
                Object source = frame.getObject(SOURCE_SLOT);
                DeviceArray result = (DeviceArray) frame.getObject(RESULT_SLOT);

                if (loopProfile.profile(index >= size)) {
                    return false;
                }
                Object element;
                try {
                    element = interop.readArrayElement(source, index);
                } catch (UnsupportedMessageException | InvalidArrayIndexException e) {
                    CompilerDirectives.transferToInterpreter();
                    throw new GrCUDAException("cannot read array element " + index + ": " + e.getMessage());
                }
                try {
                    result.writeArrayElement(index, element, elementInterop, elementTypeProfile);
                } catch (UnsupportedTypeException | InvalidArrayIndexException e) {
                    CompilerDirectives.transferToInterpreter();
                    throw new GrCUDAException("cannot coerce array element at index " + index + " to " + result.getElementType() + ": " + e.getMessage());
                }
                frame.setLong(INDEX_SLOT, index + 1);
                return true;
            } catch (FrameSlotTypeException e1) {
                CompilerDirectives.transferToInterpreter();
                throw new GrCUDAException(e1.getMessage());
            }
        }
    }

    private static final class LoopRootNode extends RootNode {

        private final Class<?> clazz;

        @Child private LoopNode loop;

        protected LoopRootNode(Object source, Class<?> clazz) {
            super(GrCUDALanguage.getCurrentLanguage(), DESCRIPTOR);
            this.clazz = clazz;
            this.loop = Truffle.getRuntime().createLoopNode(new RepeatingLoopNode(InteropLibrary.getFactory().create(source)));
        }

        @Override
        public String toString() {
            return clazz == null ? super.toString() : "grcuda map for " + clazz.getSimpleName();
        }

        @Override
        public Object execute(VirtualFrame frame) {
            frame.setLong(SIZE_SLOT, (long) frame.getArguments()[0]);
            frame.setLong(INDEX_SLOT, 0);
            frame.setObject(SOURCE_SLOT, frame.getArguments()[1]);
            frame.setObject(RESULT_SLOT, frame.getArguments()[2]);
            loop.execute(frame);
            return NoneValue.get();
        }
    }

    protected CallTarget createLoop(Object source) {
        return Truffle.getRuntime().createCallTarget(new LoopRootNode(source, null));
    }

    protected CallTarget createUncachedLoop(Object source, GrCUDAContext context) {
        ConcurrentHashMap<Class<?>, CallTarget> uncachedCallTargets = context.getMapCallTargets();
        DynamicDispatchLibrary dispatch = DynamicDispatchLibrary.getFactory().getUncached(source);
        Class<?> clazz = dispatch.dispatch(source);
        if (clazz == null) {
            clazz = source.getClass();
        }
        return uncachedCallTargets.computeIfAbsent(clazz, c -> Truffle.getRuntime().createCallTarget(new LoopRootNode(source, c)));
    }

    @Specialization(limit = "3")
    Object doMap(Object source, Type elementType, AbstractGrCUDAExecutionContext grCUDAExecutionContext,
                    @CachedLibrary("source") InteropLibrary interop,
                    @CachedContext(GrCUDALanguage.class) @SuppressWarnings("unused") GrCUDAContext context,
                    @Cached(value = "createLoop(source)", uncached = "createUncachedLoop(source, context)") CallTarget loop) {

        if (source instanceof DeviceArray && ((DeviceArray) source).getElementType() == elementType) {
            return source;
        }

        if (!interop.hasArrayElements(source)) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException("cannot map from non-array to DeviceArray");
        }

        long size;
        try {
            size = interop.getArraySize(source);
        } catch (UnsupportedMessageException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAException("cannot read array size");
        }
        DeviceArray result = new DeviceArray(grCUDAExecutionContext, size, elementType);
        loop.call(size, source, result);
        return result;
    }
}

/**
 * This function can be called with either two arguments (type and source) and will create a new
 * {@link DeviceArray} with data from the given source, or one argument (type), which will return a
 * version of the function that is curried with the given type.
 */
@ExportLibrary(InteropLibrary.class)
public final class MapDeviceArrayFunction extends Function {

    private final AbstractGrCUDAExecutionContext grCUDAExecutionContext;

    public MapDeviceArrayFunction(AbstractGrCUDAExecutionContext grCUDAExecutionContext) {
        super("MapDeviceArray");
        this.grCUDAExecutionContext = grCUDAExecutionContext;
    }

    @ExportMessage
    public Object execute(Object[] arguments,
                    @CachedLibrary(limit = "2") InteropLibrary stringInterop,
                    @Cached("createIdentityProfile()") ValueProfile elementTypeStringProfile,
                    @Cached("createIdentityProfile()") ValueProfile elementTypeProfile,
                    @Cached MapArrayNode mapNode) throws ArityException, UnsupportedTypeException {
        if (arguments.length < 1) {
            CompilerDirectives.transferToInterpreter();
            throw ArityException.create(1, arguments.length);
        }
        String typeName;
        try {
            typeName = elementTypeStringProfile.profile(stringInterop.asString(arguments[0]));
        } catch (UnsupportedMessageException e1) {
            throw UnsupportedTypeException.create(arguments, "first argument of MapDeviceArray must be string (type name)");
        }
        Type elementType;
        try {
            elementType = elementTypeProfile.profile(Type.fromGrCUDATypeString(typeName));
        } catch (TypeException e) {
            CompilerDirectives.transferToInterpreter();
            throw new GrCUDAInternalException(e.getMessage());
        }
        if (arguments.length == 1) {
            return new TypedMapDeviceArrayFunction(grCUDAExecutionContext, elementType);
        } else {
            if (arguments.length != 2) {
                CompilerDirectives.transferToInterpreter();
                throw ArityException.create(2, arguments.length);
            }
            return mapNode.execute(arguments[1], elementType, grCUDAExecutionContext);
        }
    }
}
