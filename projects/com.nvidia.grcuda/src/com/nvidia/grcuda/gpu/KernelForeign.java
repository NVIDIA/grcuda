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
package com.nvidia.grcuda.gpu;

import com.nvidia.grcuda.GrCUDALanguage;
import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.Truffle;
import com.oracle.truffle.api.frame.VirtualFrame;
import com.oracle.truffle.api.interop.ForeignAccess;
import com.oracle.truffle.api.interop.ForeignAccess.Factory;
import com.oracle.truffle.api.interop.ForeignAccess.StandardFactory;
import com.oracle.truffle.api.interop.KeyInfo;
import com.oracle.truffle.api.interop.Message;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnknownIdentifierException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.nodes.Node;
import com.oracle.truffle.api.nodes.RootNode;
import com.oracle.truffle.api.profiles.ConditionProfile;

public class KernelForeign implements StandardFactory, Factory {
    public static final ForeignAccess ACCESS = ForeignAccess.createAccess(new KernelForeign(), null);

    @Override
    public CallTarget accessExecute(int argumentsLength) {
        return Truffle.getRuntime().createCallTarget(new RootNode(GrCUDALanguage.getCurrentLanguage()) {

            @Child private Node hasSize = Message.HAS_SIZE.createNode();

            @Child private Node getSize = Message.GET_SIZE.createNode();

            @Child private Node readNode = Message.READ.createNode();

            @Child private Node isBoxed = Message.IS_BOXED.createNode();

            @Child private Node unbox = Message.UNBOX.createNode();

            private int extractNumber(Object valueObj, String argumentName) {
                if (valueObj instanceof TruffleObject) {
                    TruffleObject truffleValue = (TruffleObject) valueObj;
                    // Check if value is a boxed type and if so unbox it.
                    // This is necessary for R, since scalars in R are arrays of length 1.
                    if (ForeignAccess.sendIsBoxed(isBoxed, truffleValue)) {
                        try {
                            valueObj = ForeignAccess.sendUnbox(unbox, truffleValue);
                        } catch (UnsupportedMessageException ex) {
                            throw new RuntimeException("UNBOX message not supported on type " + truffleValue);
                        }
                    }
                }
                if (!(valueObj instanceof Number)) {
                    throw new RuntimeException(argumentName + " is " + valueObj.getClass().getName() + ", expected a Number object");
                }
                return ((Number) valueObj).intValue();
            }

            private Dim3 extractDim3(Object valueObj, String argumentName) {
                if (valueObj instanceof TruffleObject) {
                    TruffleObject truffleValue = (TruffleObject) valueObj;
                    // Check if value has a size in which case it may be multi-dimensional argument
                    if (ForeignAccess.sendHasSize(hasSize, truffleValue)) {
                        try {
                            Object numElementsObj = ForeignAccess.sendGetSize(getSize, truffleValue);
                            if (numElementsObj instanceof Number) {
                                int numElements = ((Number) numElementsObj).intValue();
                                if ((numElements < 1) || (numElements > 3)) {
                                    throw new RuntimeException(("Dim3 must have between 1 and 3 elements"));
                                }
                                int[] dim3 = new int[3];
                                final char[] suffix = {'x', 'y', 'z'};
                                for (int i = 0; i < numElements; i++) {
                                    try {
                                        Object elementObj = ForeignAccess.sendRead(readNode, truffleValue, i);
                                        dim3[i] = extractNumber(elementObj, "dim3." + suffix[i]);
                                    } catch (UnknownIdentifierException e) {
                                        throw new RuntimeException(e);
                                    }
                                }
                                if (numElements == 1) {
                                    return new Dim3(dim3[0]);
                                } else if (numElements == 2) {
                                    return new Dim3(dim3[0], dim3[1]);
                                } else {
                                    return new Dim3(dim3[0], dim3[1], dim3[2]);
                                }
                            } else {
                                throw new RuntimeException("Internal Error: Dim3 must have numeric size");
                            }
                        } catch (UnsupportedMessageException e) {
                            throw new RuntimeException("Internal Error: Dim3 must have a size");
                        }
                    }
                    // Check if value is a boxed type and if so unbox it.
                    // This is necessary for R, since scalars in R are arrays of length 1.
                    if (ForeignAccess.sendIsBoxed(isBoxed, truffleValue)) {
                        try {
                            valueObj = ForeignAccess.sendUnbox(unbox, truffleValue);
                        } catch (UnsupportedMessageException ex) {
                            throw new RuntimeException("UNBOX message not supported on type " + truffleValue);
                        }
                    }
                }
                if (!(valueObj instanceof Number)) {
                    throw new RuntimeException(argumentName + " is " + valueObj.getClass().getName() + ", expected a Number object");
                }
                return new Dim3(((Number) valueObj).intValue());
            }

            @Override
            public Object execute(VirtualFrame frame) {
                Object[] args = frame.getArguments();
                assert args.length == 3 || args.length == 4;
                Object kernelObj = args[0];
                Object gridSizeObj = args[1];
                Object blockSizeObj = args[2];
                if (!(kernelObj instanceof Kernel)) {
                    throw new RuntimeException("unsupported callable " + kernelObj + ", expected Kernel object");
                }
                Kernel kernel = (Kernel) kernelObj;
                Dim3 gridSize = extractDim3(gridSizeObj, "gridSize");
                Dim3 blockSize = extractDim3(blockSizeObj, "blockSize");
                KernelConfig config;
                if (args.length == 4) {
                    // dynamic shared memory specified
                    int dynamicSharedMemoryBytes = extractNumber(args[3], "dynamicSharedMemory");
                    config = new KernelConfig(gridSize, blockSize, dynamicSharedMemoryBytes);
                } else {
                    config = new KernelConfig(gridSize, blockSize);
                }
                return new ConfiguredKernel(kernel, config);
            }
        });
    }

    @Override
    public CallTarget accessIsExecutable() {
        return Truffle.getRuntime().createCallTarget(RootNode.createConstantNode(true));
    }

    @Override
    public boolean canHandle(TruffleObject obj) {
        return obj instanceof KernelForeign;
    }

    @Override
    public CallTarget accessIsNull() {
        return Truffle.getRuntime().createCallTarget(RootNode.createConstantNode(false));
    }

    @Override
    public CallTarget accessIsInstantiable() {
        return Truffle.getRuntime().createCallTarget(RootNode.createConstantNode(false));
    }

    @Override
    public CallTarget accessIsBoxed() {
        return Truffle.getRuntime().createCallTarget(RootNode.createConstantNode(false));
    }

    @Override
    public CallTarget accessHasSize() {
        return Truffle.getRuntime().createCallTarget(RootNode.createConstantNode(false));
    }

    @Override
    public CallTarget accessHasKeys() {
        return Truffle.getRuntime().createCallTarget(RootNode.createConstantNode(false));
    }

    @Override
    public CallTarget accessIsPointer() {
        return Truffle.getRuntime().createCallTarget(RootNode.createConstantNode(false));
    }

    @Override
    public CallTarget accessGetSize() {
        return null;
    }

    @Override
    public CallTarget accessUnbox() {
        return null;
    }

    @Override
    public CallTarget accessRead() {
        return Truffle.getRuntime().createCallTarget(new RootNode(GrCUDALanguage.getCurrentLanguage()) {
            private final ConditionProfile stringProfile = ConditionProfile.createBinaryProfile();

            @Override
            public Object execute(VirtualFrame frame) {
                Kernel kernel = (Kernel) ForeignAccess.getReceiver(frame);
                Object name = ForeignAccess.getArguments(frame).get(0);
                if (stringProfile.profile(name instanceof String)) {
                    String attribute = (String) name;
                    if ("ptx".equals(name)) {
                        String ptx = kernel.getPTX();
                        if (ptx == null) {
                            return "<no PTX code>";
                        } else {
                            return ptx;
                        }
                    } else if ("name".equals(name)) {
                        return kernel.getKernelName();
                    } else if ("launchCount".equals(name)) {
                        return kernel.getLaunchCount();
                    }
                    throw new IllegalArgumentException("no such attribute '" + attribute + "'");
                } else {
                    throw new IllegalArgumentException("unsupported read access type " + name.getClass().getName());
                }
            }
        });
    }

    @Override
    public CallTarget accessWrite() {
        return null;
    }

    @Override
    public CallTarget accessRemove() {
        return null;
    }

    @Override
    public CallTarget accessInvoke(int argumentsLength) {
        return null;
    }

    @Override
    public CallTarget accessNew(int argumentsLength) {
        return null;
    }

    @Override
    public CallTarget accessKeys() {
        return null;
    }

    @Override
    public CallTarget accessKeyInfo() {
        return Truffle.getRuntime().createCallTarget(new RootNode(GrCUDALanguage.getCurrentLanguage()) {
            private final ConditionProfile stringProfile = ConditionProfile.createBinaryProfile();

            @Override
            public Object execute(VirtualFrame frame) {
                Kernel kernel = (Kernel) ForeignAccess.getReceiver(frame);
                Object name = ForeignAccess.getArguments(frame).get(0);
                if (stringProfile.profile(name instanceof String)) {
                    String attribute = (String) name;
                    if ("ptx".equals(attribute) || "name".equals(attribute) || "launchCount".equals(attribute)) {
                        return KeyInfo.READABLE;
                    }
                }
                return KeyInfo.NONE;
            }
        });
    }

    @Override
    public CallTarget accessAsPointer() {
        return null;
    }

    @Override
    public CallTarget accessToNative() {
        return null;
    }

    @Override
    public CallTarget accessMessage(Message unknown) {
        return null;
    }
}
