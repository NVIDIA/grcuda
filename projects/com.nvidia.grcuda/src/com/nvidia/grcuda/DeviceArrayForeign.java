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
package com.nvidia.grcuda;

import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.Truffle;
import com.oracle.truffle.api.frame.VirtualFrame;
import com.oracle.truffle.api.interop.ForeignAccess;
import com.oracle.truffle.api.interop.ForeignAccess.Factory;
import com.oracle.truffle.api.interop.ForeignAccess.StandardFactory;
import com.oracle.truffle.api.interop.KeyInfo;
import com.oracle.truffle.api.interop.Message;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.nodes.Node;
import com.oracle.truffle.api.nodes.RootNode;
import com.oracle.truffle.api.profiles.ConditionProfile;
import com.oracle.truffle.api.profiles.ValueProfile;

public final class DeviceArrayForeign implements StandardFactory, Factory {

    public static final ForeignAccess ACCESS = ForeignAccess.createAccess(new DeviceArrayForeign(), null);

    @Override
    public boolean canHandle(TruffleObject obj) {
        return obj instanceof DeviceArray;
    }

    @Override
    public CallTarget accessIsNull() {
        return Truffle.getRuntime().createCallTarget(RootNode.createConstantNode(false));
    }

    @Override
    public CallTarget accessIsExecutable() {
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
        return Truffle.getRuntime().createCallTarget(RootNode.createConstantNode(true));
    }

    @Override
    public CallTarget accessGetSize() {
        return Truffle.getRuntime().createCallTarget(new RootNode(GrCUDALanguage.getCurrentLanguage()) {

            @Override
            public Object execute(VirtualFrame frame) {
                DeviceArray deviceArray = (DeviceArray) ForeignAccess.getReceiver(frame);
                long numElements = deviceArray.getSizeElements();
                // FIXME The current Truffle API expects GET_SIZE to return an Integer.
                if (numElements > Integer.MAX_VALUE) {
                    throw new GrCUDAException("array size is out of int range " + numElements, this);
                }
                return (int) numElements;
            }
        });
    }

    @Override
    public CallTarget accessRead() {
        return Truffle.getRuntime().createCallTarget(new RootNode(GrCUDALanguage.getCurrentLanguage()) {

            @Child private DeviceArray.ReadElementNode elementAccessNode = new DeviceArray.ReadElementNode();

            private final ConditionProfile numberProfile = ConditionProfile.createBinaryProfile();
            private final ConditionProfile stringProfile = ConditionProfile.createBinaryProfile();

            @Override
            public Object execute(VirtualFrame frame) {
                DeviceArray deviceArray = (DeviceArray) ForeignAccess.getReceiver(frame);
                Object index = ForeignAccess.getArguments(frame).get(0);
                if (numberProfile.profile(index instanceof Number)) {
                    return elementAccessNode.readElement(deviceArray, ((Number) index).longValue());
                } else if (stringProfile.profile(index instanceof String)) {
                    String attribute = (String) index;
                    if ("pointer".equals(attribute)) {
                        return deviceArray.getPointer();
                    }
                    throw new IllegalArgumentException("no such attribute '" + attribute + "'");
                } else {
                    throw new IllegalArgumentException("unsupported index type " +
                                    index.getClass().getName());
                }
            }
        });
    }

    @Override
    public CallTarget accessWrite() {
        return Truffle.getRuntime().createCallTarget(new RootNode(GrCUDALanguage.getCurrentLanguage()) {

            private final ConditionProfile isTruffleProfile = ConditionProfile.createBinaryProfile();
            private final ConditionProfile boxedValueProfile = ConditionProfile.createBinaryProfile();
            private final ConditionProfile valueNotANumberProfile = ConditionProfile.createBinaryProfile();
            private final ConditionProfile numberIndexProfile = ConditionProfile.createBinaryProfile();
            private final ValueProfile indexTypeProfile = ValueProfile.createClassProfile();
            private final ValueProfile valueTypeProfile = ValueProfile.createClassProfile();

            @Child private Node isBoxed = Message.IS_BOXED.createNode();

            @Child private Node unbox = Message.UNBOX.createNode();

            @Child private DeviceArray.WriteElementNode elementAccessNode = new DeviceArray.WriteElementNode();

            @Override
            public Object execute(VirtualFrame frame) {
                DeviceArray deviceArray = (DeviceArray) ForeignAccess.getReceiver(frame);
                Object index = indexTypeProfile.profile(ForeignAccess.getArguments(frame).get(0));
                Object valueObj = ForeignAccess.getArguments(frame).get(1);
                if (isTruffleProfile.profile(valueObj instanceof TruffleObject)) {
                    TruffleObject truffleValue = (TruffleObject) valueObj;
                    // Check if value is a boxed type and if so unbox it.
                    // This is necessary for R, since scalars in R are arrays of length 1.
                    if (boxedValueProfile.profile(ForeignAccess.sendIsBoxed(isBoxed, truffleValue))) {
                        try {
                            valueObj = ForeignAccess.sendUnbox(unbox, truffleValue);
                        } catch (UnsupportedMessageException ex) {
                            throw new RuntimeException("UNBOX message not supported on type " + truffleValue);
                        }
                    }
                }
                valueTypeProfile.profile(valueObj);
                if (valueNotANumberProfile.profile(!(valueObj instanceof Number))) {
                    throw new IllegalArgumentException("unsupported value type " +
                                    valueObj.getClass().getName());
                }
                Number value = (Number) valueObj;
                if (numberIndexProfile.profile(index instanceof Number)) {
                    elementAccessNode.writeElement(deviceArray, ((Number) index).longValue(), value);
                } else {
                    throw new IllegalArgumentException("unsupported index type " +
                                    index.getClass().getName());
                }
                return value;
            }
        });
    }

    @Override
    public CallTarget accessRemove() {
        return null;
    }

    @Override
    public CallTarget accessUnbox() {
        return null;
    }

    @Override
    public CallTarget accessExecute(int argumentsLength) {
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
    public CallTarget accessHasKeys() {
        return null;
    }

    @Override
    public CallTarget accessKeys() {
        return null;
    }

    @Override
    public CallTarget accessKeyInfo() {
        return Truffle.getRuntime().createCallTarget(new RootNode(GrCUDALanguage.getCurrentLanguage()) {
            private final ConditionProfile numberProfile = ConditionProfile.createBinaryProfile();
            private final ConditionProfile stringProfile = ConditionProfile.createBinaryProfile();

            @Override
            public Object execute(VirtualFrame frame) {
                DeviceArray deviceArray = (DeviceArray) ForeignAccess.getReceiver(frame);
                Object index = ForeignAccess.getArguments(frame).get(0);
                if (numberProfile.profile(index instanceof Number)) {
                    return deviceArray.isIndexValid(((Number) index).longValue()) ? (KeyInfo.READABLE | KeyInfo.MODIFIABLE) : KeyInfo.NONE;
                } else if (stringProfile.profile(index instanceof String)) {
                    String attribute = (String) index;
                    if ("pointer".equals(attribute)) {
                        return KeyInfo.READABLE;
                    }
                }
                return KeyInfo.NONE;
            }
        });
    }

    @Override
    public CallTarget accessIsPointer() {
        return null;
    }

    @Override
    public CallTarget accessAsPointer() {
        return null;
    }

    @Override
    public CallTarget accessToNative() {
        return Truffle.getRuntime().createCallTarget(new RootNode(GrCUDALanguage.getCurrentLanguage()) {

            @Override
            public Object execute(VirtualFrame frame) {
                DeviceArray deviceArray = (DeviceArray) ForeignAccess.getReceiver(frame);
                return new GPUPointer(deviceArray.getPointer());
            }
        });
    }

    @Override
    public CallTarget accessMessage(Message unknown) {
        System.out.println("access Message: " + unknown);
        return null;
    }
}
