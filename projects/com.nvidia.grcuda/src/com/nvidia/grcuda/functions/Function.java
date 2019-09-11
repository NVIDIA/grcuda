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

import com.oracle.truffle.api.frame.VirtualFrame;
import com.oracle.truffle.api.interop.ForeignAccess;
import com.oracle.truffle.api.interop.Message;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.nodes.Node;

public abstract class Function implements TruffleObject {

    private final String name;
    private final String namespace;
    private final Node isBoxed = Message.IS_BOXED.createNode();
    private final Node unbox = Message.UNBOX.createNode();

    protected Function(String name, String namespace) {
        this.name = name;
        this.namespace = namespace;
    }

    @Override
    public ForeignAccess getForeignAccess() {
        return FunctionForeign.ACCESS;
    }

    public String getName() {
        return name;
    }

    public String getNamespace() {
        return namespace;
    }

    public abstract Object execute(VirtualFrame frame);

    protected boolean isString(Object obj) {
        if (obj instanceof String) {
            return true;
        } else if (obj instanceof TruffleObject) {
            // see if the TruffleObject unboxes into a string
            TruffleObject truffleObj = (TruffleObject) obj;
            if (ForeignAccess.sendIsBoxed(isBoxed, truffleObj)) {
                try {
                    Object unboxedObj = ForeignAccess.sendUnbox(unbox, truffleObj);
                    return unboxedObj instanceof String;
                } catch (UnsupportedMessageException e) {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    protected String expectString(Object argument, String errorMessage) {
        if (argument instanceof String) {
            return (String) argument;
        } else if (argument instanceof TruffleObject) {
            TruffleObject argTruffleValue = (TruffleObject) argument;
            if (ForeignAccess.sendIsBoxed(isBoxed, argTruffleValue)) {
                try {
                    Object valueObj = ForeignAccess.sendUnbox(unbox, argTruffleValue);
                    if (valueObj instanceof String) {
                        return (String) valueObj;
                    } else {
                        throw new RuntimeException(errorMessage + ": unboxed value has type " +
                                        valueObj.getClass().getName());
                    }
                } catch (UnsupportedMessageException ex) {
                    throw new RuntimeException("UNBOX message not supported on type " + argTruffleValue);
                }
            } else {
                throw new RuntimeException(errorMessage + ": got TruffleObject " + argument.getClass().getName());
            }
        } else {
            throw new RuntimeException(errorMessage + ": got " + argument.getClass().getName());
        }
    }

    protected Number expectNumber(Object argument, String errorMessage) {
        if (argument instanceof Number) {
            return (Number) argument;
        } else if (argument instanceof TruffleObject) {
            TruffleObject argTruffleValue = (TruffleObject) argument;
            if (ForeignAccess.sendIsBoxed(isBoxed, argTruffleValue)) {
                try {
                    Object valueObj = ForeignAccess.sendUnbox(unbox, argTruffleValue);
                    if (valueObj instanceof Number) {
                        return (Number) valueObj;
                    } else {
                        throw new RuntimeException(errorMessage + ": unboxed value has type " +
                                        valueObj.getClass().getName());
                    }
                } catch (UnsupportedMessageException ex) {
                    throw new RuntimeException("UNBOX message not supported on type " + argTruffleValue);
                }
            } else {
                throw new RuntimeException(errorMessage + ": got TruffleObject " + argument.getClass().getName());
            }
        } else {
            throw new RuntimeException(errorMessage + ": got " + argument.getClass().getName());
        }
    }
}
