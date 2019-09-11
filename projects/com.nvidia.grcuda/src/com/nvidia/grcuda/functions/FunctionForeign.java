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
import com.oracle.truffle.api.CallTarget;
import com.oracle.truffle.api.Truffle;
import com.oracle.truffle.api.frame.VirtualFrame;
import com.oracle.truffle.api.interop.ForeignAccess;
import com.oracle.truffle.api.interop.ForeignAccess.Factory;
import com.oracle.truffle.api.interop.ForeignAccess.StandardFactory;
import com.oracle.truffle.api.interop.Message;
import com.oracle.truffle.api.interop.TruffleObject;
import com.oracle.truffle.api.nodes.RootNode;

public class FunctionForeign implements StandardFactory, Factory {
    public static final ForeignAccess ACCESS = ForeignAccess.createAccess(new FunctionForeign(), null);

    @Override
    public boolean canHandle(TruffleObject obj) {
        return obj instanceof Function;
    }

    @Override
    public CallTarget accessIsNull() {
        return Truffle.getRuntime().createCallTarget(RootNode.createConstantNode(false));
    }

    @Override
    public CallTarget accessIsExecutable() {
        return Truffle.getRuntime().createCallTarget(RootNode.createConstantNode(true));
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
        return null;
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
    public CallTarget accessExecute(int argumentsLength) {
        return Truffle.getRuntime().createCallTarget(new RootNode(GrCUDALanguage.getCurrentLanguage()) {

            @Override
            public Object execute(VirtualFrame frame) {
                Object[] functionAndArguments = frame.getArguments();
                assert functionAndArguments.length > 0;
                Object functionObj = functionAndArguments[0];
                if (functionObj instanceof Function) {
                    Function f = (Function) functionObj;
                    return f.execute(frame);
                } else {
                    throw new RuntimeException("unsupported callable " + functionObj);
                }
            }
        });
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
        return null;
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
