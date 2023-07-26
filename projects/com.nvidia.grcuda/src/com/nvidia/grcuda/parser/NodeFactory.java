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
package com.nvidia.grcuda.parser;

import java.util.ArrayList;
import org.antlr.v4.runtime.Token;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.GrCUDAInternalException;
import com.nvidia.grcuda.TypeException;
import com.nvidia.grcuda.nodes.ArithmeticNode;
import com.nvidia.grcuda.nodes.ArrayNode;
import com.nvidia.grcuda.nodes.ArrayNodeGen;
import com.nvidia.grcuda.nodes.CallNode;
import com.nvidia.grcuda.nodes.CallNodeGen;
import com.nvidia.grcuda.nodes.ExpressionNode;
import com.nvidia.grcuda.nodes.IdentifierNode;
import com.nvidia.grcuda.nodes.IdentifierNodeGen;
import com.nvidia.grcuda.nodes.IntegerLiteral;
import com.nvidia.grcuda.nodes.StringLiteral;
import com.oracle.truffle.api.source.Source;

public class NodeFactory {

    private final Source source;

    public NodeFactory(Source source) {
        this.source = source;
    }

    public ArrayNode createArrayNode(Token typeToken, ArrayList<ExpressionNode> sizeNodes) throws GrCUDAParserException {
        return ArrayNodeGen.create(lookupType(typeToken), sizeNodes);
    }

    public ArithmeticNode createBinary(Token opToken, ExpressionNode leftNode, ExpressionNode rightNode) {
        final ArithmeticNode result;
        switch (opToken.getText()) {
            case "+":
                result = new ArithmeticNode(ArithmeticNode.Operation.ADD, leftNode, rightNode);
                break;
            case "-":
                result = new ArithmeticNode(ArithmeticNode.Operation.SUBTRACT, leftNode, rightNode);
                break;
            case "*":
                result = new ArithmeticNode(ArithmeticNode.Operation.MULTIPLY, leftNode, rightNode);
                break;
            case "/":
                result = new ArithmeticNode(ArithmeticNode.Operation.DIVIDE, leftNode, rightNode);
                break;
            case "%":
                result = new ArithmeticNode(ArithmeticNode.Operation.MODULO, leftNode, rightNode);
                break;
            default:
                // should not happen due to lexer
                throw new GrCUDAInternalException("unexpected operation: " + opToken.getText());
        }
        return result;
    }

    public CallNode createCallNode(Token identifierToken, ExpressionNode[] arguments) {
        return CallNodeGen.create(createIdentifier(identifierToken), arguments);
    }

    public IdentifierNode createIdentifier(Token identifierToken) {
        return IdentifierNodeGen.create(identifierToken.getText());
    }

    public IdentifierNode createIdentifierInNamespace(Token identifierToken, Token namespaceToken) {
        return IdentifierNodeGen.create(namespaceToken.getText(), identifierToken.getText());

    }

    public IntegerLiteral createIntegerLiteral(Token literalToken) {
        try {
            return new IntegerLiteral(Integer.parseInt(literalToken.getText()));
        } catch (NumberFormatException e) {
            // ignore parse error cannot happen due to regular expression in lexer
            throw new GrCUDAInternalException("unable to parse integer literal " + e.getMessage());
        }
    }

    public StringLiteral createStringLiteral(Token literalToken) {
        String stringValue = literalToken.getText();
        // Skip double-quotes at the beginning and at the end: "mystring"
        assert stringValue.startsWith("\"") && stringValue.endsWith("\"");
        return new StringLiteral(stringValue.substring(1, stringValue.length() - 1));
    }

    private Type lookupType(Token typeToken) {
        try {
            return Type.fromGrCUDATypeString(typeToken.getText());
        } catch (TypeException e) {
            throw new GrCUDAParserException(e.getMessage(),
                            source,
                            typeToken.getLine(), typeToken.getCharPositionInLine(),
                            Math.max(typeToken.getStopIndex() - typeToken.getStartIndex(), 0));
        }
    }
}
