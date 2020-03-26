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
grammar GrCUDA;

@parser::header
{
import java.util.ArrayList;
import java.util.Optional;
import com.nvidia.grcuda.GrCUDALanguage;
import com.nvidia.grcuda.nodes.ArrayNode;
import com.nvidia.grcuda.nodes.CallNode;
import com.nvidia.grcuda.nodes.ExpressionNode;
import com.nvidia.grcuda.nodes.IdentifierNode;
import com.nvidia.grcuda.nodes.GrCUDARootNode;
import com.nvidia.grcuda.parser.NodeFactory;
import com.nvidia.grcuda.parser.GrCUDAParserException;
import com.oracle.truffle.api.source.Source;
}

@parser::members
{
private NodeFactory factory;

public static ExpressionNode parseCUDA(Source source) {
    GrCUDALexer lexer = new GrCUDALexer(CharStreams.fromString(source.getCharacters().toString()));
    GrCUDAParser parser = new GrCUDAParser(new CommonTokenStream(lexer));
    lexer.removeErrorListeners();
    parser.removeErrorListeners();
    parser.factory = new NodeFactory(source);
    ParserErrorListener parserErrorListener = new ParserErrorListener(source);
    parser.addErrorListener(parserErrorListener);
    ExpressionNode expression = parser.expr().result;
    Optional<GrCUDAParserException> maybeException = parserErrorListener.getException();
    if (maybeException.isPresent()) {
      throw maybeException.get();
    } else {
      return expression;
    }
}

private static class ParserErrorListener extends BaseErrorListener {
    private GrCUDAParserException exception;
    private Source source;

    ParserErrorListener(Source source) {
      this.source = source;
    }

    @Override
    public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol,
                            int line, int charPositionInLine,
                            String msg, RecognitionException e) {
      Token token = (Token) offendingSymbol;
      exception = new GrCUDAParserException(msg, source, line, charPositionInLine,
                                            Math.max(token.getStopIndex() - token.getStartIndex(), 0));
    }

    public Optional<GrCUDAParserException> getException() {
      return Optional.ofNullable(exception);
    }
}
}

// parser

expr returns [ExpressionNode result]
  : arrayExpr EOF   { $result = $arrayExpr.result; }
  | callExpr EOF    { $result = $callExpr.result; }
  | callable EOF    { $result = $callable.result; }
  ;

arrayExpr returns [ArrayNode result]
          locals [ArrayList<ExpressionNode> dims = new ArrayList<>()]
  : Identifier
    ('['
    constExpr      { $dims.add($constExpr.result); }
    ']')+          { $result = factory.createArrayNode($Identifier, $dims); }
   ;

callExpr returns [CallNode result]
  : Identifier '(' ')'  { $result = factory.createCallNode($Identifier, new ExpressionNode[0]); }
  | Identifier
   '('
     (argumentList)?
   ')'                  { ExpressionNode[] argList = new ExpressionNode[$argumentList.result.size()];
                          $result = factory.createCallNode($Identifier, $argumentList.result.toArray(argList)); }
   ;

callable returns [IdentifierNode result]
  :  ('::')? name=Identifier         { $result = factory.createIdentifier($name); }
  |     namespace=Identifier
        '::' name=Identifier         { $result = factory.createIdentifierInNamespace($name, $namespace); }
  ;

argumentList returns [ArrayList<ExpressionNode> result]
  : argExpr                          { $result = new ArrayList<>(); $result.add($argExpr.result); }
  | al=argumentList ',' argExpr      { $al.result.add($argExpr.result); $result = $al.result; }
  ;

argExpr returns [ExpressionNode result]
  : Identifier           { $result = factory.createIdentifier($Identifier); }
  | String               { $result = factory.createStringLiteral($String); }
  | constExpr            { $result = $constExpr.result; }
  ;

constExpr returns [ExpressionNode result]
  : constTerm           { $result = $constTerm.result; }
  (
    op=('+' | '-')
    constTerm           { $result = factory.createBinary($op, $result, $constTerm.result); }
  )*
  ;

constTerm returns [ExpressionNode result]
  : constFactor         { $result = $constFactor.result; }
  (
    op=('*' | '/' | '%')
    constFactor         { $result = factory.createBinary($op, $result, $constFactor.result); }
   )*
  ;

constFactor returns [ExpressionNode result]
  : IntegerLiteral      { $result = factory.createIntegerLiteral($IntegerLiteral); }
  | '('
    constExpr           { $result = $constExpr.result; }
    ')'
  ;

// lexer

String: '"' StringChar* '"';
Identifier: Letter (Letter | Digit)*;
IntegerLiteral: Digit+;

fragment Digit: [0-9];
fragment Letter: [A-Z] | [a-z] | '_' | '$';
fragment StringChar: ~('"' | '\\' | '\r' | '\n');

WS: (' ' | '\t')+ -> skip;
Comment: '/*' .*? '*/' -> skip;
LineComment: '//' ~[\r\n]* -> skip;
NL: '\r'? '\n' -> skip;
