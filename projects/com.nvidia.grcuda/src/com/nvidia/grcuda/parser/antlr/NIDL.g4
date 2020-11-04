/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
grammar NIDL;


@parser::header
{
import java.io.IOException;
import java.util.ArrayList;
import java.nio.charset.Charset;
import java.util.Optional;
import com.nvidia.grcuda.Binding;
import com.nvidia.grcuda.FunctionBinding;
import com.nvidia.grcuda.KernelBinding;
import com.nvidia.grcuda.ComputationArgument;
import com.nvidia.grcuda.Type;
import com.nvidia.grcuda.TypeException;
import com.nvidia.grcuda.parser.NIDLParserException;
}

@parser::members
{
public static ArrayList<Binding> parseNIDLFile(String filename) {
  try {
    CharStream stream = CharStreams.fromFileName(filename, Charset.forName("UTF-8"));
      NIDLLexer lexer = new NIDLLexer(stream);
      NIDLParser parser = new NIDLParser(new CommonTokenStream(lexer));
      parser.filename = filename;
      lexer.removeErrorListeners();
      parser.removeErrorListeners();
      ParserErrorListener parserErrorListener = new ParserErrorListener(filename);
      parser.addErrorListener(parserErrorListener);
      ArrayList<Binding> bindings = parser.nidlSpec().result;
      Optional<NIDLParserException> maybeException = parserErrorListener.getException();
      if (maybeException.isPresent()) {
        throw maybeException.get();
      } else {
        return bindings;
      }
    } catch(IOException e) {
      throw new NIDLParserException("file not found or cannot be opened", filename, 0, 0);
    }
}

private String filename;

private static class ParserErrorListener extends BaseErrorListener {
    private Optional<NIDLParserException> maybeException = Optional.empty();
    private final String filename;

    public ParserErrorListener(String filename) {
      this.filename = filename;
    }

    @Override
    public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol,
                            int line, int charPositionInLine,
                            String msg, RecognitionException e) {
    // keep first exception only
    if (!maybeException.isPresent()) {
      Token token = (Token) offendingSymbol;
      maybeException = Optional.of(new NIDLParserException(msg, filename, line, charPositionInLine));
    }
  }

    public Optional<NIDLParserException> getException() {
      return maybeException;
    }
}
}

// parser

nidlSpec returns [ArrayList<Binding> result]
  : scopes EOF         {$result = $scopes.result; }
  ;

scopes returns [ArrayList<Binding> result]
  : ss=scopes scope { $result = $ss.result;
                      if ($scope.result != null) {
                        // prevent NPE of a parser error occurred in the subtree
                        $result.addAll($scope.result);
                      }
                    }
  |                 { $result = new ArrayList<Binding>(); }
  ;

scope returns [ArrayList<Binding> result]
  : 'kernels'  '{' kernels '}'           { $result = $kernels.result;  }
  | 'ckernels' '{' ckernels '}'          { $result = $ckernels.result; }
  | 'hostfuncs' '{' hostFunctions '}'    { $result = $hostFunctions.result; }
  | 'chostfuncs' '{' chostFunctions '}'  { $result = $chostFunctions.result; }
  | 'kernels' namespaceIdentifier '{' kernels '}'
     { $result = $kernels.result;
       for (Binding binding : $kernels.result) {
         binding.setNamespace($namespaceIdentifier.result);
       }
	 }
  | 'hostfuncs' namespaceIdentifier '{' hostFunctions '}'
    { $result = $hostFunctions.result;
      for (Binding binding : $hostFunctions.result) {
        binding.setNamespace($namespaceIdentifier.result);
      }
    }
  ;

kernels returns [ArrayList<Binding> result]
  : ks=kernels kernel    { $result = $ks.result;
                           $result.add($kernel.result); }
  |                      { $result = new ArrayList<Binding>(); }
  ;

kernel returns [Binding result]
  : n=Identifier '(' pl=computationArgumentList ')'
    { $result = KernelBinding.newCxxBinding($n.getText(), $pl.result); }
  ;

ckernels returns [ArrayList<Binding> result]
  : cks=ckernels ckernel  { $result = $cks.result;
                            $result.add($ckernel.result); }
  |                       { $result = new ArrayList<Binding>(); }
  ;

ckernel returns [Binding result]
  : n=Identifier '(' pl=computationArgumentList ')'
    { $result = KernelBinding.newCBinding($n.getText(), $pl.result); }
  ;

hostFunctions returns [ArrayList<Binding> result]
  : hfs=hostFunctions hostFunction { $result = $hfs.result;
                                     if ($hostFunction.result != null) {
                                        // prevent NPE of a parser error occurred in the subtree
                                        $result.add($hostFunction.result);
                                     }
                                   }
  |                                { $result = new ArrayList<Binding>(); }
  ;

hostFunction returns [Binding result]
  : n=Identifier '(' pl=computationArgumentList ')' ':' rt=Identifier
    { try {
      $result = FunctionBinding.newCxxBinding($n.getText(), $pl.result,
    		  Type.fromNIDLTypeString($rt.getText()));
      } catch (TypeException e) {
        throw new NIDLParserException(e.getMessage(), filename, $rt.getLine(), $rt.getCharPositionInLine());
      }
    }
  ;

chostFunctions returns [ArrayList<Binding> result]
  : chfs=chostFunctions chostFunction { $result = $chfs.result;
                                        $result.add($chostFunction.result); }
  |                                   { $result = new ArrayList<Binding>(); }
  ;

chostFunction returns [Binding result]
  : n=Identifier '(' pl=computationArgumentList ')' ':' rt=Identifier
    { try {
      $result = FunctionBinding.newCBinding($n.getText(), $pl.result,
              Type.fromNIDLTypeString($rt.getText()));
      } catch (TypeException e) {
        throw new NIDLParserException(e.getMessage(), filename, $rt.getLine(), $rt.getCharPositionInLine());
      }
    }
  ;

computationArgumentList returns [ArrayList<ComputationArgument> result]
  : pl=computationArgumentList ',' paramExpr
    { $result = $pl.result;
      if ($paramExpr.result != null) {
        // avoids NPE during parser error
        $paramExpr.result.setPosition($pl.result.size());
        $result.add($paramExpr.result);
      }
    }
  | paramExpr
    { $result = new ArrayList<ComputationArgument>();
      $result.add($paramExpr.result);
    }
  |
    { $result = new ArrayList<ComputationArgument>(); }
  ;

paramExpr returns [ComputationArgument result]
  : n=Identifier ':' t=Identifier
    { try {
        $result = ComputationArgument.createByValueComputationArgument($n.getText(), Type.fromNIDLTypeString($t.getText()));
      } catch(TypeException e) {
        throw new NIDLParserException(e.getMessage(), filename, $t.getLine(), $t.getCharPositionInLine());
      }
    }
  | n=Identifier ':' direction 'pointer' t=Identifier
    { try {
        $result = ComputationArgument.createPointerComputationArgument($n.getText(), Type.fromNIDLTypeString($t.getText()), $direction.result);
      } catch(TypeException e) {
        throw new NIDLParserException(e.getMessage(), filename, $t.getLine(), $t.getCharPositionInLine());
      }
    }
  ;

direction returns [ComputationArgument.Kind result]
  : 'in'    { $result = ComputationArgument.Kind.POINTER_IN; }
  | 'out'   { $result = ComputationArgument.Kind.POINTER_OUT; }
  | 'inout' { $result = ComputationArgument.Kind.POINTER_INOUT; }
  ;

namespaceIdentifier returns [ArrayList<String> result]
  : ons=namespaceIdentifier '::' id=Identifier
    {$result = $ons.result;
     $result.add($id.getText());
    }
  | id=Identifier
    {$result = new ArrayList<String>();
     $result.add($id.getText()); }
  ;

// lexer
Identifier: Letter (Letter | Digit)*;

fragment Digit: [0-9];
fragment Letter: [A-Z] | [a-z] | '_' | '$';

WS: (' ' | '\t')+ -> skip;
Comment: '/*' .*? '*/' -> skip;
LineComment: '//' ~[\r\n]* -> skip;
NL: '\r'? '\n' -> skip;
