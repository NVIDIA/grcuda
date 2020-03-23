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
 grammar NIDL;
 
 
@parser::header
{
import java.io.IOException;
import java.util.ArrayList;
import java.nio.charset.Charset;
import java.util.Optional;
import com.nvidia.grcuda.Argument;
import com.nvidia.grcuda.Binding;
import com.nvidia.grcuda.FunctionBinding;
import com.nvidia.grcuda.KernelBinding;
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
	    ArrayList<Binding> bindings = parser.bindings().result;
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
    private NIDLParserException exception;
    private final String filename;
    
    public ParserErrorListener(String filename) {
    	this.filename = filename;
   	}
  
    @Override
    public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol, int line, int charPositionInLine,
    	                    String msg, RecognitionException e) {
        Token token = (Token) offendingSymbol;
        exception = new NIDLParserException(msg, filename, line, charPositionInLine);
    }
    
    public Optional<NIDLParserException> getException() {
        return Optional.ofNullable(exception);
    }
}
}



// parser

bindings returns [ArrayList<Binding> result]
  : bds=bindings binding { $result = $bds.result;
                           $result.add($binding.result); }
  | binding          { $result = new ArrayList<Binding>(); 
                       $result.add($binding.result); } 
  ;

binding returns [Binding result]
  : cxxOption 'kernel' n=Identifier '(' al=argumentList ')'
    { $result = new KernelBinding($n.getText(), $al.result, $cxxOption.result); }
  | cxxOption 'func' n=Identifier '(' al=argumentList ')' ':' rt=Identifier
  	{ try {
  		$result = new FunctionBinding($n.getText(), $al.result,
  				Type.fromNIDLTypeString($rt.getText()), $cxxOption.result);
  	  } catch (TypeException e) {
  		throw new NIDLParserException(e.getMessage(), filename, $rt.getLine(), $rt.getCharPositionInLine());
  	  }
  	}
  ;
 
cxxOption returns [Boolean result]
  : 'cxx'          { $result = true; }
  |         	   { $result = false; }
  ;

argumentList returns [ArrayList<Argument> result]
  : al=argumentList ',' argExpr  
     { $result = $al.result; 
       $argExpr.result.setPosition($al.result.size());
       $result.add($argExpr.result);
     }
  | argExpr
  	 { $result = new ArrayList<Argument>(); 
  	   $result.add($argExpr.result);
  	 }
  |  
     { $result = new ArrayList<Argument>(); 
     }
  ;

argExpr returns [Argument result]
  : n=Identifier ':' direction t=Identifier 
  	{ try {
	  	$result = Argument.createArgument($n.getText(), Type.fromNIDLTypeString($t.getText()), $direction.result);
  	  } catch(TypeException e) {
  		  throw new NIDLParserException(e.getMessage(), filename, $t.getLine(), $t.getCharPositionInLine());
  	  }
  	}
  ; 

direction returns [Argument.Direction result]
  : 'in'    {$result = Argument.Direction.IN; }
  | 'out'   {$result = Argument.Direction.OUT; }
  | 'inout' {$result = Argument.Direction.INOUT; }
  |         {$result = Argument.Direction.BY_VALUE; }
  ;


// lexer

Identifier: Letter (Letter | Digit)+;

fragment Digit: [0-9];
fragment Letter: [A-Z] | [a-z] | '_' | '$';

WS: (' ' | '\t')+ -> skip;
Comment: '/*' .*? '*/' -> skip;
LineComment: '//' ~[\r\n]* -> skip;
NL: '\r'? '\n' -> skip;
