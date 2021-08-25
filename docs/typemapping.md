# Type Mapping for C++

GrCUDA needs to interact with tree different type systems:

- The Java types of the Truffle interop protocol (`boolean`, `byte`, `short`, `int`, `long`, `float`, `double`, `String`)
  as described in [InteropLibrary](https://www.graalvm.org/truffle/javadoc/com/oracle/truffle/api/interop/InteropLibrary.html).
  These types are used in the interop with the different GraalVM langauges.
- C++ language types (see [C++ Fundamental Types](https://en.cppreference.com/w/cpp/language/types)
  These types are used to used to bind to GPU kernels as well as to C++ host functions.
- C interface types from TruffleNFI (`void`, `sint8`, `uint8`, `sint16`, `uint16`, `sint32`, `uint32`, `sint64`, `uint64`,
    `float`, `double`, `pointer`, `object`, `string`).
  [NativeSimpleType.java](https://github.com/oracle/graal/blob/master/truffle/src/com.oracle.truffle.nfi.spi/src/com/oracle/truffle/nfi/spi/types/NativeSimpleType.java)
  defines these types. GrCUDA uses TruffleNFI to invoke native host functions. While TruffleNFI is designed for C APIs, the C++ types are necessary create the mangled symbols names to invoke with TruffleNFI.

Types for clang/gcc under Linux (LP64)

 C++ Type          |Bytes| Sym | NIDL Type          | NFI      | Java Type
-------------------|-----|-----|--------------------|----------|----------
bool               |  1  |  b  | bool               | uint8    | boolean
char               |  1  |  c  | char               | sint8    | byte
unsigned char      |  1  |  h  | uint8              | uint8    | short
signed char        |  1  |  a  | sint8              | sint8    | byte
wchar_t            |  4  |  w  | wchar              | sint32   | int
char8_t            |  1  | Du  | char8              | uint8    | short
char16_t           |  2  | Ds  | char16             | uint16   | int
char32_t           |  4  | Di  | char32             | uint32   | long
(signed) short     |  2  |  s  | sint16             | sint16   | short
unsigned short     |  2  |  t  | uint16             | uint16   | long
(signed) int       |  4  |  i  | sint32             | sint32   | int
unsigned int       |  4  |  j  | uint32             | uint32   | long
(signed) long      |  8  |  l  | sint64             | sint64   | long
unsigned long      |  8  |  m  | uint64             | uint64   | long+
(singed) long long |  8  |  x  | sll64              | sint64   | long
unsigned long long |  8  |  y  | ull64              | uint64   | long+
float              |  4  |  f  | float              | float    | float
double             |  8  |  d  | double             | double   | double
long double        | 16  |  e  | *                  | *        | *
void               |  0  |  v  | void               | void     | void
U *                |  8  | PU  | inout pointer V    | pointer  | n/a
U *                |  8  | PU  | out pointer V      | pointer  | n/a
const U *          |  8  | PKU | in pointer V       | pointer  | n/a
void *             |  8  | Pv  | inout pointer void | pointer  | n/a
void *             |  8  | Pv  | out pointer void   | pointer  | n/a
const void *       |  8  | PKv | in pointer void    | pointer  | n/a
const char *       |  8  | PKc | string             | string   | String

(*) not supported in GrCUDA or NFI
(+) does not support all values (NFI limitation)

## Synonymous NIDL types

Synonymous types are that can substituted without coercion:

- char, uint8
- bool, char8, uint8
- char16, uint16
- sint32, wchar
- char32, uint32t
- sint64, sll64
- uint64, ull64
- inout pointer V, out pointer U, synonymous(U, V), inout pointer void, out pointer void
- in pointer V, in pointer U, synonymous(U, V), in pointer void
