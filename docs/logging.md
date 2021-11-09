# LOGGING

Support for logging in Truffle languages and instruments is made by the TruffleLogger class.
Different levels of logging are provided by the Level class, to differenciate the importance of occurring errors or warnings. This gives the possibility to decide up to which severity it is convenient to have them printed on the terminal. It is also possible to print them on a file.

## LEVELS

The logging Level objects are ordered and are specified by ordered integers. Enabling logging at a given level also enables logging at all higher levels.
The levels in descending order are:
- **SEVERE** (highest value)
- **WARNING**
- **INFO**
- **CONFIG**
- **FINE**
- **FINER**
- **FINEST** (lowest value)
In addition there is a level **OFF** that can be used to turn off logging, and a level **ALL** that can be used to enable logging of all messages.
[TruffleLogger](https://www.graalvm.org/truffle/javadoc/com/oracle/truffle/api/TruffleLogger.html) and [Level](https://docs.oracle.com/javase/7/docs/api/java/util/logging/Level.html) class are already implemented in Java. Click for further information.

## LOGGERS IN THE CODE

### AVAILABLE LOGGERS

GrCUDA is characterized by the presence of several different types of loggers, each one with its own functionality. The GrCUDALogger class is implemented to have access to loggers of interest when specific features are needed.
Main examples of loggers in GrCUDALogger follow:
- **GRCUDA_LOGGER** : all the logging action in grcuda can refer to this principal logger;

```java
public static final String GRCUDA_LOGGER = "com.nvidia.grcuda";
```

- **RUNTIME_LOGGER** : referral for each logging action in runtime project of grcuda;

```java
public static final String RUNTIME_LOGGER = "com.nvidia.grcuda.runtime";
```

- **EXECUTIONCONTEXT_LOGGER** : referral for each logging action in exectution context project of runtime;

```java
public static final String EXECUTIONCONTEXT_LOGGER = "com.nvidia.grcuda.runtime.executioncontext";
```

If further loggers are needed to be implemented, it can be easily done by adding them to the GrCUDALogger class, being sure of respecting the name convention, like in the following example.

```java
public static final String LOGGER_NAME = "com.nvidia.grcuda.file_name";
```

### USING AVAILABLE LOGGERS

To use the available loggers in the code, follow the instructions below:
1. create the specific logger in the project's class as TruffleLogger type object.

```java
public static final TruffleLogger LOGGER_NAME = GrCUDALogger.getLogger(GrCUDALogger.LOGGER_NAME);
```
2. set the *logger_level* to the message (severe, warning, info...).

```java
LOGGER_NAME.logger_level("message");
```

As alternative of step 2. it is also possible to directly associate logging level to messages by using the following form:

```java
GrCUDALogger.getLogger(GrCUDALogger.LOGGER_NAME).logger_level("*message*");
```

## LOGGERS CONFIGURATION

All loggers are set to level INFO by default.
It is possible to modify the level of all the messages in a file with graal options from the command line.
In particular, it is possible to specify a unique output file for all the logger messages.
Set the *path_to_file* (see examples below).

```java
-- log.file=path_to_file
```
It is also possible to specify the *logger_level* for each logger (see all possible levels above).

```java
--log.grcuda.com.nvidia.grcuda.file_name.level=logger_level
```

Some examples of loggers usage, starting from benchmark b1 (all the options are set to the default value):
- sets all the loggers of grcuda to ALL, printed on screen as StandardOutput.
```java
graalpython --jvm --polyglot --log.grcuda.com.nvidia.grcuda.level=ALL benchmark_main.py -d -b b1
```
- sets all the loggers of grcuda to ALL, saved on file b1.log in the same folder from which the command is launched.
```java
graalpython --jvm --polyglot --log.grcuda.com.nvidia.grcuda.level=ALL --log.file=./b1.log benchmark_main.py -d -b b1
```
- set all the loggers of grcuda.runtime to ALL and all the other loggers of grcuda to OFF, saved on file b1.log in the root folder of grcuda *GRCUDA_HOME*.
```java
graalpython --jvm --polyglot --log.grcuda.com.nvidia.grcuda.level=OFF --log.grcuda.com.nvidia.grcuda.runtime.level=ALL --log.file=$GRCUDA_HOME/b1.log benchmark_main.py -d -b b1
```
