#!/bin/sh

mx build;

# Install for Java 8+;
mkdir -p $GRAAL_HOME/languages/grcuda;
cp $GRCUDA_HOME/mxbuild/dists/jdk1.8/grcuda.jar $GRAAL_HOME/languages/grcuda/.;
