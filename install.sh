#!/bin/sh

mx build;

# Java 8;
mkdir -p $GRAAL_HOME/jre/languages/grcuda;
cp mxbuild/dists/jdk1.8/grcuda.jar $GRAAL_HOME/jre/languages/grcuda/.;

# Java 11;
mkdir -p $GRAAL_HOME/languages/grcuda;
cp mxbuild/dists/jdk11/grcuda.jar $GRAAL_HOME/languages/grcuda/.;
