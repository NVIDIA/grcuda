#!/bin/sh

mx build;

# Install for Java 11;
mkdir -p $GRAAL_HOME/languages/grcuda;
cp mxbuild/dists/jdk11/grcuda.jar $GRAAL_HOME/languages/grcuda/.;
