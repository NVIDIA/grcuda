#!/bin/sh

mx build;
mkdir -p $GRAAL_HOME/jre/languages/grcuda;
cp mxbuild/dists/jdk1.8/grcuda.jar $GRAAL_HOME/jre/languages/grcuda/.;

