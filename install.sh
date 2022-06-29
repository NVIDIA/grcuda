#!/bin/sh
mx build;

# Install for Java 8+;
mkdir -p $GRAAL_HOME/languages/grcuda;
cp $GRCUDA_HOME/mxbuild/dists/jdk1.8/grcuda.jar $GRAAL_HOME/languages/grcuda/.;
cp $GRCUDA_HOME/mxbuild/dists/grcuda.jar $GRAAL_HOME/languages/grcuda/.;

# Compute interconnection graph (connection_graph.csv)
cd $GRCUDA_HOME/projects/resources/connection_graph
./run.sh
cd $GRCUDA_HOME
