#!/bin/bash

## This script assumes that the current machine was set up using the 
## setup_machine_from_scratch.sh script

echo "Building GrCUDA"
cd ../../
./install.sh

cd -

# Base Dependency Install 
sudo apt-get install cmake libopencv-dev -y

# For grcuda-data repo
echo "Initializing and downloading GrCUDA Data store repo"
git submodule init
cd ../../grcuda-data
git submodule update --remote

cd -

# Create symbolic link for the images
echo "Creating symbolic link for the images"
cd frontend/images
ln -s ../../../../grcuda-data/datasets/web_demo/images/dataset512 dataset512
ln -s ../../../../grcuda-data/datasets/web_demo/images/dataset1024 dataset1024
mkdir full_res 
mkdir thumb
cd -

# Compile cuda binary
echo "Compiling CUDA binary"
mkdir ../image_pipeline_local/cuda/build
cd ../image_pipeline_local/cuda/build
cmake ..
make

cd -

# Build backend 
echo "Building and running backend"
cd backend 
npm i
npm run build


