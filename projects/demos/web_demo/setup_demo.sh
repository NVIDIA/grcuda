#!/bin/bash

## This script assumes that the current machine was set up using the 
## setup_machine_from_scratch.sh script

echo "Building GrCUDA"
cd ../../../
./install.sh

cd -

# Base Dependency Install 
sudo apt-get install cmake libopencv-dev -y

# For grcuda-data repo
echo "Initializing and downloading GrCUDA Data store repo"
git submodule init
cd ../../../grcuda-data
git submodule update --remote

cd -

# Create symbolic link for the images
echo "Creating symbolic link for the images"
cd frontend
ln -s ../../../../grcuda-data/datasets/web_demo/images images
cd -

# Compile cuda binary
echo "Compiling CUDA binary"
mkdir ../image_pipeline/cuda/build
cd ../image_pipeline/cuda/build
cmake ..
make

cd -

# Build backend 
echo "Building and running backend"
cd backend 
npm i
npm run build
npm run runall &

# Run frontend
echo "Starting frontend"
python3 -m http.server 8085 --directory ../frontend



