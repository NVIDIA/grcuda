#!/bin/bash

# Installation flags;
INSTALL_CUML=false
INSTALL_RECENT_CMAKE=false

# basic update on a newly created machine;
sudo apt update
sudo apt upgrade -y
# library needed later to run: gu rebuild-images polyglot and setting up graalpython;
sudo apt install build-essential -y
sudo apt install lib32z1-dev -y
sudo apt install unzip -y
sudo apt-get install -y python-ctypes

# clone repositories (GraalVM, MX, GrCUDA).
#   We use the freely available GraalVM CE.
#   At the bottom of this guide, it is explained how to install EE;
git clone https://github.com/oracle/graal.git
git clone https://github.com/graalvm/mx.git
git clone https://github.com/AlbertoParravicini/grcuda.git

# checkout commit of GraalVM corresponding to the release (21.1);
cd graal
git checkout  192eaf62331679907449ee60dad9d6d6661a3dc8
cd ..

# checkout commit of mx compatible with versions of other tools;
cd mx
git checkout dcfdd847c5b808ffd1a519713e0598242f28ecd1
cd ..

# download the GraalVM release build (21.1.0) and the corresponding JVM;
wget https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-21.1.0/graalvm-ce-java11-linux-amd64-21.1.0.tar.gz
wget https://github.com/graalvm/labs-openjdk-11/releases/download/jvmci-21.1-b05/labsjdk-ce-11.0.11+8-jvmci-21.1-b05-linux-amd64.tar.gz
# extract them;
tar xfz labsjdk-ce-11.0.11+8-jvmci-21.1-b05-linux-amd64.tar.gz
tar xfz graalvm-ce-java11-linux-amd64-21.1.0.tar.gz
# remove temporary files;
rm labsjdk-ce-11.0.11+8-jvmci-21.1-b05-linux-amd64.tar.gz
rm graalvm-ce-java11-linux-amd64-21.1.0.tar.gz

# install CUDA and Nvidia drivers;

# -> option 1 (more automatic, but possibly outdated);
# sudo apt install nvidia-cuda-toolkit -y
# sudo apt install ubuntu-drivers-common -y
# sudo ubuntu-drivers autoinstall
# -> option 2 (from Nvidia's website).
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# symlink for python (use it with care! some system tools require Python 2.7);
# sudo ln -s /usr/bin/python3 /usr/bin/python

# update ~/.bashrc with new variables;
echo '' >> ~/.bashrc
echo '########## GrCUDA Configuration ##########' >> ~/.bashrc
echo '' >> ~/.bashrc
echo '# CUDA;' >> ~/.bashrc
echo 'export CUDA_DIR=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_DIR/bin:$PATH' >> ~/.bashrc
echo '# GraalVM and GrCUDA;' >> ~/.bashrc
echo 'export PATH=~/mx:$PATH' >> ~/.bashrc
echo 'export JAVA_HOME=~/labsjdk-ce-11.0.11-jvmci-21.1-b05' >> ~/.bashrc
echo 'export GRAAL_HOME=~/graalvm-ce-java11-21.1.0' >> ~/.bashrc
echo 'export PATH=$GRAAL_HOME/bin:$PATH' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export GRCUDA_HOME=~/grcuda' >> ~/.bashrc
echo '' >> ~/.bashrc
echo '##########################################' >> ~/.bashrc
# reload  ~/.bashrc;
source  ~/.bashrc

# setup GraalVM;
gu install native-image
gu install llvm-toolchain
gu install python
gu install nodejs
gu rebuild-images polyglot

# create environment for Graalpython and set it up;
graalpython -m venv ~/graalpython_venv
source ~/graalpython_venv/bin/activate
graalpython -m ginstall install setuptools
graalpython -m ginstall install Cython
graalpython -m ginstall install numpy

# install miniconda (Python is required to build with mx);
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 777 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
$HOME/miniconda/bin/conda init

# optional: install cuML;
if [ "$INSTALL_CUML" = true ] ; then
    $HOME/miniconda/bin/conda create -n rapids-21.08 -c rapidsai -c nvidia -c conda-forge cuml=21.08 python=3.8 cudatoolkit=11.2 -y
    echo 'export LIBCUML_DIR=/home/ubuntu/miniconda/envs/rapids-21.08/lib/' >> ~/.bashrc
    source  ~/.bashrc
fi

# optional: install TensorRT - Currently not supported, it does not work with CUDA 11.4;

# Install a recent version of CMake, following https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line;
if [ "$INSTALL_RECENT_CMAKE" = true ] ; then
    sudo apt remove --purge --auto-remove cmake -y
    sudo apt update && sudo apt install -y software-properties-common lsb-release && sudo apt clean all
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
    sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
    sudo apt update -y
    sudo apt install kitware-archive-keyring -y
    sudo rm /etc/apt/trusted.gpg.d/kitware.gpg
    sudo apt update
    sudo apt install cmake -y
fi

# Installing TensorRT cannot be automated.
# Go to https://developer.nvidia.com/tensorrt, click "Get Started", login to an Nvidia account, download it to a local machine and upload it here;
# sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.1.6-ga-20210626_1-1_amd64.deb
# sudo apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.1.6-ga-20210626/7fa2af80.pub
# sudo apt-get update
# sudo apt-get install tensorrt

# reboot the machine to load the Nvidia drivers;
sudo reboot

##########################################
##########################################

# # alternative: install GraalVM EE.
# # 1. go to https://www.oracle.com/downloads/graalvm-downloads.html?selected_tab=1
# # 2. download Oracle GraalVM Enterprise Edition Core for Java 11. An Oracle account is required
# # 3. transfer the tar.gz to your machine, and extract it with
# tar -xzf graalvm-ee-java11-linux-amd64-21.2.0.1.tar.gz
# rm graalvm-ee-java11-linux-amd64-21.2.0.1.tar.gz
# # Install Labs-JDK to build GrCUDA;
# wget https://github.com/graalvm/labs-openjdk-11/releases/download/jvmci-21.2-b08/labsjdk-ce-11.0.12+6-jvmci-21.2-b08-linux-amd64.tar.gz
# tar -xzf labsjdk-ce-11.0.12+6-jvmci-21.2-b08-linux-amd64.tar.gz
# rm labsjdk-ce-11.0.12+6-jvmci-21.2-b08-linux-amd64.tar.gz
# # checkout commit of GraalVM corresponding to the release (21.1);
# cd graal
# git checkout e9c54823b71cdca08e392f6b8b9a283c01c96571 # 21.2:
# cd ..
# # checkout commit of mx compatible with versions of other tools;
# cd mx
# git checkout b39c4a551c4e99909f2e83722472329324ed4e42
# cd ..
# echo '# GraalVM EE' >> ~/.bashrc
# echo 'export JAVA_HOME=~/labsjdk-ce-11.0.12-jvmci-21.2-b08' >> ~/.bashrc
# echo 'export GRAAL_HOME=~/graalvm-ee-java11-21.2.0.1' >> ~/.bashrc
# echo 'export PATH=$GRAAL_HOME/bin:$PATH' >> ~/.bashrc
# echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
# # Install components. Note: this requires user interaction, and an email address associated to an Oracle account
# gu install native-image
# gu install llvm-toolchain
# gu install python
# gu install nodejs
# gu rebuild-images polyglot
