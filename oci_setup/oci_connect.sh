# Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NECSTLab nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#  * Neither the name of Politecnico di Milano nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/bin/bash

# Script used to connect to OCI instances.
# As we have to create new instances very often,
# the public key of new instances is updated,
# requiring us to delete the previous one we have.
# This script semplifies the process,
# so you can connect as "./oci_connect.sh"

# IP address of the OCI instance;
OCI_IP=152.67.254.100  # Some default IP, change it to whatever you have;
# Path to the .ssh folder;
SSH_FOLDER=~/.ssh
# Path to the private SSH key used to connect to OCI,
# relative to ${SSH_FOLDER};
PRIVATE_SSH_KEY_PATH=id_rsa

# Flags used to set debug (print commands),
# OCI IP, SSH folder and SSH private key path;
for arg in "$@"
do
    case $arg in
        -d|--debug)  # Debug flag;
        set -x
        shift
        ;;
        -i=*|--ip=*)  # OCI IP address;
        OCI_IP="${arg#*=}"
        shift
        ;;
        -s=*|--ssh_folder=*)  # SSH folder;
        SSH_FOLDER="${arg#*=}"
        shift
        ;;
        -k=*|--ssh_key=*)  # SSH key;
        PRIVATE_SSH_KEY_PATH="${arg#*=}"
        shift
        ;;
        *)  # Ignore other flags;
        shift
        ;;
    esac
done

# Remove the outdated public SSH key of the OCI instance;
ssh-keygen -f ${SSH_FOLDER}/known_hosts -R ${OCI_IP}
# Connect to the OCI instance (assuming a default Ubuntu installation);
ssh -i ${SSH_FOLDER}/${PRIVATE_SSH_KEY_PATH} -o StrictHostKeyChecking=no ubuntu@${OCI_IP}