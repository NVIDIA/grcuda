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