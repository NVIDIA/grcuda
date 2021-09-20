#!/bin/bash

# This script must be executed from the OCI console to terminate a running instance,
# while preserving the boot volume for later use. 
# The script comes in handy as GPU instances are paid even when shut down,
# so it common to terminate them after use, while preserving the boot volume so it can be reused;

# The name of the instance to terminate is passed as the first argument of the script;
DISPLAY_NAME=$1

# OCID of the compartment where the instance is, substitute with OCID of your compartment.
# You can specify it as optional parameter, otherwise a default value is used;
if [ -z "$2" ]
then
      COMPARTMENT_ID=ocid1.compartment.oc1.your.comparment.ocid
else
      COMPARTMENT_ID=$2
fi

# Get instance id;
INSTANCE_ID=$(oci compute instance list -c $COMPARTMENT_ID --lifecycle-state RUNNING --display-name $DISPLAY_NAME --query data[0].id --raw-output)

if [ -z "$INSTANCE_ID" ]
then
    echo "INSTANCE_ID not found using DISPLAY_NAME=${DISPLAY_NAME} and COMPARTMENT_OCID=${COMPARTMENT_ID}"; exit -1; 
fi

# Print info (name, id) about the instance to terminate;
echo display-name=${DISPLAY_NAME}
echo comparment-ocid=${COMPARTMENT_ID}
echo id=$INSTANCE_ID

# Terminate instance (the terminate command automatically asks for confirmation).
# Set --preserve-boot-volume to false if you want to permanently erase the boot volume attached to the instance;
oci compute instance terminate --instance-id $INSTANCE_ID --preserve-boot-volume true --wait-for-state TERMINATED