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