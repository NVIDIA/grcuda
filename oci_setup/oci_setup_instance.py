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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to simplify the creation of GPU instances on OCI.
The script must be used inside your OCI console.
You can specify an existing boot volume, an existing public IP to use, 
and a public key in your possession to use for login.
Settings can be specified through a separate JSON file,
whose keys are the same as specified in the CONFIG section of this script;

Created on Mon Jan 25 11:43:34 2021
@author: aparravi
"""

import json
import os
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import shutil

##############################
# CONFIG #####################
##############################

# Here you can specify some default values, 
# if you don't want to specify them through a separate file;
CONFIG = dict(
    REGION = "rNHZ:US-SANJOSE-1-AD-1",
    VM = "VM.Standard.E2.1",
    NUM_GPUS = 0,
    PUBLIC_IP = "152.67.254.100",
    # OCID of the Compartment;
    COMPARTMENT = "ocid1.compartment.oc1.your.comparment.ocid",
    # OCID of the Public Subnet;
    SUBNET = "ocid1.subnet.oc1.us-sanjose-1.your.subnet.ocid",
    # OCID of the Boot Volume;
    BOOT_VOLUME = "ocid1.bootvolume.oc1.us-sanjose-1.your.bootvolume.ocid",
    # Public key employed when creating the instance the first time (with a "fresh" Boot Volume).
    # If you use a Boot Volume created by another user, make sure to add your public key to ~/.ssh/authorized_keys
    SSH_KEY = "ssh-rsa your-key",
)
##############################
# SETUP ######################
##############################

DEBUG = False

# Map GPU number to default instance shapes;
NUM_GPU_TO_SHAPE = {
    0: CONFIG["VM"],
    1: "VM.GPU3.1",
    2: "VM.GPU3.2",
    4: "VM.GPU3.4",
    8: "BM.GPU3.8",
}

DEFAULT_SETUP_JSON = """
{{
  "compartmentId": "{}",
  "sourceBootVolumeId": "{}",
  "sshAuthorizedKeys": "{}",
  "subnetId": "{}",
  "assignPublicIp": false
}}
"""

# Temporary directory where data are stored;
DEFAULT_TEMP_DIR = "tmp_oci_setup"

# OCI commands;
OCI_LAUNCH_INSTANCE = "oci compute instance launch --from-json file://{} --wait-for-state RUNNING"
OCI_OBTAIN_VNIC = "oci compute instance list-vnics --limit 1 --instance-id {}"
OCI_OBTAIN_PRIVATE_IP = "oci network private-ip list --vnic-id {}"
OCI_OBTAIN_PUBLIC_IP = "oci network public-ip get --public-ip-address {}"
OCI_UPDATE_PUBLIC_IP = "oci network public-ip update --public-ip-id {} --private-ip-id {}"

##############################
##############################

def log_message(message: str) -> None:
    date = datetime.now()
    date_str = date.strftime("%Y-%m-%d-%H-%M-%S-%f")
    print(f"[{date_str} oci-setup] {message}")


def parse_shape_name(shape: str) -> str:
    if shape == CONFIG["VM"]:
        return "cpu-default"
    elif "gpu" in shape.lower():
        gpu_count = shape.split(".")[-1]
        return f"gpu-{gpu_count}"
    else:
        return shape.replace(".", "-")


def create_instance_launch_dict(shape: str, json_setup: str, debug: bool=DEBUG) -> dict:
    instance_launch_dict = json.loads(json_setup)
    # Add region;
    instance_launch_dict["availabilityDomain"] = CONFIG["REGION"]
    # Add shape;
    instance_launch_dict["shape"] = shape
    # Create hostname and display name;
    hostname = display_name = parse_shape_name(shape)
    instance_launch_dict["hostname"] = f"grcuda-{hostname}"
    instance_launch_dict["displayName"] = f"grcuda-{display_name}"
    if debug:
        log_message(instance_launch_dict)
    return instance_launch_dict


def run_oci_command(command_template: str, *command_format_args, debug: bool=DEBUG) -> dict:
    # Setup the OCI command;
    oci_command = command_template.format(*command_format_args)
    if debug:
        log_message(f"launching OCI command: {oci_command}")
    # Launch the OCI command;
    try:
        result = subprocess.run(oci_command, shell=True, env=os.environ, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        if debug:
            log_message(f"caught exception {e.output} during OCI command")
        exit(-1)
    if result.stderr:
        if debug:
            log_message("OCI command completed with error")
            log_message(result.stderr)
        exit(-1)
    # Everything is ok, we extract the result as a dictionary.
    # There might be other stuff printed along with the JSON, so we have to remove it;
    res_tmp = result.stdout.decode("utf-8")
    res_tmp = res_tmp[res_tmp.index("{"):]  # Delete everything until the first "{";
    res_tmp = res_tmp[:-res_tmp[::-1].index("}")]  # Delete everything after the last "}";
    return json.loads(res_tmp)


def launch_instance(instance_launch_dict: dict, debug: bool=DEBUG) -> str:
    # We have to store the dictionary to a temporary file;
    launch_json_file_name = os.path.join(DEFAULT_TEMP_DIR, instance_launch_dict["displayName"] + ".json") 
    if debug:
        log_message(f"storing temporary launch JSON into {launch_json_file_name}")
    # Create temporary folder;
    Path(DEFAULT_TEMP_DIR).mkdir(parents=True, exist_ok=True)
    # Store dictionary to JSON;
    with open(launch_json_file_name, "w") as f:
        json.dump(instance_launch_dict, f)
    # Setup the launch command;
    result = run_oci_command(OCI_LAUNCH_INSTANCE, launch_json_file_name, debug=debug)
    # Extract the instance OCID for later use;
    instance_ocid = result["data"]["id"]
    if debug:
        log_message(f"created instance with OCID={instance_ocid}")

    # Remove the temporary configuration file;
    os.remove(launch_json_file_name)
    if len(os.listdir(DEFAULT_TEMP_DIR)) == 0: 
        shutil.rmtree(DEFAULT_TEMP_DIR)  # Remove the folder if it is empty;

    return instance_ocid


def attach_reserved_public_ip(instance_ocid: str, debug: bool=DEBUG) -> None:
    # We have to obtain the VNIC attached to the instance (assume only 1 VNIC is available);
    result = run_oci_command(OCI_OBTAIN_VNIC, instance_ocid, debug=debug)
    # Extract the VNIC OCID;
    vnic_ocid = result["data"][0]["id"]
    if debug:
        log_message(f"obtained VNIC with OCID={vnic_ocid}")
    # Obtain the private address OCID associated to the VNIC;
    result = run_oci_command(OCI_OBTAIN_PRIVATE_IP, vnic_ocid, debug=debug)
    # Extract the private IP OCID;
    private_ip_ocid = result["data"][0]["id"]
    if debug:
        log_message(f"obtained private IP with OCID={private_ip_ocid}")
    # Obtain the public IP OCID;
    result = run_oci_command(OCI_OBTAIN_PUBLIC_IP, CONFIG["PUBLIC_IP"], debug=debug)
    # Extract the VNIC OCID;
    public_ip_ocid = result["data"]["id"]
    if debug:
        log_message(f"obtained public IP with OCID={public_ip_ocid}")
    # Assign the reserved public IP;
    run_oci_command(OCI_UPDATE_PUBLIC_IP, public_ip_ocid, private_ip_ocid, debug=debug)
    if debug:
        log_message(f"assigned public IP {CONFIG['PUBLIC_IP']}")


def update_config_with_json(json_path: str, debug: bool=DEBUG) -> None:
    try:
        if debug:
            log_message(f"loading configuration file {json_path}")
        with open(json_path) as f:
            json_config = json.load(f)
            for k in CONFIG.keys():
                if k in json_config:
                    CONFIG[k] = json_config[k]
    except Exception as e:
        log_message(f"warning: failed to load configuration file {json_path}, using default values")
        log_message(f"  encountered exception: {e}")

##############################
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup OCI instances from the command line")
    parser.add_argument("-d", "--debug", action="store_true", help="If present, print debug messages", default=DEBUG)
    parser.add_argument("-g", "--num_gpus", metavar="N", type=int, default=CONFIG["NUM_GPUS"], help="Number of GPUs present in the instance")
    parser.add_argument("-p", "--print_config", action="store_true", help="If present, print the configuration options provided as input to the setup", default=False)
    parser.add_argument("-j", "--json", type=str, help="Load the configuration provided in the following JSON")

    # 1. Parse the input arguments;
    args = parser.parse_args()
    debug = args.debug
    json_path = args.json
    if json_path:
        update_config_with_json(json_path, debug)  # Try loading the JSON configuraiton, if specified;
    num_gpus = args.num_gpus

    if debug and args.print_config:
        log_message(f"provided input configuration:")
        for k, v in CONFIG.items():
            log_message(f"  {k}" + ("\t\t" if len(k) < 7 else "\t") + f"= {v}")

    # 2. Select shape;
    NUM_GPU_TO_SHAPE[0] = CONFIG["VM"]
    if num_gpus in NUM_GPU_TO_SHAPE:
        shape = NUM_GPU_TO_SHAPE[num_gpus]
    else:
        shape = NUM_GPU_TO_SHAPE[0]
    if debug:
        log_message(f"using {num_gpus} GPUs")
        log_message(f"selected shape {shape}")
    json_setup = DEFAULT_SETUP_JSON.format(CONFIG["COMPARTMENT"], CONFIG["BOOT_VOLUME"], CONFIG["SSH_KEY"], CONFIG["SUBNET"])

    # 3. Obtain configuration dictionary;
    instance_launch_dict = create_instance_launch_dict(shape, json_setup, debug)

    # 4. Launch the instance;
    instance_id = launch_instance(instance_launch_dict, debug)

    # 5. Attach the reserved public IP to the instance;
    attach_reserved_public_ip(instance_id, debug)

    if debug:
        log_message("setup completed successfully!")
