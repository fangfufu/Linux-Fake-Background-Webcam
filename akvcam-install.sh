#!/bin/bash
# Create the modprobe files for akvm

LOAD_FILE="/etc/modules-load.d/akvcam.conf"


# check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi

# start the module if not started
echo "Starting the module..."
sudo modprobe akvcam

echo "Make the module start on a system startup..."

# create load file
if [ -f $LOAD_FILE ]; then
    echo "File exists: ${LOAD_FILE}"
else
    echo "akvcam" > $LOAD_FILE
    echo "created: ${LOAD_FILE}"
fi

