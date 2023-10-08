#!/bin/bash
# Create the modprobe files for v4l2loopback

LOAD_FILE="/etc/modules-load.d/v4l2loopback.conf"
OPT_FILE="/etc/modprobe.d/linux-fake-background.conf"
VIDEO_NR=2

# If parameter supplied, use that
if [ "$#" -ne 0 ]; then
 VIDEO_NR="$1"
fi

#echo "VIDEO_NR = ${VIDEO_NR}"
#OPT_FILE2="./testfile.txt"
#echo "options v4l2loopback devices=1 exclusive_caps=1 video_nr=${VIDEO_NR} card_label=\"fake-cam\"" > $OPT_FILE2
#exit 1

# check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi

# create load file
if [ -f $LOAD_FILE ]; then
    echo "File exists: ${LOAD_FILE}"
else
    echo "v4l2loopback" > $LOAD_FILE
    echo "created: ${LOAD_FILE}"
fi

# create options file and load the changes
if [ -f $OPT_FILE ]; then
    echo "file exists: ${OPT_FILE}, no changes have been made"
else
    echo "options v4l2loopback devices=1 exclusive_caps=1 video_nr=${VIDEO_NR} card_label=\"fake-cam\"" > $OPT_FILE
    echo "created: ${OPT_FILE}"
    echo "reloading kernel modules..."
    systemctl restart systemd-modules-load.service
    echo "..done"
fi
