# Linux-Fake-Background-Webcam
Video conferencing software support under Linux is relatively poor. The Linux version of Zoom only supports background replacement via chroma key. The Linux version of Microsoft Team does not support background blur. 

Benjamen Elder wrote a [blog post](https://elder.dev/posts/open-source-virtual-background/), describing a background replacement solution using Python, OpenCV, Tensorflow and Node.js. The scripts in Elder's blogpost do not work out of box. In this repository, I tidy up his scripts, and provide a turn-key solution for creating a virtual webcam with background replacement. 

This whole setup has been tested under Debian GNU/Linux, using Thinkpad T440p with an Intel i7-4900MQ CPU. 

## Prerequisite
### Docker
You need to set up Docker. If you want GPU acceleration, you want to set up [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-docker). There are a lot of guides on the Internet on how to set these up. I am not going to describe them here. Note that if you don't manage to set up Nvidia Container Toolkit, Tensorflow will fall back to CPU. This is in fact how I run mine - my GPU is too old for the current version of Tensorflow. 

### V4l2loopback
You need to install v4l2loopback. If you are on Debian Buster, you can do the following:
    
    sudo apt install v4l2loopback-dkms

I added module options for v4l2loopback by creating ``/etc/modprobe.d/v4l2loopback.conf`` with the following content: 

    options v4l2loopback devices=1  exclusive_caps=1 video_nr=2 card_label="v4l2loopback"
    
``exclusive_caps`` is required by some programs, e.g. Zoom and Chrome. ``video_nr`` specifies which ``/dev/video*`` file is the v4l2loopback device. In this repository, I assume that ``/dev/video2`` is the virtual webcam, and ``/dev/video0`` is the physical webcam. 

I also created ``/etc/modules-load.d/v4l2loopback`` with the following content:
    
    v4l2loopback
    
This automatically loads v4l2loopback module at boot, with the specified module options.

## How to use this repository
### Setting up
1. Replace ``fakecam/background.jpg`` with your own background image. 
2. Build the Docker image by running ``./build.sh``.
3. Create the Docker network bridge and containers by running ``./create-container.sh``.

### Using the virtual webcam
4. Run the containers by running ``./run-containers.sh``. 
5. Do whatever you need to do, remember that the virtual webcam is at ``/dev/video2``, and your physical webcam should be at ``/dev/video0``. You cannot access the physical webcam when the containers are running. 
6. Stop the containers by running ``./stop-containers.sh``. 

### Remove the containers
7. If you want to remove the containers, run ``./remove-containers.sh``.
8. If you want to remove the Docker images, run ``./remove-images.sh``.

### If you want to change the background image
If you want to change your background image, do these steps: 1, 2, 7, 3.

### If your bodypix container crashes due to GPU related packages
Please switch to the ``cpu-only`` and try again. 

## Modification to Elder's original post
I removed the ``hologram_effect()`` function, because I don't want the hologram effect. I also corrected the command for launching the container instances - the network communication between the container wasn't set up properly. I also replaced his background image to something I took myself. 

## Afterthoughts
This whole set up is just so amazingly ridiculous - I cannot believe we are running two container instances with network communication, just for a simple webcam background removal. Someone please make a simpler version... 
