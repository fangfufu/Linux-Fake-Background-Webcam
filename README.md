# Linux-Fake-Background-Webcam
## IMPORTANT UPDATES
I removed all the GPU related code and Docker related code, as the dependencies
are not that hard to satisfy, and the code now run fast enough without GPU.

## Background
Video conferencing software support under Linux is relatively poor. The Linux
version of Zoom only supports background replacement via chroma key. The Linux
version of Microsoft Team does not support background blur.

Benjamen Elder wrote a
[blog post](https://elder.dev/posts/open-source-virtual-background/), describing
a background replacement solution using Python, OpenCV, Tensorflow and Node.js.
The scripts in Elder's blogpost do not work out of box. In this repository, I
tidy up his scripts, and provide a turn-key solution for creating a virtual
webcam with background replacement.

This whole setup is relatively CPU intensive. However it has been tested under
Debian GNU/Linux, using Thinkpad T440p with an Intel i7-4900MQ CPU. The webcam
performs at an adequate frame rate.

## Prerequisite
You need to have Node.js. Node.js version 12 is known to work. You will also
need Python 3.

Please also make sure that your TCP port ``127.0.0.1:9000`` is free, as we will
be using it.

### Python 3
You need to have pip installed. Please make sure that you have installed the
correct version pip, if you have both Python 2 and Python 3 installed. Please
make sure that the command ``pip3`` runs.

I am assuming that you have set up your user environment properly, and when you
install Python packages, they will be installed locally within your home
directory.

You might want to add the following line in your ``.profile``. This line is
needed for Debian Buster.

    export PATH="$HOME/.local/bin":$PATH

### V4l2loopback
You need to install v4l2loopback. If you are on Debian Buster, you can do the
following:
    
    sudo apt install v4l2loopback-dkms

I added module options for v4l2loopback by creating
``/etc/modprobe.d/v4l2loopback.conf`` with the following content:

    options v4l2loopback devices=1  exclusive_caps=1 video_nr=2 card_label="v4l2loopback"
    
``exclusive_caps`` is required by some programs, e.g. Zoom and Chrome.
``video_nr`` specifies which ``/dev/video*`` file is the v4l2loopback device.
In this repository, I assume that ``/dev/video2`` is the virtual webcam, and
``/dev/video0`` is the physical webcam.

I also created ``/etc/modules-load.d/v4l2loopback`` with the following content:
    
    v4l2loopback
    
This automatically loads v4l2loopback module at boot, with the specified module
options.

## Installation
Run ``./install.sh``.

## Usage
You need to open two terminal windows. In one terminal window, do the following:

    cd bodypix
    node app.js

In the other terminal window, do the following:

    cd fakecam
    python3 fake.py


## Modification to Elder's original post
I removed all the Docker related nonsense.

I removed the ``hologram_effect()`` function, because I don't want the hologram
effect. I also corrected the command for launching the container instances - the
network communication between the container wasn't set up properly. I also
replaced his background image to something I took myself.

I scale down the image send to bodypix, and scale up the mask received from
bodypix. I dilated the mask further to compensate for the scaling. This made the
whole thing to run fast enough using only CPU.
