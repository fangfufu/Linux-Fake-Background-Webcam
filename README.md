# Linux-Fake-Background-Webcam

## Background
Video conferencing software support under Linux is relatively poor. The Linux
version of Zoom only supports background replacement via chroma key. The Linux
version of Microsoft Team does not support background blur.

Benjamen Elder wrote a
[blog post](https://elder.dev/posts/open-source-virtual-background/), describing
a background replacement solution using Python, OpenCV, Tensorflow and Node.js.
The scripts in Elder's blogpost do not work out of box. In this repository, I
tidied up his scripts, and provide a turn-key solution for creating a virtual
webcam with background replacement and additionally foreground object placement,
e.g. a podium. 

Rather than using GPU for acceleration as described by the original blog post, 
this version is CPU-only to avoid all the unnecessary complexities. By 
downscaling the image sent to bodypix neural network, and upscaling the 
received mask, this whole setup runs sufficiently fast under Intel i7-4900MQ. 

## Prerequisite
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

If you get an error like
```
OSError: [Errno 22] Invalid argument
```

when opening the webcam from Python, please install v4l2loopback from the [github](https://github.com/umlaeute/v4l2loopback) repo, 
as you could have an old version from your package manager.

## Installing with Docker
Please refer to [DOCKER.md](DOCKER.md). The updated Docker related files were
added by [liske](https://github.com/liske).

Using Docker is unnecessary. However it makes starting up and shutting down
the virtual webcam very easy and convenient. The only downside is that you
lose the ability to change background and foreground images on the fly.

## Installing without Docker
Please also make sure that your TCP port ``127.0.0.1:9000`` is free, as we will
be using it.

You need to have Node.js. Node.js version 12 is known to work. 

You will need Python 3. You need to have pip installed. Please make sure that 
you have installed the correct version pip, if you have both Python 2 and 
Python 3 installed. Please make sure that the command ``pip3`` runs.

I am assuming that you have set up your user environment properly, and when you
install Python packages, they will be installed locally within your home
directory.

You might want to add the following line in your ``.profile``. This line is
needed for Debian Buster.

    export PATH="$HOME/.local/bin":$PATH

### Installation
Run ``./install.sh``.

### Usage
You need to open two terminal windows. In one terminal window, do the following:

    cd bodypix
    node app.js

In the other terminal window, do the following:

    cd fakecam
    python3 fake.py

The files that you might want to replace are the followings:

  - ``fakecam/background.jpg`` - the background image
  - ``fakecam/foreground.jpg`` - the foreground image
  - ``fakecam/foreground-mask.jpg`` - the foreground image mask

If you want to change the files above in the middle of streaming, replace them
and press ``CTRL-C``
