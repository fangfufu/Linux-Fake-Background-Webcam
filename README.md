[![CodeFactor](https://www.codefactor.io/repository/github/fangfufu/linux-fake-background-webcam/badge)](https://www.codefactor.io/repository/github/fangfufu/linux-fake-background-webcam)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/9ab112be62e4472aa114181bcde1a885)](https://app.codacy.com/gh/fangfufu/Linux-Fake-Background-Webcam/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

# Linux-Fake-Background-Webcam

## Background
Video conferencing software support for background blurring and background
replacement under Linux is relatively poor. The Linux version of Microsoft
Team does not support background blur. Over at Webcamoid, we tried to figure out
if we can do these reliably using open source software
([issues/250](https://github.com/webcamoid/webcamoid/issues/250)).

This repository started of as a tidy up of Benjamen Elder's
[blog post](https://elder.dev/posts/open-source-virtual-background/). His
blogpost described a background replacement solution using Python, OpenCV,
Bodypix neural network, which is only available under Tensorflow.js. The scripts
in Elder's blogpost do not work out of box. This repository originally provided
a turn-key solution for creating a virtual webcam with background replacement
and additionally foreground object placement, e.g. a podium.

Over time this repository got strangely popular. However it has been clear over
time that Bodypix is slow and difficult to set up. Various users wanted to use
their GPU with Tensorflow.js, this does not always work. The extra code that
provided GPU support sometimes created problems for CPU-only users.

Recently Google released selfie segmentation support for
[Mediapipe](https://github.com/google/mediapipe/releases/tag/v0.8.5). This
repository has been updated to use Mediapipe for image segmentation. This
significantly increased the performance. The older version of this repository
is now stored in the ``bodypix`` branch.

The performance improvement introduced by
[2f7d698](https://github.com/fangfufu/Linux-Fake-Background-Webcam/commit/2f7d6988a3275b8aa4cbc73bed8151666c5aedef)
means that you can get at least 25FPS on an i7-4900MQ.

## Prerequisite
You need to install either v4l2loopback or akvcam. This repository was
originally written with v4l2loopback in mind. However, there has been report
that v4l2loopback does not work with certain versions of Ubuntu. Therefore 
support for akvcam has been added.

### v4l2loopback
The v4l2loopback kernel module can be installed through the package manager of
your Linux distribution or compiled from source following the instructions in
the [v4l2loopback github repository](https://github.com/umlaeute/v4l2loopback).

Once installed, the module needs to be loaded. This can be done manually for
the current session by running

```shell
sudo modprobe v4l2loopback devices=1 exclusive_caps=1 video_nr=2 card_label="fake-cam"
```

which will create a virtual video device `/dev/video2`, however, this will not
persist past reboot. (Note that the `exclusive_caps=1` option is required for
programs such as Zoom and Chrome).

To create the virtual video device on startup, run the `v4l2loopback-install.sh`
script to create `/etc/modules-load.d/v4l2loopback.conf` to load the module and
`/etc/modprobe.d/linux-fake-background.conf` to configure the module.

You can provide the video device number you want to use as an argument to this installation script.
For example, if you run `v4l2-ctl --list-devices` and get something like this:

```text
Integrated Camera: Integrated C (usb-0000:00:14.0-4):
	/dev/video0
	/dev/video1
	/dev/video2
	/dev/video3
	/dev/media0
	/dev/media1
```

You would want to run the command as follows to use the next available device number:

```shell
./v4l2loopback-install.sh 4
```

If you don't provide a video device number, the script will try to infer the next available device number for you.
This functionality requires `grep` with PCRE support and `bc`.

The camera will appear as `fake-cam` in your video source list.

If you get an error like
```
OSError: [Errno 22] Invalid argument
```

when opening the webcam from Python, please try the latest version of
v4l2loopback from the its
[GitHub repository](https://github.com/umlaeute/v4l2loopback), as the version
from your package manager may be too old.

### v4l2loopback-ctl

You can also use `v4l2loopback-ctl` to control virtual video device.

To add a virtual video device, use `sudo v4l2loopback-ctl add --exclusive-caps=1 --name="fake-cam" /dev/video2`.

To remove a virtual video device, use `sudo v4l2loopback-ctl delete /dev/video2`.

To list available virtual video devices, use `sudo v4l2loopback-ctl list`.

#### Ubuntu 18.04
If you are using Ubuntu 18.04, and if you want to use v4l2loopback, please
compile v4l2loopback from the source. You need to do the following:
1. Remove the ``v4l2loopback`` package
    - `sudo rmmod -r v4l2loopback`
    - `sudo apt-get remove v4l2loopback-dkms`
2. Install DKMS and the Linux kernel headers
    - ``sudo apt-get install dkms linux-headers-`uname -r` ``
3. Install v4l2loopback from the repository
    - `git clone https://github.com/umlaeute/v4l2loopback.git`
    - `cd v4l2loopback`
4. Install the module via DKMS
    - `sudo cp -R . /usr/src/v4l2loopback-1.1`
    - `sudo dkms add -m v4l2loopback -v 1.1`
    - `sudo dkms build -m v4l2loopback -v 1.1`
    - `sudo dkms install -m v4l2loopback -v 1.1`
5. Load the module
    - `sudo modprobe v4l2loopback`

This may apply for other versions of Ubuntu as well. For more information,
please refer to the following Github
[issue](https://github.com/jremmons/pyfakewebcam/issues/7#issuecomment-616617011).

### Akvcam
To install akvcam, you need to do the following:
1. Install the driver by following the instruction at
[Akvcam wiki](https://github.com/webcamoid/akvcam/wiki/Build-and-install). I
recommend installing and managing the driver via DKMS.
2. Configure the driver by copying ``akvcam`` to ``/etc/``. Please note that the
configuration file I supplied locks your Akvcam instance
to a resolution of 1280x720. It is different to the configuration file
automatically generated by Webcamoid, as my configuration locks the input/output
colour format of Akvcam.
3. Note down the output of ``ls /dev/video*``.
4. Run ``sudo modprobe akvcam`` or make the akvcam start with a system:
``sudo akvcam-install.sh``
5. Akvcam should have created two extra ``video`` devices.
6. When running ``lfbw.py``, you need to set ``-v`` to the first video device
that Akvcam created, e.g. if Akvcam created ``/dev/video5`` and ``/dev/video6``,
you need to set ``-v /dev/video5``.
7. The software that uses the virtual webcam should the second device that
Akvcam created, e.g. if Akvcam created ``/dev/video5`` and ``/dev/video6``,
you need to set the software to use ``/dev/video6``.

Note that in ``akvcam/config.ini``, ``Akvcam (Output device)`` is the  device
that ``lfbw.py`` outputs to, and ``Akvcam (Capture device)`` is the "capture
device", which is opened by the software that you want to use the virtual webcam
with.

You might have to specify the ``--no-ondemand`` flag when using Akvcam.

For more information on configuring Akvcam, please refer to
[Akvcam wiki](https://github.com/webcamoid/akvcam/wiki/Configure-the-cameras)

### Disabling UEFI Secure boot
Both v4l2loopback and Akvcam require custom kernel module. This might not be
possible if you have secure boot enabled. Please refer to your device
manufacturer's manual on disabling secure boot.

## Installation

Set up a virtual environment running Python >= 3.8 and <=3.11. You can use conda, pyenv, virtual-env or whatever you like.

Activate this virtual environment.

Mediapipe requires pip version 19.3 or above. (Please refer to [here](https://pypi.org/project/mediapipe/#files) and [here](https://github.com/pypa/manylinux)).
Upgrade pip by running:

```
python -m pip install --upgrade pip
```

Then clone the repository and install the software:

```
git clone https://github.com/fangfufu/Linux-Fake-Background-Webcam
cd Linux-Fake-Background-Webcam
python -m pip install --upgrade .
```

If pip complains about being unable to resolve `mediapipe`, it means you are running an unsupported Python version (<3.8 or >3.11). Mediapipe currently supports only Python 3.8-3.11.

### Installing with Docker
The use of Docker is no longer supported. I no longer see any reason for using
Docker with this software. However I have left behind the files
related to Docker, for those who want to fix Docker support.
Please also refer to [DOCKER.md](DOCKER.md). The Docker related files were
provided by [liske](https://github.com/liske).

Docker made starting up and shutting down the virtual webcam more convenient
for when Bodypix was needed. The ability to change background and foreground
images on-the-fly is unsupported when running under Docker.

## Usage

Inside the virtual environment in which you installed the software, simply run
```shell
lfbw
```

You configure it using a ini file, see `./config-example.ini`.
To run with a config file, use the following command:

```shell
lfbw -c ./config-example.ini
```

The files that you might want to replace are the followings:

  - ``background.jpg`` - the background image
  - ``foreground.jpg`` - the foreground image
  - ``foreground-mask.jpg`` - the foreground image mask

By default this program uses on-demand processing. The program processes images
from the real webcam only when there are programs using the fake webcam. If
there are no programs reading from the fake webcam, this program disconnects the
real webcam, pauses processing and outputs a black image at 1 FPS to reduce CPU
usage. You can manually toggle between the paused/unpaused state by pressing
``CTRL-C``. Unpausing the program also reload the files listed above. This
allows you to replace them without restarting the program. You can disable the
on-demand processing behaviour by specifying the ``--no-ondemand`` flag.

Note that animated background is supported. You can use any video file that can
be read by OpenCV.

### lfbw

`lfbw` supports the following options:

```
usage: lfbw    [-h] [-c CONFIG] [-W WIDTH] [-H HEIGHT] [-F FPS] [-C CODEC]
               [-w WEBCAM_PATH] [-v V4L2LOOPBACK_PATH] [--no-background]
               [-b BACKGROUND_IMAGE] [--tile-background] [--background-blur k]
               [--background-blur-sigma-frac frac] [--background-keep-aspect]
               [--no-foreground] [-f FOREGROUND_IMAGE]
               [-m FOREGROUND_MASK_IMAGE] [--hologram] [--no-ondemand]
               [--background-mask-update-speed BACKGROUND_MASK_UPDATE_SPEED]
               [--use-sigmoid] [--threshold THRESHOLD] [--no-postprocess]
               [--select-model SELECT_MODEL] [--cmap-person CMAP_PERSON]
               [--cmap-bg CMAP_BG]

Faking your webcam background under GNU/Linux. Please refer to:
https://github.com/fangfufu/Linux-Fake-Background-Webcam

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Config file (default: None)
  -W WIDTH, --width WIDTH
                        Set real webcam width (default: 1280)
  -H HEIGHT, --height HEIGHT
                        Set real webcam height (default: 720)
  -F FPS, --fps FPS     Set real webcam FPS (default: 30)
  -C CODEC, --codec CODEC
                        Set real webcam codec (default: MJPG)
  -w WEBCAM_PATH, --webcam-path WEBCAM_PATH
                        Set real webcam path (default: /dev/video0)
  -v V4L2LOOPBACK_PATH, --v4l2loopback-path V4L2LOOPBACK_PATH
                        V4l2loopback device path (default: /dev/video2)
  --no-background       Disable background image and blur the real background
                        (default: False)
  -b BACKGROUND_IMAGE, --background-image BACKGROUND_IMAGE
                        Background image path, animated background is
                        supported. (default: background.jpg)
  --tile-background     Tile the background image (default: False)
  --background-blur k   The gaussian bluring kernel size in pixels (default:
                        21)
  --background-blur-sigma-frac frac
                        The fraction of the kernel size to use for the sigma
                        value (ie. sigma = k / frac) (default: 3)
  --background-keep-aspect
                        Crop background if needed to maintain aspect ratio
                        (default: False)
  --no-foreground       Disable foreground image (default: False)
  -f FOREGROUND_IMAGE, --foreground-image FOREGROUND_IMAGE
                        Foreground image path (default: foreground.jpg)
  -m FOREGROUND_MASK_IMAGE, --foreground-mask-image FOREGROUND_MASK_IMAGE
                        Foreground mask image path (default: foreground-
                        mask.png)
  --hologram            Add a hologram effect. Shortcut for --selfie=hologram
                        (default: False)
  --selfie SELFIE       Foreground effects. Can be passed multiple time and
                        support the following effects: "hologram",
                        "solid=<N,N,N>", "cmap=<name>" and "blur=<N>"
                        (default: [])
  --no-ondemand         Continue processing when there is no application
                        using the virtual webcam (default: False)
  --background-mask-update-speed BACKGROUND_MASK_UPDATE_SPEED
                        The running average percentage for background mask
                        updates (default: 50)
  --use-sigmoid         Force the mask to follow a sigmoid distribution
                        (default: False)
  --threshold THRESHOLD
                        The minimum percentage threshold for accepting a pixel
                        as foreground (default: 75)
  --no-postprocess      Disable postprocessing (masking dilation and blurring)
                        (default: True)
  --select-model SELECT_MODEL
                        Select the model for MediaPipe. For more information,
                        please refer to https://github.com/fangfufu/Linux-
                        Fake-Background-
                        Webcam/issues/135#issuecomment-883361294 (default: 1)
  --cmap-person CMAP_PERSON
                        Apply colour map to the person using cmapy. Shortcut
                        for --selfie=cmap=<name>. For examples, please refer
                        to https://gitlab.com/cvejarano-
                        oss/cmapy/blob/master/docs/colorize_all_examples.md
                        (default: None)
  --cmap-bg CMAP_BG     Apply colour map to background using cmapy (default:
                        None)

--selfie=<filter> can be specified multiple times and accept a filter + its optional
arguments like --selfie=FILTER[=FILTER_ARGUMENTS].

Each filter is applied to the foreground (self) in the order they appear.
The following are supported:
- hologram: Apply an hologram effect?
- solid=<B,G,R>: Fill-in the foreground fowith the specific color
- cmap=<name>: Apply colour map <name> using cmapy
- blur=<N>: Blur (0-100)

Example:
lfbw.py --selfie=blur=30 --selfie=hologram # slightly blur and apply the hologram effect

Args that start with '--' (eg. -W) can also be set in a config file (specified
via -c). Config file syntax allows: key=value, flag=true, stuff=[a,b,c] (for
details, see syntax at https://goo.gl/R74nmi). If an arg is specified in more
than one place, then commandline values override config file values which
override defaults.
```

### Per-user systemd service

Modify `./systemd-user/lfbw_start_wrapper.sh` to suit your needs.
In particular, point to the correct `activate` shim for your virtual environment, and to the correct path to your config file.
Copy the file inside `$HOME/.local/bin` folder.
```
cp ./systemd-user/lfbw_start_wrapper.sh $HOME/.local/bin/
```

Copy the systemd service file from `systemd-user` folder to a location
suitable for user-defined systemd services (typically
`$HOME/.config/systemd/user`).

```
cp ./systemd-user/lfbw.service $HOME/.config/systemd/user/
```

To start the service and enable it so that it is run after login, run the
following (as normal user):
```
$ systemctl --user start lfbw
$ systemctl --user enable lfbw
```

Check that the process is running smoothly:
```
$ systemctl --user status lfbw
```

## License
The source code of this repository are released under GPLv3.

    Linux Fake Background Webcam
    Copyright (C) 2020-2024  Fufu Fang

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.


