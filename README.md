[![CodeFactor](https://www.codefactor.io/repository/github/fangfufu/linux-fake-background-webcam/badge)](https://www.codefactor.io/repository/github/fangfufu/linux-fake-background-webcam)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/9ab112be62e4472aa114181bcde1a885)](https://app.codacy.com/gh/fangfufu/Linux-Fake-Background-Webcam/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

# Linux-Fake-Background-Webcam
> [!WARNING]
> Please read the "Installation of the Python package" section carefully for
> the installation on system with Python 3.13 and above. This includes
> **Debian 13 "Trixie"**

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

## Installation of the Python package
This had been tested under Debian 13 "Trixie". 

You need pyenv because Mediapipe does not support Python 3.13 yet! Debian 13
ships with Python 3.13. 
```
sudo apt-get install pyenv
```

Add the following to your `.profile`:
```
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
```

Log out, then log back in, then install Python 3.12 locally:
```
pyenv install 3.12
```

Clone this repository:
```
git clone https://github.com/fangfufu/Linux-Fake-Background-Webcam
```

Set up a Python virtual environment inside the downloaded repository: 
```
cd Linux-Fake-Background-Webcam
```

Set local Python version to 3.12:
```
pyenv local 3.12
```

Set up a Python virtual environment:
```
python -m venv venv
```

Activate the virtual environment:
```
source venv/bin/activate
```

Install the package
```
pip install --upgrade .
```

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
usage: lfbw.py [-h] [-c CONFIG] [-W WIDTH] [-H HEIGHT] [-F FPS] [-C CODEC] [-w WEBCAM_PATH] [-v V4L2LOOPBACK_PATH] [--selfie SELFIE] [--background BACKGROUND] [--mask MASK] [--no-ondemand]
               [--use-sigmoid] [--threshold THRESHOLD] [--no-postprocess] [--select-model SELECT_MODEL] [--dump]

Faking your webcam background under GNU/Linux. Please refer to: https://github.com/fangfufu/Linux-Fake-Background-Webcam

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Config file
  -W WIDTH, --width WIDTH
                        Set real webcam width
  -H HEIGHT, --height HEIGHT
                        Set real webcam height
  -F FPS, --fps FPS     Set real webcam FPS
  -C CODEC, --codec CODEC
                        Set real webcam codec
  -w WEBCAM_PATH, --webcam-path WEBCAM_PATH
                        Set real webcam path
  -v V4L2LOOPBACK_PATH, --v4l2loopback-path V4L2LOOPBACK_PATH
                        V4l2loopback device path
  --selfie SELFIE       Selfie effects (comma-separated)
  --background BACKGROUND
                        Background effects (comma-separated)
  --mask MASK           Mask effects (comma-separated)
  --no-ondemand         Continue processing when there is no application using the virtual webcam
  --use-sigmoid         Force the mask to follow a sigmoid distribution
  --threshold THRESHOLD
                        The minimum percentage threshold for accepting a pixel as foreground
  --no-postprocess      Disable postprocessing (masking dilation and blurring)
  --select-model SELECT_MODEL
                        Select the model for MediaPipe. For more information, please refer to https://github.com/fangfufu/Linux-Fake-Background-Webcam/issues/135#issuecomment-883361294
  --dump                Dump the filter configuration and exit

Unified filter options for selfie, background, and mask:

Each component accepts a comma-separated list of effects:
- file=<filename>: Use specified image/video file
- hologram: Apply hologram effect
- blur=<N>: Apply blur with intensity 0-100
- solid=<B,G,R>: Fill with specific BGR color
- cmap=<name>: Apply color map using cmapy
- brightness=<N>: Adjust brightness 0-200 (100=normal)
- no: Disable component (background and mask)
- tile: Tile the image (background only)
- crop: Maintain aspect ratio by cropping (background only)
- mask-update-speed=<N>: Control mask update speed (background only)
- foreground=<filename>: Specify foreground image to overlay (mask only)
- mask-file=<filename>: Explicitly specify mask file when using foreground (mask only)
- opacity=<N>: Set opacity 0-100 (selfie: person transparency, mask: overlay transparency)

Examples:
lfbw.py --selfie=blur=30,hologram
lfbw.py --background=file=mybg.jpg,cmap=viridis
lfbw.py --background=no,blur=50
lfbw.py --mask=file=mask.png,blur=20
lfbw.py --mask=foreground=logo.png,mask-file=logo-mask.png,opacity=80

Args that start with '--' can also be set in a config file (specified via -c). Config file syntax allows: key=value, flag=true, stuff=[a,b,c] (for details, see syntax at https://goo.gl/R74nmi). In
general, command-line values override config file values which override defaults.
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

## Contributors

Thanks for your contribution to the project!

[![Contributors Avatars](https://contributors-img.web.app/image?repo=fangfufu/Linux-Fake-Background-Webcam)](https://github.com/fangfufu/Linux-Fake-Background-Webcam/graphs/contributors)
[![Contributors Count](https://img.shields.io/github/contributors-anon/fangfufu/Linux-Fake-Background-Webcam?style=for-the-badge&logo=Linux-Fake-Background-Webcam)](https://github.com/fangfufu/Linux-Fake-Background-Webcam/graphs/contributors)

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


