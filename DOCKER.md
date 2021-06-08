# Linux-Fake-Background-Webcam with Docker

## Setup

copy `docker_defaults.env` to a file named `.env` in same directory e.g.
```shell script
cp docker_defaults.env .env
```

Get list of your video devices. An app like `v4l2-ctl` should help:
```shell script
v4l2-ctl --list-devices
```

Make any changes you need (use nvidia gpu, update images, change video volume mapping) to the newly created `.env` file.

## Prerequisites

* v4l2loopback
* Docker
* docker-compose


### Usage

 - Start it up: `docker-compose up --build` (or `docker-compose up -d --build`)
 - Stop and remove containers: `docker-compose down`
 - Note: *Ctrl-C* is currently stops the containers instead of changing images
    - You can instead rebuild with new settings: `docker-compose up -d --build`