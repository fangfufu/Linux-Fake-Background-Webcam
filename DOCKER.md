# Linux-Fake-Background-Webcam with Docker

## Docker Compose (CPU)

### Prerequisite

You need to have *docker* and *docker-compose* available and the *correct version of v4l2loopback* installed.

### Configuration

You should create a copy of `docker-compose.yml` and change to meet your needs:

- add a customer background image using a volume mapping:
  ```
      fakecam:
          # ...
          volumes:
            - /path/to/background.jpg:/src/background.jpg:ro
          # ...
  ```

- change the device mappings if you are using diffent devices:
  ```
      fakecam:
          # ...
          devices:
              # input (webcam)
              - /dev/video0:/dev/video0
              # output (virtual webcam)
              - /dev/video1:/dev/video2
          # ...
  ```
### Usage
 - Run and initial build containers: ``docker-compose up`` (or ``docker-compose up -d``)
 - Stop and remove containers: ``docker-compose down``
 - Note: *Ctrl-C* is currently stops the containers instead of changing images


## Docker (GPU)

### Prerequisites

* v4l2loopback
* Docker
* [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker#quickstart)

### Configuration


Build Images:

```bash
docker build -t bodypix -f ./bodypix/Dockerfile.gpu ./bodypix
docker build -t fakecam ./fakecam
```

Create a Network:

```bash
docker network create --driver bridge fakecam
```

Create a Volume:

```bash
docker volume create --name fakecam
```

Start the bodypix app with GPU support and listen on a UNIX socket:

```bash
docker run -d \
  --rm \
  --name=bodypix \
  --network=fakecam \
  -v fakecam:/socket \
  -e PORT=/socket/bodypix.sock \
  --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  bodypix
```

Start the camera, note that we need to pass through video devices,
and we want our user ID and group to have permission to them
you may need to `sudo groupadd $USER video`:

```bash
docker run -d \
  --rm \
  --name=fakecam \
  --network=fakecam \
  --device=/dev/video2:/dev/video0 \
  --device=/dev/video11:/dev/video2 \
  -v fakecam:/socket \
  fakecam \
  -B /socket/bodypix.sock --no-foreground --scale-factor 1
```

After you've finished, clean up:

```bash
docker rm -f fakecam bodypix
docker volume rm fakecam
docker network rm fakecam
```