#!/bin/bash
docker network create --driver bridge fakecam

docker create \
  --name=bodypix \
  --network=fakecam \
  -p 9000:9000 \
  --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  bodypix

docker create \
  --name=fakecam \
  --network=fakecam \
  -u "$(id -u):$(getent group video | cut -d: -f3)" \
  $(find /dev -name 'video*' -printf "--device %p ") \
  fakecam
