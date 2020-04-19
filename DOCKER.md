# Linux-Fake-Background-Webcam with Docker

## Prerequisite

You need to have *docker* and *docker-compose* available.

## Configuration 

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
## Usage
 - Run and initial build containers: ``docker-compose up`` (or ``docker-compose up -d``)
 - Stop and remove containers: ``docker-compose down``
