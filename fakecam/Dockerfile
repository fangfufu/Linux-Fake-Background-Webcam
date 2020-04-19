FROM debian:bullseye

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-numpy \
      python3-opencv \
      python3-requests \
      && rm -rf /var/cache/apt/* /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir pyfakewebcam

WORKDIR /src

COPY foreground.jpg foreground-mask.png background.jpg fake.py /src/

ENTRYPOINT python3 -u fake.py
