FROM debian:bullseye

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-numpy \
      python3-opencv \
      python3-requests \
      && rm -rf /var/cache/apt/* /var/lib/apt/lists/*

WORKDIR /src

COPY foreground.jpg \
    foreground-mask.png \
    background.jpg \
    fake.py \
    akvcam.py \
    requirements.txt \
    /src/

RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "-u", "fake.py"]
CMD ["--bodypix-url", "http://bodypix:9000/"]
