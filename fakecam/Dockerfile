FROM python:3-buster
# ensure pip is up to date
RUN pip install --upgrade pip
# install opencv dependencies
RUN apt-get update && \
    apt-get install -y \
      `# opencv requirements` \
      libsm6 libxext6 libxrender-dev \
      `# opencv video opening requirements` \
      libv4l-dev
# install our requirements
WORKDIR /src
COPY requirements.txt /src/
RUN pip install --no-cache-dir -r /src/requirements.txt
# copy in the virtual background
COPY background.jpg /data/
# run our fake camera script (with unbuffered output for easier debug)
COPY fake.py /src/
ENTRYPOINT python -u fake.py
