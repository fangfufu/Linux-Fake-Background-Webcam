FROM ubuntu:18.04
# Install node
RUN apt update && apt install -y curl make build-essential \
    && curl -sL https://deb.nodesource.com/setup_12.x | bash - \
    && apt-get -y install nodejs \
    && mkdir /.npm \
    && chmod 777 /.npm
# Install node package dependencies
WORKDIR /src
COPY package.json /src/
RUN npm install
# Setup our app as the entrypoint
COPY app.js /src/
ENTRYPOINT node /src/app.js
