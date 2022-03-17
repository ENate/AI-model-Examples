FROM gitpod/workspace-full

RUN apt update && \
apt install --no-install-recommends -y build-essential gcc curl      ca-certificates python3 && \
apt clean && rm -rf /var/lib/apt/lists/*
# system clean up
RUN pip install --no-cache-dir --user -r supervised/req.txt




# install tensorflow
RUN pip install tensorflow
RUN apt-get install protobuf-compiler python-pil python-lxml
FROM python:3.8-slim
RUN pip install --no-cache-dir matplotlib pandas jupyter