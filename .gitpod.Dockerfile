FROM gitpod/workspace-full

# RUN apt update && apt install --no-install-recommends -y build-essential gcc curl      ca-certificates python3 
# && \
# apt clean && rm -rf /var/lib/apt/lists/*
# system clean up
RUN pip install -r supervised/requirements.txt

RUN wget https://github.com/wagoodman/dive/releases/download/v0.9.2/dive_0.9.2_linux_amd64.deb
RUN sudo apt install ./dive_0.9.2_linux_amd64.deb


# install tensorflow
# RUN pip install tensorflow
# RUN apt-get install protobuf-compiler python-pil python-lxml
# FROM python:3.8-slim
# RUN pip install --no-cache-dir matplotlib pandas jupyter
