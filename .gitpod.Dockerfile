FROM gitpod/workspace-full

# install tensorflow
RUN pip install tensorflow
RUN apt-get install protobuf-compiler python-pil python-lxml
RUN pip install jupyter
RUN pip install matplotlib