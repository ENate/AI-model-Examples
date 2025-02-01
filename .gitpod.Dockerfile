FROM gitpod/workspace-full

# install miniconda
RUN sudo mkdir /home/gitpod/.conda
# Install conda
# RUN sudo mkdir /var/lib/pgadmin
# RUN sudo chmod -R 777 /var/lib/pgadmin
# RUN sudo chown -R 5050:5050 /var/lib/pgadmin
# RUN sudo mkdir /var/lib/pgadmin/sessions

#Install Python Packages
COPY requirements.txt /tmp/
RUN  pip3 install --requirement /tmp/requirements.txt
RUN cat /tmp/requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip3 install

# Install helm and kubectl 
# install tweepy and tweety 
RUN pip install tweety && pip install tweepy
# install tensorflow io
RUN pip install tensorflow-io
RUN pip install kafka-python
#add jupyter
# WORKDIR /supervised/notebooks/
# few init
RUN pip install --upgrade pip
RUN sudo apt-get install -y protobuf-compiler python-pil python-lxml

# Install tensorflow ranking and datasets
RUN pip install tensorflow
RUN pip install -q tensorflow-ranking && pip install -q --upgrade tensorflow-datasets
RUN pip install pip install --upgrade tensorflow-hub

RUN pip install --no-cache-dir matplotlib pandas jupyter jupyterlab
# RUN protoc object_detection/protos/*.proto --python_out=.

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
# Install tensorflow probability
# RUN pip install jaxlib
# install other useful components
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
RUN ~/miniconda3/bin/conda init bash
RUN ~/miniconda3/bin/conda init zsh
