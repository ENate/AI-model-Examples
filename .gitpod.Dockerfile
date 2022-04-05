FROM gitpod/workspace-full
# install miniconda
RUN sudo mkdir /home/gitpod/.conda
# Install conda
RUN sudo mkdir /var/lib/pgadmin/session

#Install Python Packages
COPY requirements.txt /tmp/
RUN  pip3 install --requirement /tmp/requirements.txt
RUN cat /tmp/requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip3 install

# Install helm and kubectl 
# install tweepy and tweety 
RUN pip install tweety && pip install tweepy

#add jupyter
# WORKDIR /supervised/notebooks/
# few init
RUN pip install --upgrade pip
RUN sudo apt-get install -y protobuf-compiler python-pil python-lxml

# Install tensorflow ranking and datasets
RUN pip install -q tensorflow-ranking && pip install -q --upgrade tensorflow-datasets


RUN pip install --no-cache-dir matplotlib pandas jupyter jupyterlab
# RUN protoc object_detection/protos/*.proto --python_out=.

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
# Install tensorflow probability
# RUN pip install jaxlib
# install other useful components
