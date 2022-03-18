FROM gitpod/workspace-full

#Install Python Packages
COPY requirements.txt /tmp/
RUN  pip3 install --requirement /tmp/requirements.txt
RUN cat /tmp/requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip3 install

# RUN mkdir -p /supervised/notebooks/


#add jupyter
# WORKDIR /supervised/notebooks/
# few inits
RUN RUN pip install --upgrade pip
RUN apt-get install -y protobuf-compiler python-pil python-lxml


RUN pip install --no-cache-dir matplotlib pandas jupyter jupyterlab
# RUN protoc object_detection/protos/*.proto --python_out=.

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
# Install tensorflow probability
# RUN pip install jaxlib
