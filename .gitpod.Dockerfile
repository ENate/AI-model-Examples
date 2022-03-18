FROM gitpod/workspace-full

#Install Python Packages
COPY requirements.txt /tmp/
RUN  pip3 install --requirement /tmp/requirements.txt
RUN cat /tmp/requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip3 install

#add jupyter
WORKDIR /supervised/notebooks

RUN pip install --no-cache-dir matplotlib pandas jupyter jupyterlab

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
# Install tensorflow probability
