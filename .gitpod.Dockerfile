FROM gitpod/workspace-full

#Install Python Packages
COPY requirements.txt /tmp/
RUN  pip3 install --requirement /tmp/requirements.txt
RUN cat /tmp/requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip3 install
RUN pip install --no-cache-dir matplotlib pandas jupyter
