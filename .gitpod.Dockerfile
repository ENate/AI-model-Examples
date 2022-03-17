FROM gitpod/workspace-full

# Install conda
RUN sudo wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
    
RUN chown -R gitpod:gitpod /opt/conda \
    && chmod -R 777 /opt/conda \
    && chown -R gitpod:gitpod /home/gitpod/.conda \
    && chmod -R 777 /home/gitpod/.conda


#RUN chown -R gitpod:gitpod /home/gitpod/.cache \
#    && chmod -R 777 /home/gitpod/.cache

#Install Python Packages
# COPY requirements.txt /tmp/
# #RUN  pip3 install --requirement /tmp/requirements.txt
# RUN cat /tmp/requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip3 install
RUN pip install --no-cache-dir matplotlib pandas jupyter
