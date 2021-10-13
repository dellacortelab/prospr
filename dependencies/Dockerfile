FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN \
  # Update package list
  apt-get update -y && \
  # Install...
  apt-get install -y \
  git \
  vim \
  wget \
  python3 \
  python3-dev \
  python3-pip \
  # Remove package lists
  && rm -rf /var/lib/apt/lists/*

# Install conda
RUN cd /home && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh && \
    bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p $HOME/miniconda
RUN $HOME/miniconda/bin/conda init bash

# RUN git clone https://github.com/dellacortelab/prospr.git
# # # Install prospr dependencies
# RUN $HOME/miniconda/bin/conda env create -f prospr/prospr-env.yml
COPY prospr-env.yml /tmp/prospr-env.yml
RUN $HOME/miniconda/bin/conda env create -f tmp/prospr-env.yml
# COPY prospr-env.yml /tmp/prospr-env.yml
# RUN $HOME/miniconda/bin/conda activate prospr-env && $HOME/miniconda/bin/conda env update -f tmp/prospr-env.yml
RUN echo "export PATH=$HOME/miniconda/bin:$PATH" >> $HOME/.bashrc
RUN echo "conda init bash" >> $HOME/.bashrc
RUN echo "conda activate prospr-env" >> $HOME/.bashrc

# # Install DSSP
# RUN wget https://github.com/cmbi/dssp/archive/refs/tags/3.1.4.tar.gz
# RUN tar -zxvf 3.1.4.tar.gz --no-same-owner
# RUN cd dssp-3.1.4 && ./autogen.sh && ./configure && make mkdssp && make install

SHELL ["/bin/bash"]