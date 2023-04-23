FROM docker.io/nvidia/cuda:11.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /opt

RUN \
    apt-get update        && \
    apt-get upgrade --yes && \
    apt-get install --yes    \
        wget                 \
        vim              &&  \
    apt-get clean all    &&  \
    rm -rf /var/lib/apt/lists/*

#install miniconda
#pin to python 3.9 for rapids compatibility
ENV installer=Miniconda3-py39_4.12.0-Linux-x86_64.sh

RUN wget https://repo.anaconda.com/miniconda/$installer && \
    /bin/bash $installer -b -p /opt/miniconda3          && \
    rm -rf $installer

ENV PATH=/opt/miniconda3/bin:$PATH

#use mamba to speed up package resolution
RUN /opt/miniconda3/bin/conda install mamba -c conda-forge -y

RUN /opt/miniconda3/bin/mamba install -c conda-forge openmm cudatoolkit=11.7 mpi4py git pip gcc -y

ENV OPENMM_PLUGIN_DIR=/opt/miniconda3/lib/plugins

RUN wget https://raw.githubusercontent.com/orionarcher/pymatgen-io-openmm/main/requirements.txt

RUN /opt/miniconda3/bin/mamba install -c conda-forge --file requirements.txt -y

RUN /opt/miniconda3/bin/conda uninstall pymatgen -y

RUN pip install git+https://github.com/materialsproject/pymatgen.git

RUN pip install git+https://github.com/orionarcher/pymatgen-io-openmm.git

WORKDIR /app
