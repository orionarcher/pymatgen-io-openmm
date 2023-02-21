# Start with a base image that includes conda
FROM continuumio/miniconda3

RUN conda create -n md python=3.9

# Copy in simulation files and change to that directory
COPY ./requirements.txt ./requirements.txt
RUN conda install -n md -c conda-forge --file ./requirements.txt

# Automatically activate the new environment on load
RUN echo "source activate md" > ~/.bashrc
ENV PATH /opt/conda/envs/md/bin:$PATH

COPY openmm_files/ openmm/
WORKDIR openmm/

# Start a new shell with base activated
ENTRYPOINT [ "bash", "-c", "source activate md && exec bash" ]
