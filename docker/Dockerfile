# Start with a base image that includes conda
FROM continuumio/miniconda3

# Install openmm and gcc
RUN
RUN conda install -n base -c conda-forge openmm
RUN conda install -n base -c conda-forge gcc=12.2.0

# Copy in simulation files and change to that directory
COPY openmm_files/ openmm/
WORKDIR openmm/

# Start a new shell with base activated
ENTRYPOINT [ "bash", "-c", "source activate base && exec bash" ]
