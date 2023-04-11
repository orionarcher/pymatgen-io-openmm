# How to Use Podman to Run on Perlmutter

### Downloading the Container

```bash
# podman-hpc needs this to store the container
mkdir $SCRATCH/storage

# this will prompt you for your username and password, leave both blank
podman-hpc login docker.io

# pull the image
podman-hpc pull orioncohen/pymatgen_openmm_gpu:latest

# test that the image is present
podman-hpc images
```

### Inputs and Outputs

For the production.py script that's included in this repository, the
INPUT_DIRECTORY should contain four subdirectories, each containing an input
set, e.g.: 'topology.pdb', 'system.xml', 'integrator.xml', and 'state.xml'.

The OUTPUT_DIRECTORY can be anywhere, but putting it in scratch is best.

### Running the Container

You will need to get onto a compute node to run the container. You can do this
either with `salloc`:

```bash
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=ACCOUNT
```

or with `sbatch` and an appropriate script like this:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --account=ACCOUNT
````

Either way, you can then execute the container with the following script.
Note that perlmuter_runner.sh is included in this repository and must
be in your PATH.

```bash
chmod +x perlmutter_runner.sh

srun -n 4 -G 4 perlmutter_runner.sh \
  docker.io/orioncohen/pymatgen_openmm_gpu \
  production.py \
  <INPUT_DIRECTORY> \
  <OUTPUT_DIRECTORY>
```
