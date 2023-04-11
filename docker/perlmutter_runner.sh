#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 4 ]; then
  echo "Usage: ./perlmutter_runner.sh <CONTAINER_NAME> <PYTHON_FILE> <INPUT_DIRECTORY> <OUTPUT_DIRECTORY>"
  exit 1
fi

# Get the input and output directories
container_executable="$1"
script_path="$(realpath "$2")"
script_dir=$(dirname "$script_path")
script=$(basename "$script_path")
input_dir="$(realpath "$3")"
output_dir="$(realpath "$4")"

podman-hpc run \
    -e SLURM_PROCID=$SLURM_PROCID \
    -v "${input_dir}:/input" \
    -v "${output_dir}:/output" \
    -v "${script_dir}:/app" \
    --gpu "${container_executable}" \
    python "$script"
