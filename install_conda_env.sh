#!/bin/bash

CONDA_ENV=nct

conda env create -f conda_env.yml

conda run -n $CONDA_ENV pip3 install -e tonic/

# Clean up any unnecessary files
conda run -n $CONDA_ENV conda clean -y --all

