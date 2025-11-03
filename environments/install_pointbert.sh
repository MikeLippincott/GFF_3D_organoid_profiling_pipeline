#!/bin/bash

# check for the environment pb_env
if conda info --envs | grep -q '^pb_env '; then
    echo "Conda environment 'pb_env' already exists."
    mamba env update -f pointbert_env.yaml
    conda activate pb_env
else
    echo "Creating conda environment 'pb_env'..."
    mamba create -n pb_env python=3.8 -y
    conda activate pb_env
fi


