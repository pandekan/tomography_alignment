#!/bin/bash

# Store the original directory
original_directory=$(pwd)

# activate the virtual environment
poetry shell

# Change to the desired directory using a relative path
cd tomoalign

# Run your commands in the desired directory
python -m numpy.f2py -c -m ray_wt_grad ray_wt_grad.f90
python -m numpy.f2py -c -m vox_wt_grad vox_wt_grad.f90

# Return to the original directory
cd "$original_directory"
