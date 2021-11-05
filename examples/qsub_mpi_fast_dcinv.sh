#!/bin/bash

source ~/.profile

conda activate evodcinv_github_env1
cd evodcinv_github/EvoDCinv/examples/
./mpi_fast_dcinv_non_interactive.sh 8 4 'group' 'fastdcinv_group_data' 'data/groupV_model'
#./mpi_fast_dcinv_non_interactive.sh 8 4 'phase' 'fastdcinv_phase' 'data'