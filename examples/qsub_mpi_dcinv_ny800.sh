#!/bin/bash

source ~/.profile

conda activate evodcinv_github_env1
cd evodcinv_github/EvoDCinv/examples/
#./mpi_dcinv_non_interactive.sh 8 2 'group' 'dcinv_group_data_ny800' 'data/groupV_data'
#./mpi_dcinv_non_interactive.sh 8 4 'phase' 'dcinv_phase' 'data'
./mpi_dcinv_ny800_non_interactive.sh 8 8 'group' 'dcinv_group_ny800' 'data/groupV_data'
