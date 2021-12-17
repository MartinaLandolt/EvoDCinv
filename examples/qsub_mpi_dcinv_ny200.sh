#!/bin/bash

source ~/.profile

conda activate evodcinv_github_env1
cd evodcinv_github/EvoDCinv/examples/
#./mpi_dcinv_non_interactive.sh 8 2 'group' 'dcinv_group_data' 'data/groupV_data'
#./mpi_dcinv_non_interactive.sh 8 4 'phase' 'dcinv_phase' 'data'
./mpi_dcinv_ny200_non_interactive.sh 8 8 'phase' 'dcinv_phase_ny200' 'data'
