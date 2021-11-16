#!/bin/bash

source ~/.profile

conda activate evodcinv_github_env1
cd evodcinv_github/EvoDCinv/examples/
#./mpi_dcinv_non_interactive.sh 8 4 'group' 'dcinv_group_data_nouveau_plot' 'data/groupV_model'
./mpi_dcinv_non_interactive.sh 8 4 'phase' 'dcinv_phase_only_f_under_5Hz_bis' 'data/only_f_under_5Hz'
