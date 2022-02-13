#!/bin/bash

source ~/.profile

conda activate passivetomochain_env_last_martina_v1
cd /home_c/KAZANTSEV/Programs/evodcinv_last_martina/EvoDCinv/examples
#./mpi_dcinv_non_interactive.sh 8 2 'group' 'dcinv_group_data_ny800' 'data/groupV_data'
#./mpi_dcinv_non_interactive.sh 8 4 'phase' 'dcinv_phase' 'data'
./mpi_dcinv_ny800_3d_example_non_interactive.sh 5 8 'phase' 'output' 'data' 'data/disp_curves_rayleigh_group_tomo.pickle'