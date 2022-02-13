#!/bin/bash

source ~/.profile

conda activate passivetomochain_env_last_martina_v1
cd /home_c/KAZANTSEV/Programs/evodcinv_last_martina/EvoDCinv/examples
#./mpi_dcinv_ny800_3d_example_non_interactive_small_pop.sh 4 8 'phase' 'output_test_parallelism_8_4' 'data' 'data/disp_curves_rayleigh_group_tomo.pickle'
#./mpi_dcinv_ny800_3d_example_non_interactive_small_pop.sh 4 8 'phase' 'output_test_parallelism_4_8' 'data' 'data/disp_curves_rayleigh_group_tomo.pickle'
#./mpi_dcinv_ny800_3d_example_non_interactive_small_pop.sh 2 16 'phase' 'output_test_parallelism_2_16' 'data' 'data/disp_curves_rayleigh_group_tomo.pickle'
#./mpi_dcinv_ny800_3d_example_non_interactive_small_pop.sh 1 32 'phase' 'output_test_parallelism_1_32' 'data' 'data/disp_curves_rayleigh_group_tomo.pickle'
#./mpi_dcinv_ny800_3d_example_non_interactive_small_pop.sh 1 16 'phase' 'output_test_parallelism_1_16' 'data' 'data/disp_curves_rayleigh_group_tomo.pickle'
#./mpi_dcinv_ny800_3d_example_non_interactive_small_pop.sh 2 8 'phase' 'output_test_parallelism_2_8' 'data' 'data/disp_curves_rayleigh_group_tomo.pickle'
#./mpi_dcinv_ny800_3d_example_non_interactive_small_pop.sh 4 4 'phase' 'output_test_parallelism_4_4' 'data' 'data/disp_curves_rayleigh_group_tomo.pickle'
#./mpi_dcinv_ny800_3d_example_non_interactive_small_pop.sh 1 8 'phase' 'output_test_parallelism_1_8' 'data' 'data/disp_curves_rayleigh_group_tomo.pickle'
#./mpi_dcinv_ny800_3d_example_non_interactive_small_pop.sh 1 2 'phase' 'output_test_parallelism_1_2' 'data' 'data/disp_curves_rayleigh_group_tomo.pickle'
./mpi_dcinv_ny800_3d_example_non_interactive_small_pop.sh 1 4 'phase' 'output_test_parallelism_1_16' 'data' 'data/disp_curves_rayleigh_group_tomo.pickle'