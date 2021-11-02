#!/bin/bash

source ~/.profile

conda activate evodcinv_old_with_uncertainties_env
cd evodcinv_old_merge_uncertainties_and_groupV/examples/
./mpi_fast_dcinv_non_interactive.sh 8 4 'group' 'test'
