#!/bin/bash

# MPI Parameters
MPIEXEC="mpiexec"
EXEC="python fast_example_dcinv.py"

# Command
#read -p "Number of MPI process: " num_procs
#read -p "Number of threads per process: " num_threads
num_procs=$1
num_threads=$2
dtype=$3
out_name=$4
data=$5
$MPIEXEC -np $num_procs --bind-to core:overload-allowed --map-by ppr:$num_procs:node:pe=$num_threads $EXEC -n $num_threads $dtype $out_name > output/log_file_fast_dcinv.txt

python example_dcres.py -n $num_threads $dtype $out_name $data>> output/log_file_fast_dcinv.txt

python example_dcres_view.py $dtype $out_name $data>> output/log_file_fast_dcinv.txt
