#!/bin/bash

# MPI Parameters
MPIEXEC="mpiexec"
EXEC="python example_dcres_ny800_3d_vp_over_vs.py"

# Command
#read -p "Number of MPI process: " num_procs
#read -p "Number of threads per process: " num_threads
num_procs=$1
echo $num_procs
num_threads=$2
dtype=$3
out_name=$4
data=$5
data_tomo=$6
$MPIEXEC -np $num_procs --bind-to core:overload-allowed --map-by ppr:$num_procs:node:pe=$num_threads $EXEC -n $num_threads $dtype $out_name $data $data_tomo > pbs_logs/log_run_dcres_MPI_${num_procs}_OMP_${num_threads}.txt

#python example_dcres.py -n $num_threads $dtype $out_name $data>> output/log_file_dcinv_ny800.txt

#python example_dcres_view.py $dtype $out_name $data>> output/log_file_dcinv_ny800.txt