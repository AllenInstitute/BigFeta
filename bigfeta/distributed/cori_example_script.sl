#!/bin/bash -l


#SBATCH -J dk_test
#SBATCH -A m2043
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:05:00
#SBATCH -q premium
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -o "/global/homes/d/danielk/log/%j.out"

module load cray-petsc-64
module load cray-hdf5-parallel

input=$SCRATCH/test/solution_input.h5
output=$SCRATCH/solution_output.h5

srun $HOME/BigFeta/bigfeta/distributed/bin/bigfeta_solver_cori \
-input ${input} \
-output ${output} \
-ksp_type preonly -pc_type lu -pc_factor_mat_solver_package superlu_dist \
-log_view
