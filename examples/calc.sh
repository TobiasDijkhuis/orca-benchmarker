#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 1
#SBATCH --threads-per-core 1
#SBATCH --time 10:00:00 
#SBATCH --mem-per-cpu 5G

calculation_dir=$(pwd)

# Replace NPROCS_HERE by the amount of tasks per node.
sed -i -e "s/NPROCS_HERE/${SLURM_NTASKS_PER_NODE}/g" ${calculation_dir}/orca.inp

# Always run stuff in local scratch on node for job and not in ${HOME} to avoid NFS network slowness
scratch_dir="/scratch/""${SLURM_JOB_USER}"/"${SLURM_JOB_ID}"
cd $scratch_dir

# ORCA has the tendency to put temporary files in the same directory as the input file,
# so we copy the input file from the calculation directory to the scratch directory
cp ${calculation_dir}/orca.inp ${scratch_dir}/orca.inp

# Run calculation. Take input file from scratch directory, and put output there too.
${HOME}/orca_6_1_0/orca ${scratch_dir}/orca.inp > ${calculation_dir}/orca.out

# Copy relevant things to calculation directory
cp ${scratch_dir}/orca.* ${calculation_dir}/.

# Remove scratch directory
rm -r ${scratch_dir}
