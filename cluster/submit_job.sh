#!/bin/bash
#
### ADAPT TO YOUR PREFERRED SLURM OPTIONS ###
#SBATCH --mail-type=ALL
#SBATCH --mail-user=max.mustermann@hhi.fraunhofer.de
#SBATCH --job-name="cissir"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --output=log/%x.%j.out


# include the definition of the LOCAL_JOB_DIR which is autoremoved after each job
source "/etc/slurm/local_job_dir.sh"

CODE_DIR="/data/cluster/users/${USER}/cissir"
CODE_MNT="/mnt/project"

singularity run --nv --bind ${CODE_DIR}:${CODE_MNT}  ./cluster/sionna.sif
