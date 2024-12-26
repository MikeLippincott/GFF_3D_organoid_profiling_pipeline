#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=30:00
#SBATCH --output=main_job_submit_output-%j.out

cd slurm_scripts || exit

job1=$(sbatch inital_semgentation_HPC.sh | awk '{print $4}')

sbatch --dependency=afterok:$job1 process_segmentation_parent.sh

exit 0
