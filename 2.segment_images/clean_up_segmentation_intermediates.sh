#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=6:00:00
#SBATCH --output=segmentation_cleanup-%j.out

module load anaconda
conda init bash
conda activate GFF_segmentation

echo "Cleaning up segmentation files"
python 7.clean_up_segmentation.py >> segmentation.log
echo -ne "\n"

conda deactivate
echo "Segmentation cleanup completed"
