#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --account=amc-general
#SBATCH --time=6:00:00
#SBATCH --output=cellpose-%j.out

# activate  cellprofiler environment
module load anaconda
conda init bash
conda activate GFF_cellpose

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd notebooks/ || exit

papermill 0.segment_organoids.ipynb 0.segment_organoids.ipynb


cd ../ || exit

# deactivate cellprofiler environment
conda deactivate

echo "Test complete"
