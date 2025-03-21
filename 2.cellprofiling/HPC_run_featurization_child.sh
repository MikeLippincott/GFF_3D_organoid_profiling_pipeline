#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --partition=amem
#SBATCH --qos=mem
#SBATCH --account=amc-general
#SBATCH --time=48:00:00
#SBATCH --output=child_featurize-%j.out

module load miniforge
conda init bash
conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd notebooks/ || exit

# dir=$1

# python run_featurization.py --input_dir $dir

papermill run_featurization.ipynb run_featurization.ipynb

cd ../ || exit

conda deactivate

echo "Featurization done"
