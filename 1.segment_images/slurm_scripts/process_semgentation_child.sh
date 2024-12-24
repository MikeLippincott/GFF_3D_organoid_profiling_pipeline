#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=1:00:00
#SBATCH --output=preprocessing-%j.out


module load anaconda

conda activate GFF_segmentation

cd scripts/ || exit

dir=$1
compartment=$2

echo "$dir"
echo "$compartment"

python 2.segmentation_decoupling.py --input_dir "$dir" --compartment "$compartment"
python 3.reconstruct_3D_masks.py --input_dir "$dir" --compartment "$compartment" --radius_constraint 10
python 4.create_cytoplasm_masks.py --input_dir "$dir" --compartment "$compartment"
python 5.make_cell_segmentation_videos.py --input_dir "$dir" --compartment "$compartment"
python 5.make_cell_segmentation_videos.py --input_dir "$dir" --compartment "cytoplasm"

cd ../ || exit

conda deactivate

echo "Segmentation complete"
