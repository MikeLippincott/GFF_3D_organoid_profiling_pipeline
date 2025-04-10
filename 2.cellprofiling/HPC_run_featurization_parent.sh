#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=24:00:00
#SBATCH --output=annotate_sc_parent-%j.out

conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

parent_dir="../../data/NF0014/cellprofiler"
# get the list of all dirs in the parent_dir
dirs=$(ls -d $parent_dir/*)

# for dir in $dirs; do
#     dir_name=$(basename $dir)
#     echo $dir_name
#     number_of_jobs=$(squeue -u $USER | wc -l)
#     while [ $number_of_jobs -gt 990 ]; do
#         sleep 1s
#         number_of_jobs=$(squeue -u $USER | wc -l)
#     done
#     sbatch HPC_run_featurization_child.sh $dir
# done

use_GPU="TRUE"
cd slurm_scripts || exit

if [ "$use_GPU" = "TRUE" ]; then
    echo "Running GPU version"

    sbatch \
        --nodes=1 \
        --ntasks=8 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=30:00 \
        run_area_shape_child.sh $use_GPU

    sbatch \
        --nodes=1 \
        --ntasks=8 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=30:00 \
        run_colocalization_child.sh $use_GPU

    sbatch \
        --nodes=1 \
        --ntasks=8 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=30:00 \
        run_granularity_child.sh $use_GPU

    sbatch \
        --nodes=1 \
        --ntasks=8 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=30:00 \
        run_instensity_child.sh $use_GPU

else
    echo "Running CPU version"

    sbatch \
        --nodes=1 \
        --mem=300G \
        --partition=amem \
        --qos=mem \
        --account=amc-general \
        --time=30:00 run_area_shape_child.sh

    sbatch \
        --nodes=1 \
        --mem=300G \
        --partition=amem \
        --qos=mem \
        --account=amc-general \
        --time=30:00 run_colocalization_child.sh

    sbatch \
        --nodes=1 \
        --mem=300G \
        --partition=amem \
        --qos=mem \
        --account=amc-general \
        --time=30:00 run_granularity_child.sh

    sbatch \
        --nodes=1 \
        --mem=300G \
        --partition=amem \
        --qos=mem \
        --account=amc-general \
        --time=30:00 run_instensity_child.sh

fi

sbatch \
    --nodes=1 \
    --mem=300G \
    --partition=amem \
    --qos=mem \
    --account=amc-general \
    --time=30:00 run_neighbors_child.sh



sbatch \
    --nodes=1 \
    --mem=300G \
    --partition=amem \
    --qos=mem \
    --account=amc-general \
    --time=30:00 run_texture_child.sh


cd ../ || exit

echo "Featurization done"
