#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=10:00
#SBATCH --output=segmentation_grandparent-%j.out

# activate  cellprofiler environment
module load anaconda
conda init bash
conda activate GFF_segmentation

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

# read in the patient IDs
patient_id_file="$git_root/data/patient_IDs.txt"
if [ ! -f "$patient_id_file" ]; then
    echo "Error: Patient ID file $patient_id_file does not exist."
    exit 1
fi
# read the patient IDs into an array
mapfile -t patient_array < "$patient_id_file"

for patient in "${patient_array[@]}"; do
    echo "Submitting parent segmentation for patient $patient"
    number_of_jobs=$(squeue -u "$USER" | wc -l)
    while [ "$number_of_jobs" -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u "$USER" | wc -l)
    done

    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=amilan \
        --qos=long \
        --account=amc-general \
        --time=7-00:00:00 \
        --output=segmentation_parent-%j.out \
        "${git_root}"/2.segment_images/parent_segmentation.sh "$patient"

done

conda deactivate

echo "All patients submitted for segmentation"
