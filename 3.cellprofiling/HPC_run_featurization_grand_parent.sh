#!/bin/bash


patient=$1

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

json_file="${git_root}/3.cellprofiling/load_data/input_combinations.json"

# Check if JSON file exists
if [ ! -f "$json_file" ]; then
    echo "Error: JSON file not found at $json_file"
    exit 1
fi

parent_dir="${git_root}/data/$patient/zstack_images"
# get the list of all dirs in the parent_dir
dirs=$(ls -d "$parent_dir"/*)

# Parse JSON using pure bash (no jq or python required)
# Extract combinations using grep and sed
# Alternative: Using awk for better JSON parsing
# Extract all four fields using grep and sed
grep -E '"(feature|compartment|channel|processor_type)"[[:space:]]*:[[:space:]]*"[^"]*"' "$json_file" | \
sed -E 's/.*"(feature|compartment|channel|processor_type)"[[:space:]]*:[[:space:]]*"([^"]+)".*/\2/' | \
paste - - - - | \
while read -r feature compartment channel processor_type; do

    # loop through each dir and submit a job
    for dir in $dirs; do
        well_fov=$(basename "$dir")
        echo "$well_fov"
        # check that the number of jobs is less than 990
        # prior to submitting a job
        number_of_jobs=$(squeue -u "$USER" | wc -l)
        while [ "$number_of_jobs" -gt 990 ]; do
            sleep 1s
            number_of_jobs=$(squeue -u "$USER" | wc -l)
        done
        sbatch \
            --nodes=1 \
            --ntasks=1 \
            --partition=amilan \
            --qos=normal \
            --account=amc-general \
            --time=5:00 \
            --output="featurize_parent_${patient}_${well_fov}_${feature}_${processor_type}_%j.out" \
            "$git_root"/3.cellprofiling/HPC_run_featurization_parent.sh \
            "$patient" \
            "$well_fov" \
            "$compartment" \
            "$channel" \
            "$feature" \
            "$processor_type"
    done
done


echo "Featurization done"
