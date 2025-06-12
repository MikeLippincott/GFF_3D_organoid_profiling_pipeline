process SEGMENTATION {
    // specify the conda environment to use for this process
    conda "${params.segmentation_env}"

    tag { "${patient}-${well_fov}" }

    // specify the input for this process
    input:
        tuple val(patient), val(well_fov)

    output:
        tuple val(patient), val(well_fov), path("data/${patient}/processed_data/${well_fov}/")

    script:
        """
        # Set the base directory to the current working directory
        cd ${projectDir}/../2.segment_images/ || exit 1
        echo "Processing patient: ${patient}, well_fov: ${well_fov}"
        # call the chile script to run segmentation for the given patient and well_fov
        source child_segmentation.sh ${well_fov} ${patient}
        cd ${projectDir}/ || exit 1
        """
}
