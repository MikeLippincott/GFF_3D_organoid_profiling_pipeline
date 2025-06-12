process SEGMENTATION_CLEANUP {
    tag { "${patient}-${well_fov}" }
    conda "${params.segmentation_env}"


    input:
        tuple val(patient), val(well_fov)

    output:
        tuple val(patient), val(well_fov), path("data/${patient}/processed_data/${well_fov}/")

    script:
        """
        # Set the base directory to the current working directory
        cd ${projectDir}/../2.segment_images/ || exit 1
        echo "Cleaning up segmentation for patient: ${patient}, well_fov: ${well_fov}"
        # call the child script to clean up segmentation for the given patient and well_fov
        source clean_up_segmentation_intermediates_child.sh ${patient}
        cd ${projectDir}/ || exit 1
        """
}
