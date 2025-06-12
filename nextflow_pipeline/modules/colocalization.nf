process COLOCALIZATION_CPU {
    tag { "colocalization_cpu" }
    conda "${params.featurization_env}"

    input:
        tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
        stdout emit: dummy_output_ch_txt

    script:
        """
        cd ${projectDir}/../../3.cellprofiling/slurm_scripts/ || exit 1
        echo "Processing patient: ${patient}, well_fov: ${well_fov}"
        bash run_colocalization_child.sh ${well_fov} ${featurize_with_gpu} ${patient}
        cd ${projectDir}/ || exit 1
        """
}

process COLOCALIZATION_GPU {
    tag { "colocalization_gpu" }
    conda "${params.featurization_env}"

    input:
        tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
        stdout emit: dummy_output_ch_txt

    script:
        """
        cd ${projectDir}/../../3.cellprofiling/slurm_scripts/ || exit 1
        echo "Processing patient: ${patient}, well_fov: ${well_fov}"
        bash run_colocalization_child.sh ${well_fov} ${featurize_with_gpu} ${patient}}
        cd ${projectDir}/ || exit 1
        """
}
