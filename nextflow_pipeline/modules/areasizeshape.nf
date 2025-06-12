process AREASIZESHAPE_CPU {
    tag { "areasizeshape_cpu" }
    conda "${params.featurization_env}"

    input:
        tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
        stdout emit: dummy_output_ch_txt

    script:
        """
        echo "Running areasizeshape_cpu for patient: ${patient}, well_fov: ${well_fov}, use_gpu: ${featurize_with_gpu}"
        echo "Current environment: \$(conda info --envs | grep active | awk '{print \$1}')"
        echo "Env to use: ${params.featurization_env}"
        cd  ${projectDir}/../../3.cellprofiling/slurm_scripts/ || exit 1
        echo "Processing patient: ${patient}, well_fov: ${well_fov}, use_gpu: ${featurize_with_gpu}"
        bash run_area_shape_child.sh ${well_fov} ${featurize_with_gpu} ${patient}
        cd  ${workflow.projectDir}/ || exit 1
        """
}

process AREASIZESHAPE_GPU {
    conda "${params.featurization_env}"

    tag { "areasizeshape_gpu" }

    input:
        tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
        stdout emit: dummy_output_ch_txt

    script:
        """
        cd  ${projectDir}/../../3.cellprofiling/slurm_scripts/ || exit 1
        echo "Processing patient: ${patient}, well_fov: ${well_fov}, use_gpu: ${featurize_with_gpu}"
        bash run_area_shape_child.sh ${well_fov} ${featurize_with_gpu} ${patient}
        cd  ${workflow.projectDir}/ || exit 1
        """
}
