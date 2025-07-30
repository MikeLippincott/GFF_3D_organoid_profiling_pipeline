#!/bin/bash

conda activate r_shiny_env

# check for faults - use direct exit code checking
if ! Rscript deploy_app.R; then
    echo "Deployment failed. Please check the logs for errors."
    exit 1
fi

conda deactivate

echo "Deployment completed successfully."
