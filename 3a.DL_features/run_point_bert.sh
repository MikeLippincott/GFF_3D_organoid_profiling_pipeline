#!/bin/bash

cd utils || exit 1
git clone https://github.com/Julie-tang00/Point-BERT.git
# add a pyproject file to make it installable
cd .. || exit 1
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb
docker build -t pointcloud_gpu ./utils

# spin up the docker and run point-bert
# we first need to compile and install the chamfer distance and emd extensions
# these fail during build as we do not have access to a GPU at that time
docker run --gpus all --rm -it -v "$(pwd):/workspace" -w /workspace pointcloud_gpu \
#   bash -lc "source /opt/conda/etc/profile.d/conda.sh && \ # --- IGNORE ---
    cd /workspace/utils/Point-BERT/extensions/chamfer_dist && \
        /opt/conda/bin/mamba run -n point_env python setup.py install --user && \
        /opt/conda/bin/mamba run -n point_env pip install -e . && \
    # cd /workspace/utils/Point-BERT/extensions/emd && \ # --- IGNORE ---
    #     /opt/conda/bin/mamba run -n point_env python setup.py install --user && \ # --- IGNORE ---
    #     /opt/conda/bin/mamba run -n point_env pip install -e . && \ # --- IGNORE ---
  cd ../../ && \
  conda activate point_env && python ../../scripts/pointBert_call.py
