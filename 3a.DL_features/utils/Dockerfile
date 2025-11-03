# ----------------------------
# This Docker file sets up an environment for PointBERT with GPU support
# It installs CUDA, cuDNN, Miniconda, PyTorch with CUDA support,
# and necessary dependencies including pointnet2, KNN_CUDA, chamfer distance, and emd.
# I could not get any of these installs to work on my local machine and
# the only way to successfully install was through a Docker container with GPU support.
# ----------------------------
# ----------------------------
# Base image
# ----------------------------
# FROM ubuntu:18.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04

# ----------------------------
# Set environment variables
# ----------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

# ----------------------------
# Install system dependencies
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    gnupg2 \
    build-essential \
    cmake \
    python3-dev \
    python3-pip \
    gcc \
    g++ \
    ninja-build \
    libopenblas-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

# # ----------------------------
# # Install CUDA 10.2
# # ----------------------------
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin && \
#     mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
#     wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb && \
#     dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb && \
#     apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub && \
#     apt-get update && \
#     apt-get -y install cuda && \
#     rm cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb && \
#     rm -rf /var/lib/apt/lists/*

# # ----------------------------
# # Install cuDNN
# # ----------------------------
# COPY cudnn-local-repo-ubuntu1804-8.5.0.96_1.0-1_amd64.deb /tmp/
# RUN dpkg -i /tmp/cudnn-local-repo-ubuntu1804-8.5.0.96_1.0-1_amd64.deb && \
#     cp /var/cudnn-local-repo-ubuntu1804-8.5.0.96/cudnn-local-7DA6B9A9-keyring.gpg /usr/share/keyrings/ && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends libcudnn8 libcudnn8-dev && \
#     rm -f /tmp/cudnn-local-repo-ubuntu1804-8.5.0.96_1.0-1_amd64.deb && \
#     rm -rf /var/lib/apt/lists/*

# # ----------------------------
# # Set CUDA environment variables (BEFORE conda installation)
# # ----------------------------
# RUN ln -s /usr/local/cuda-10.2 /usr/local/cuda
# ENV CUDA_HOME=/usr/local/cuda-10.2
# ENV CUDA_PATH=/usr/local/cuda-10.2
# ENV PATH=/usr/local/cuda-10.2/bin:${PATH}
# ENV LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH}

# ----------------------------
# Install Miniconda
# ----------------------------
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O Miniforge3-latest-Linux-x86_64.sh && \
    bash Miniforge3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniforge3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda init bash

# ----------------------------
# Create environment with Python 3.7
# ----------------------------
RUN /opt/conda/bin/mamba create -y -n point_env python=3.7 && \
    /opt/conda/bin/mamba clean -afy

# ----------------------------
# Upgrade pip
# ----------------------------
RUN /opt/conda/bin/mamba run -n point_env pip install --upgrade pip setuptools wheel

# ----------------------------
# Install PyTorch with CUDA 10.2
# ----------------------------
# RUN /opt/conda/bin/mamba run -n point_env pip install torch==1.10.2+cu102 torchvision==0.11.3+cu102 torchaudio==0.10.2 --extra-index-url https://download.pytorch.org/whl/cu102
RUN /opt/conda/bin/mamba run -n point_env pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# ----------------------------
# Install dependencies for PointBERT
# ----------------------------
RUN /opt/conda/bin/mamba run -n point_env pip install numpy pyarrow scipy tifffile pandas h5py easydict scikit-learn matplotlib tqdm

# ----------------------------
# Install PointBERT (from repo)
# ----------------------------
WORKDIR /workspace
RUN git clone https://github.com/Julie-tang00/Point-BERT.git && \
    cd Point-BERT && \
    /opt/conda/bin/mamba run -n point_env pip install -r requirements.txt

# replace all instances of extensions. with "" in the pointbert files
RUN sed -i 's/extensions\.//g' /workspace/Point-BERT/**.py
# # replace chamfer_dist with chamfer_distance in the pointbert files
# RUN sed -i 's/chamfer_dist/chamfer_distance/g' /workspace/Point-BERT/**.py

# replace "from extensions.chamfer_dist" with "from chamfer_distance import ChamferDistance as chamfer_dist"


RUN cat > pyproject.toml <<'PYPROJECT'
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "point-bert"
version = "0.1.0"
authors = [{name = "Your Name"}]
description = """Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling"""
readme = "README.md"
license = {text = "MIT"}
dependencies = []

[tool.setuptools.packages.find]
where = ["."]
PYPROJECT
RUN /opt/conda/bin/mamba run -n point_env pip install -e .


# ----------------------------
# Install KNN_CUDA with proper environment
# ----------------------------
# Note: This will install but won't be testable until runtime with GPU
RUN /opt/conda/bin/mamba run -n point_env pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# ----------------------------
# Install Pointnet2_ops (optional - uncomment if needed)
# ----------------------------
RUN /opt/conda/bin/mamba run -n point_env bash -c "\
    cd /tmp && \
    git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git && \
    cd Pointnet2_PyTorch && \
    git checkout b5ceb6d9ca0467ea34beb81023f96ee82228f626 && \
    cd pointnet2_ops_lib && \
    echo '__version__ = \"0.0.0\"' > pointnet2_ops/_version.py && \
    python setup.py install"


# ----------------------------
# Install local utils package
# ----------------------------
COPY . /workspace/utils/
RUN /opt/conda/bin/mamba run -n point_env pip install -e /workspace/utils/

# ----------------------------

# ----------------------------
# Verify CUDA is accessible (will show version but GPU ops won't work until runtime)
# ----------------------------
RUN nvcc --version && \
    echo "CUDA_HOME: $CUDA_HOME" && \
    ls -la /usr/local/cuda-11.7/bin/nvcc

# ----------------------------
# Create runtime test script
# ----------------------------
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate point_env\n\
echo "=== CUDA Runtime Check ==="\n\
python -c "import torch; print(f\"PyTorch version: {torch.__version__}\")"\n\
python -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")"\n\
python -c "import torch; print(f\"CUDA version: {torch.version.cuda}\")"\n\
echo "=== Attempting to import knn_cuda ==="\n\
python -c "import knn_cuda; print(f\"knn_cuda version: {knn_cuda.__version__}\")" || echo "knn_cuda import failed (expected without GPU)"\n\
' > /test_cuda.sh && chmod +x /test_cuda.sh

# Activate conda environment in bash
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate point_env" >> ~/.bashrc

# install other dependencies from the pointbert extensions
# ----------------------------
# RUN echo $(pwd)
# RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# RUN /opt/conda/bin/mamba run -n point_env pip install git+'https://github.com/otaheri/chamfer_distance'

# RUN cd /workspace/utils/Point-BERT/extensions/chamfer_dist && \
#     # /opt/conda/bin/mamba run -n point_env python setup.py install --user
#     /opt/conda/bin/mamba run -n point_env pip install -e /workspace/Point-BERT/extensions/chamfer_dist
# RUN cd /workspace/utils/Point-BERT/extensions/chamfer_dist && \
#     /opt/conda/bin/mamba run -n point_env python setup.py install --user && \
#     /opt/conda/bin/mamba run -n point_env pip install -e .
# RUN cd /workspace/utils/Point-BERT/extensions/emd && \
#     /opt/conda/bin/mamba run -n point_env python setup.py install --user && \
#     /opt/conda/bin/mamba run -n point_env pip install -e .

# ----------------------------
# Default command
# ----------------------------
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate point_env && bash"]

# ----------------------------
# End of Dockerfile
# ----------------------------
