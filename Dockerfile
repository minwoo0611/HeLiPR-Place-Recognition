# Start from the specified PyTorch base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CPATH=/usr/local/cuda/include:$CPATH2

# Update and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libopenblas-dev \
    libopenexr-dev \
    libeigen3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-setuptools \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools==59.5.0

# Install Python packages
RUN pip install \
    pandas \
    tensorboard \
    numpy==1.24.1 \
    laspy \
    pytest \
    addict \
    pytorch-metric-learning==0.9.97 \
    yapf==0.40.1 \
    bitarray==1.6.0 \
    h5py \
    transforms3d \
    open3d \
    tqdm \
    einops

RUN pip install bagpy \
    utm \
    pyexr \
    pyntcloud
    
# Copy MinkowskiEngine source code into the image
COPY MinkowskiEngine /opt/MinkowskiEngine

# Build and install MinkowskiEngine
WORKDIR /opt/MinkowskiEngine
RUN python setup.py install --blas=openblas --force_cuda

# Clean up
WORKDIR /
RUN rm -rf /opt/MinkowskiEngine

# Set the default command
CMD ["bash"]
