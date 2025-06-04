FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    wget \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libfreetype6-dev \
    libxft-dev \
    tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip
RUN pip install \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    mne \
    nibabel \
    nilearn \
    jupyterlab

# Set up working directory
WORKDIR /app
COPY . /app

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]