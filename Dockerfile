FROM ubuntu:20.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p $CONDA_DIR && \
    rm /miniconda.sh && \
    conda clean -afy

# Copy and create conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Set the environment (activate in CMD or shell later)
ENV CONDA_DEFAULT_ENV=pytorch
ENV PATH=$CONDA_DIR/envs/pytorch/bin:$PATH

# Copy code
COPY . /workspace
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8889

# Default command to run notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8889", "--allow-root"]
