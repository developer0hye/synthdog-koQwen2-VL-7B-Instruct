# Use the specified PyTorch base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install required Python packages
RUN pip install \
    transformers \
    accelerate \
    scikit-learn \
    optimum \
    auto-gptq \
    qwen-vl-utils \
    datasets \
    unsloth \
    "Pillow>=9.4.0"

# Set the default working directory
WORKDIR /workspace

# Default command
CMD ["bash"]