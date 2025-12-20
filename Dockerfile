# Use the requested CUDA 12.8.1 Devel image
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PATH=/usr/local/cuda/bin:${PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
WORKDIR /app

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# 1. Install the PyTorch 2.8.0 Stack for CUDA 12.8
# This satisfies the CVE-2025-32434 security requirement (>2.6.0)
RUN pip3 install --no-cache-dir \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 2. Install ONNX Runtime and Model dependencies
# Note: Using version 1.20+ for compatibility with CUDA 12.8
RUN pip3 install --no-cache-dir \
    onnxruntime-gpu \
    transformers>=4.48.0 \
    accelerate \
    omegaconf \
    einops \
    lmdeploy \
    librosa \
    fastapi \
    uvicorn \
    pydantic \
    soundfile \
    numpy

# 3. Install MiraTTS specific git dependencies
RUN pip3 install --no-cache-dir \
    "fastaudiosr @ git+https://github.com/ysharma3501/FlashSR.git" \
    "ncodec @ git+https://github.com/ysharma3501/FastBiCodec.git"

# 4. Copy and Install the MiraTTS project
COPY mira /app/mira
COPY README.md /app/
COPY pyproject.toml /app/
RUN pip3 install --no-cache-dir --no-deps -e .

# 5. Copy API code
COPY app /app/app
RUN mkdir -p /app/models /app/data

EXPOSE 8000

# Health check to ensure GPU is visible
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]