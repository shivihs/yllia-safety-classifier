ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UNSLOTH_DISABLE_STATISTICS=1 \
    BNB_CUDA_VERSION=130 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo && \
    python -m pip install --upgrade --no-cache-dir transformers accelerate peft trl datasets bitsandbytes

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# PEFT/Transformers may detect an incompatible torchao version in NVIDIA images.
# This server does not use torchao quantization, so remove it to avoid startup failures.
RUN python -m pip uninstall -y torchao || true

COPY app.py /app/app.py
COPY index.html /app/index.html

EXPOSE 11434

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "11434"]
