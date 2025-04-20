FROM python:3.7-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system build tools, curl, git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      libgl1 \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      ffmpeg \
      curl && \
    rm -rf /var/lib/apt/lists/*

# Install Rust toolchain >=1.74 via rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set workdir
WORKDIR /ros

# Install Python dependencies (including safetensors)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
 