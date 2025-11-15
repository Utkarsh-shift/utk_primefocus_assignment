# ---------------------------------------------------------
# Base image: Ubuntu + Python 3.11 (official + minimal)
# ---------------------------------------------------------
FROM python:3.11-slim

# ---------------------------------------------------------
# System dependencies required for:
# - dlib / face_recognition
# - opencv
# - whisper (FFmpeg)
# - facenet-pytorch
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    pkg-config \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Install Ollama (optional local LLM support)
# ---------------------------------------------------------
# Comment out if you don't need Ollama inside container
RUN curl -fsSL https://ollama.com/install.sh | bash || true

# ---------------------------------------------------------
# Create working directory
# ---------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------
# Copy requirements file
# ---------------------------------------------------------
COPY requirements.txt /app/requirements.txt

# ---------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt

# facenet-pytorch must be installed separately
RUN pip install facenet-pytorch

# ---------------------------------------------------------
# Copy your full project code
# ---------------------------------------------------------
COPY . /app

# ---------------------------------------------------------
# Default command (can be overridden)
# ---------------------------------------------------------
CMD [ "bash" ]
