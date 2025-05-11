# Base image with Python
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create tmpdir inside container with more space
RUN mkdir -p /mnt/block/pip_tmp
ENV TMPDIR=/mnt/block/pip_tmp

# Optionally set pip cache to larger disk
ENV PIP_CACHE_DIR=/mnt/block/pip_cache

# Copy app code and install dependencies
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose FastAPI port
EXPOSE 8001

# Start app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]