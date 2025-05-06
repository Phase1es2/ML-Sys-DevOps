# Use the official PyTorch image as the base image (with CUDA support)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy application files into the container
COPY main.py .
COPY requirements.txt .

# Upgrade pip and install required Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Define the default command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
