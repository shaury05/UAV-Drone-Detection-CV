# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies (ffmpeg is crucial for video processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep the container running in the background
CMD ["tail", "-f", "/dev/null"]