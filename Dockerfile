# Start with an Ubuntu base image with CUDA support
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglew-dev \
    libglfw3 \
    x11-apps \ 
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    ffmpeg \
    "dm-acme[envs]" \
    "dm_control>=1.0.16" \
    mujoco

# Create output directory
RUN mkdir output_videos

# Set working directory
WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt 

CMD ["/bin/bash"]
