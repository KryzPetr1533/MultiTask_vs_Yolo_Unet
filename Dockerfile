FROM ros:humble

# Set a cool terminal prompt
ENV TERM=xterm-256color
RUN echo "PS1='\e[92m\u\e[0m@\e[94m\h\e[0m:\e[35m\w\e[0m# '" >> /root/.bashrc

# Set workdir
WORKDIR /ros

# Add deadsnakes PPA and install Python 3.7 and its venv module
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.7 python3.7-venv

# Install editor tools, camera utils, and CUDA build dependencies
RUN apt-get update && apt-get install -y \
    tmux \
    vim \
    nano \
    v4l-utils \
    build-essential \
    python3-colcon-clean \
    wget \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    python3-pip && \
    python3.7 -m pip install --upgrade pip && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-6 && \
    rm -rf /var/lib/apt/lists/*
    
# Set environment variables for CUDA
ENV PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Copy requirements and additional files
COPY requirements.txt .
# COPY /var/tmp/cityscapes/* /var/cityscapes/

# Create a Python virtual environment using Python 3.7
RUN python3.7 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies from requirements.txt using pip from our 3.7 environment
RUN pip install --no-cache-dir -r requirements.txt

# Configure shell environment: ROS, virtualenv activation, PYTHONPATH and ROS_PYTHON_VERSION for ROS nodes
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /opt/venv/bin/activate" >> /root/.bashrc && \
    echo "export PYTHONPATH=\$VIRTUAL_ENV/lib/python3.7/site-packages:\$PYTHONPATH" >> /root/.bashrc && \
    echo "export ROS_PYTHON_VERSION=3" >> /root/.bashrc && \
    touch /root/setup.sh
# Optional: Uncomment if needed 
# RUN echo "source /ros/install/setup.bash" >> /root/.bashrc

# Launch terminal
CMD ["/bin/bash"]
