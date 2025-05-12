# Use NVIDIA CUDA base image with Ubuntu 20.04
FROM nvidia/cuda:12.9.0-runtime-ubuntu20.04

# Arguments and environment
ARG ROS_DISTRO=foxy
ENV ROS_DISTRO=${ROS_DISTRO}
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install OS dependencies, Rust, and Python tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    gnupg2 \
    lsb-release \
    curl \
    apt-transport-https \
    ca-certificates \
    software-properties-common \
    python3-pip \
    git \
    build-essential \
    rustc \
    cargo \
    python3-empy \
  && rm -rf /var/lib/apt/lists/*

# Generate locale
RUN locale-gen en_US en_US.UTF-8

COPY ./requirements.txt .

# Install Python-based ROS tooling via pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Setup ROS 2 apt repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - \
 && echo "deb [arch=amd64] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    > /etc/apt/sources.list.d/ros2.list

# Install ROS 2 base packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-ros-base \
      ros-${ROS_DISTRO}-vision-msgs \
      libgl1-mesa-glx \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

# Create an (initially empty) workspace
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# Install PyTorch with CUDA support for neural networks
# RUN python3 -m pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Source ROS 2 in all shells
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /etc/bash.bashrc

# Default entrypoint: if 'src' is populated, install deps and build; then drop to bash
CMD ["/bin/bash", "-lc", "source /opt/ros/$ROS_DISTRO/setup.bash && if [ -n \"$(ls -A src)\" ]; then rosdep install --from-paths src --ignore-src -r -y && colcon build --symlink-install; fi && exec bash"]
