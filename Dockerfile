# Use the official ROS 2 Humble base image
FROM ros:humble

# Install additional packages and build tools.
# - python3-colcon-common-extensions: for building with colcon.
# - ros-humble-cv-bridge: for ROS 2 cv_bridge package.
# - libopencv-dev: provides OpenCV libraries for image processing.
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    ros-humble-cv-bridge \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a ROS 2 workspace directory.
WORKDIR /root/ros2_ws/src

# Copy your package(s) into the workspace.
# Adjust this if you have multiple packages.
COPY image_node ./image_node

# Move to the root of the workspace.
WORKDIR /root/ros2_ws

# Build the workspace. Source ROS 2 environment first.
RUN . /opt/ros/humble/setup.sh && colcon build

# Set up the entrypoint to source ROS 2 and workspace setup files.
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /root/ros2_ws/install/setup.bash && exec \"$@\"", "--"]

# Default command to run your node.
CMD ["ros2", "run", "image_mask_publisher", "image_mask_publisher_node"]
