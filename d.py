import argparse
import subprocess
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Docker runner for ROS2 Node')

    # Action flags: -b to build, -s to start the container
    parser.add_argument('-b', dest='actions', action='append_const', const="build",
                        help="Build the Docker image")
    parser.add_argument('-s', dest='actions', action='append_const', const="start",
                        help="Start the Docker container")
    
    # Parameters with updated defaults for the ROS2 image
    parser.add_argument('--image', action='store', type=str, default='ros2_image_mask_publisher',
                        help="Name of the Docker image")
    parser.add_argument('--hostname', action='store', type=str, default='ros2node',
                        help="Hostname for the Docker container")
    parser.add_argument('--name', action='store', type=str, default='ros2-docker-instance',
                        help="Name of the Docker container instance")
    
    # If you need to pass camera devices for additional hardware (optional)
    parser.add_argument('--cam1_ind', action='store', type=str, default='0',
                        help="Index of the first camera device (e.g., 0 for /dev/video0)")
    parser.add_argument('--cam2_ind', action='store', type=str, default='2',
                        help="Index of the second camera device (e.g., 2 for /dev/video2)")

    args = parser.parse_args()

    # Docker build command. This builds your Docker image from the Dockerfile in the current directory.
    if args.actions and "build" in args.actions:
        build_cmd = [
            "docker", "build",
            "-t", args.image,
            "."
        ]
        print("Building Docker image with command:")
        print(" ".join(build_cmd))
        subprocess.check_call(build_cmd)

    # Docker run command. This starts the container with host networking, GPU support, camera devices,
    # and mounts to support GUI applications if required.
    if args.actions and "start" in args.actions:
        # Use the current DISPLAY environment variable if available, otherwise default to :0.
        display_env = os.getenv("DISPLAY", ":0")
        run_cmd = [
            "docker", "run",
            "--rm", "-it",
            "--network", "host",
            "--gpus", "all",
            # Pass through camera devices; adjust or remove if not using cameras.
            # "--device", "/dev/video" + args.cam1_ind,
            # "--device", "/dev/video" + args.cam2_ind,
            "--hostname", args.hostname,
            # Pass through environment variables for X11 forwarding.
            "-e", f"DISPLAY={display_env}",
            "-e", "QT_X11_NO_MITSHM=1",
            # Mount the current directory; adjust as needed
            "-v", f"{os.getcwd()}:/ros2_ws",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
            "--name", args.name,
            args.image
        ]
        print("Starting Docker container with command:")
        print(" ".join(run_cmd))
        try:
            subprocess.check_call(run_cmd)
        except subprocess.CalledProcessError as exception:
            print("An error occurred while running the Docker container:")
            print(exception)
