from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1) grab and publish GT image/masks/boxes
        Node(
            package='obstacle_detector',
            executable='image_retriever',
            name='image_retriever',
            output='screen',
        ),

        # 2) run your detector on the raw image
        Node(
            package='obstacle_detector',
            executable='detector',
            name='detector',
            output='screen',
        ),

        # 3) (later) run your segmentor on the raw image
        Node(
            package='obstacle_detector',
            executable='segmentor',
            name='segmentor',
            output='screen',
        ),

        # 4) (later) compute metrics by matching stamps
        Node(
            package='obstacle_detector',
            executable='metrics',
            name='metrics',
            output='screen',
        ),
    ])
