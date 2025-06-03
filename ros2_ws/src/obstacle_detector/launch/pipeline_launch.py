from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare arguments
    declare_dataroot = DeclareLaunchArgument(
        'dataroot',
        default_value='/var/tmp/full_nuImages',
        description='Path to NuImages dataroot'
    )
    declare_version = DeclareLaunchArgument(
        'version',
        default_value='v1.0-mini',
        description='NuImages version'
    )

    return LaunchDescription([
        declare_dataroot,
        declare_version,
        Node(
            package='obstacle_detector',
            executable='image_retriever',
            name='image_retriever',
            output='screen',
            parameters=[{
                'dataroot': LaunchConfiguration('dataroot'),
                'version': LaunchConfiguration('version'),
            }],
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
