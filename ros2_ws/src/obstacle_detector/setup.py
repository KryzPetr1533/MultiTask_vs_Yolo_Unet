from setuptools import setup
import os
from glob import glob

package_name = 'obstacle_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),

    ],
    install_requires=[
        'setuptools',
        'numpy',
        'segmentation-models-pytorch',
        'colcon-common-extensions',
        'rosdep',
        'rospkg',
        'catkin-pkg' ,
        'rosdistro',
        'vcstool' ,
        'Pillow' ,
        'opencv-python' ,
        'nuscenes-devkit',
        'cv_bridge',
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = obstacle_detector.camera_publisher:main',
            'image_retriever = obstacle_detector.image_retriever:main',
            'detector        = obstacle_detector.detector:main',
            'segmentor = obstacle_detector.segmentor:main',
            'metrics         = obstacle_detector.metrics:main',
        ],
    },
)
