from setuptools import find_packages, setup


package_name = 'yolo_detector_viewer'


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='ROS 2 image subscriber that runs YOLO detection and shows OpenCV results.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_viewer = yolo_detector_viewer.detect_viewer:main',
            'pruned_detect_viewer = yolo_detector_viewer.pruned_detect_viewer:main',
            'yolo_publisher = yolo_detector_viewer.yolo_publisher:main',
        ],
    },
)
