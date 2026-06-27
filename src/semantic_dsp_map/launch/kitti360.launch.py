from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'map_cfg', default_value='options_kitti360.yaml'),
        DeclareLaunchArgument(
            'object_info_cfg', default_value='object_info_kitti360.csv'),

        Node(
            package='semantic_dsp_map',
            executable='mapping',
            name='mapping',
            output='screen',
            arguments=[
                LaunchConfiguration('map_cfg'),
                LaunchConfiguration('object_info_cfg'),
            ],
        ),
    ])
