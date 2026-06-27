from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'map_cfg', default_value='options_zed2.yaml'),
        DeclareLaunchArgument(
            'object_info_cfg', default_value='object_info_zed2.csv'),

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

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=[
                '-d',
                PathJoinSubstitution([
                    FindPackageShare('semantic_dsp_map'), 'rviz', 'zed2.rviz',
                ]),
            ],
            output='screen',
        ),
    ])
