from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'map_cfg', default_value='options.yaml'),
        DeclareLaunchArgument(
            'object_info_cfg', default_value='object_info.csv'),
        DeclareLaunchArgument(
            'tracker_cfg', default_value='coda.yaml'),

        Node(
            package='single_camera_tracking',
            executable='instance_segmentation.py',
            name='instance_segmentation',
            output='screen',
            arguments=['--yaml', LaunchConfiguration('tracker_cfg')],
        ),

        Node(
            package='single_camera_tracking',
            executable='tracking',
            name='tracking',
            output='screen',
            arguments=[LaunchConfiguration('tracker_cfg')],
        ),

        # Node(
        #     package='semantic_dsp_map',
        #     executable='mapping',
        #     name='mapping',
        #     output='screen',
        #     arguments=[
        #         LaunchConfiguration('map_cfg'),
        #         LaunchConfiguration('object_info_cfg'),
        #     ],
        # ),
    ])
