from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'out_points_topic', default_value='/occupied_point'),
        DeclareLaunchArgument(
            'out_pose_topic', default_value='/map_pose'),
        DeclareLaunchArgument(
            'out_file',
            default_value='/home/cc/chg_ws/ros_ws/semantic_map_coda_ws/src/dsp_global_mapping/data/result.ply'),
        DeclareLaunchArgument(
            'map_range_x', default_value='38.4'),
        DeclareLaunchArgument(
            'map_range_y', default_value='38.4'),
        DeclareLaunchArgument(
            'map_range_z', default_value='38.4'),
        DeclareLaunchArgument(
            'voxel_size', default_value='0.15'),
        DeclareLaunchArgument(
            'write_color', default_value='1'),
        DeclareLaunchArgument(
            'object_csv',
            default_value='/home/cc/chg_ws/ros_ws/semantic_map_coda_ws/src/dsp_global_mapping/cfg/object_info_kitti360.csv'),

        Node(
            package='dsp_global_mapping',
            executable='global_mapping',
            name='global_mapping',
            output='screen',
            arguments=[
                LaunchConfiguration('out_points_topic'),
                LaunchConfiguration('out_pose_topic'),
                LaunchConfiguration('out_file'),
                LaunchConfiguration('map_range_x'),
                LaunchConfiguration('map_range_y'),
                LaunchConfiguration('map_range_z'),
                LaunchConfiguration('voxel_size'),
                LaunchConfiguration('write_color'),
                LaunchConfiguration('object_csv'),
            ],
        ),
    ])
