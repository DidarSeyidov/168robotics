"""Launch dsp_global_mapping against the recorded bag.

This is the simpler pipeline — no segmentation needed.
It saves the merged LiDAR point cloud to a PLY file as the robot moves.

Bag location: ~/semantic_map_ws/bag/
Key bag topics:
  /lidar/merged/points         → PointCloud2 (no RGB — handled automatically)
  /bnk/omninav_combine/out_lo  → Odometry → converted to PoseStamped

Play the bag in a separate terminal:
  ros2 bag play ~/semantic_map_ws/bag --clock

Output PLY: ~/semantic_map_ws/output/global_map.ply
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    odom_topic_arg  = DeclareLaunchArgument(
        'odom_topic',  default_value='/bnk/omninav_combine/out_lo')
    lidar_topic_arg = DeclareLaunchArgument(
        'lidar_topic', default_value='/lidar/merged/points')
    output_file_arg = DeclareLaunchArgument(
        'output_file',
        default_value='/home/didar/168robotics_ws/output/global_map.ply',
        description='Output PLY file path')
    map_x_arg   = DeclareLaunchArgument('map_range_x', default_value='200.0')
    map_y_arg   = DeclareLaunchArgument('map_range_y', default_value='200.0')
    map_z_arg   = DeclareLaunchArgument('map_range_z', default_value='30.0')
    voxel_arg   = DeclareLaunchArgument('voxel_size',  default_value='0.1')

    # Convert Odometry → PoseStamped
    odom_to_pose = Node(
        package='semantic_dsp_map',
        executable='odom_to_pose.py',
        name='odom_to_pose',
        parameters=[{
            'input_topic':  LaunchConfiguration('odom_topic'),
            'output_topic': '/bag/pose_stamped',
            'use_sim_time': True,
        }],
        output='screen',
    )

    global_mapping = Node(
        package='dsp_global_mapping',
        executable='global_mapping',
        name='global_mapping',
        # argv: points_topic pose_topic output_file range_x range_y range_z voxel write_color
        arguments=[
            LaunchConfiguration('lidar_topic'),
            '/bag/pose_stamped',
            LaunchConfiguration('output_file'),
            LaunchConfiguration('map_range_x'),
            LaunchConfiguration('map_range_y'),
            LaunchConfiguration('map_range_z'),
            LaunchConfiguration('voxel_size'),
            '0',   # write_color=0: lidar has no RGB fields
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription([
        odom_topic_arg,
        lidar_topic_arg,
        output_file_arg,
        map_x_arg, map_y_arg, map_z_arg,
        voxel_arg,
        odom_to_pose,
        global_mapping,
    ])
