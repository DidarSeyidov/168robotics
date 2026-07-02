"""
Publishes the two missing static transforms that connect base_link to the
camera frame, using the calibration values sent by your colleague:

  base_link -> lidar_top
  lidar_top -> cam_front

These are NOT in the bag's /tf tree (confirmed by view_frames), so they must
be republished live while you play back the bag. Run this launch file at the
same time as `ros2 bag play --clock`.

Written for Humble/Iron/Jazzy syntax (named --flags, quaternion only).
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    base_link_to_lidar_top = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_lidar_top',
        arguments=[
            '--x', '0.304',
            '--y', '0.0',
            '--z', '1.048',
            '--qx', '0.0',
            '--qy', '0.0',
            '--qz', '-0.0231236',
            '--qw', '0.9997326',
            '--frame-id', 'base_link',
            '--child-frame-id', 'lidar_top',
        ],
        parameters=[{'use_sim_time': True}],  # sync with bag clock
    )

    lidar_top_to_cam_front = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_top_to_cam_front',
        arguments=[
            '--x', '0.19644791362537',
            '--y', '0.129130179925657',
            '--z', '-0.310924551167351',
            '--qx', '-0.596487210506752',
            '--qy', '0.582829651789514',
            '--qz', '-0.378265530109409',
            '--qw', '0.401780777822727',
            '--frame-id', 'lidar_top',
            '--child-frame-id', 'cam_front',
        ],
        parameters=[{'use_sim_time': True}],  # sync with bag clock
    )

    return LaunchDescription([
        base_link_to_lidar_top,
        lidar_top_to_cam_front,
    ])