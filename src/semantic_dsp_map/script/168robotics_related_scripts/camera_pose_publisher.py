#!/usr/bin/env python3
"""
camera_pose_publisher.py

Once the bag is playing AND static_camera_tf.launch.py is running (so the
tf tree is fully connected: local_map -> odom -> base_link -> lidar_top ->
cam_front), this node looks up the camera's pose in whatever reference
frame you want and republishes it as geometry_msgs/PoseStamped on
/camera_pose - i.e. it recreates the topic your bag never recorded.

Usage (note use_sim_time:=true - required so this node's clock tracks the
bag's /clock topic instead of your wall clock):
    ros2 run <your_pkg> camera_pose_publisher.py --ros-args \
        -p use_sim_time:=true -p reference_frame:=local_map \
        -p camera_frame:=cam_front

Also start `ros2 bag play your_bag --clock` (the --clock flag is what
publishes /clock in the first place - without it use_sim_time has nothing
to sync to).

If you just want to sanity-check the chain without writing/running any
code at all, you can skip this node entirely and just run:
    ros2 run tf2_ros tf2_echo local_map cam_front
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
import rclpy.parameter
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException


class CameraPosePublisher(Node):
    def __init__(self):
        super().__init__(
        'camera_pose_publisher',
        parameter_overrides=[
            rclpy.parameter.Parameter(
                'use_sim_time',
                rclpy.parameter.Parameter.Type.BOOL,
                True
            )
        ]
    )

        # REQUIRED when driving from a bag: makes self.get_clock() follow the
        # /clock topic published by `ros2 bag play --clock` instead of your
        # wall clock. Without this, timestamps you compare against are
        # meaningless relative to the bag data.

        self.declare_parameter('reference_frame', 'local_map')
        self.declare_parameter('camera_frame', 'cam_front')
        # The frame whose incoming /tf messages we trigger a lookup on.
        # Pick whichever dynamic frame updates at the rate you actually want
        # camera_pose published at - base_link updates at ~24 Hz per your
        # view_frames output.
        self.declare_parameter('trigger_child_frame', 'base_link')

        self.reference_frame = self.get_parameter('reference_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.trigger_child_frame = self.get_parameter('trigger_child_frame').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub = self.create_publisher(PoseStamped, '/camera_pose', 10)

        # Event-driven instead of a free-running timer: we only compute a
        # camera_pose when a *new* dynamic transform actually arrives from
        # the bag, and we look up cam_front's pose AT THAT EXACT STAMP -
        # not "whatever is freshest" - so camera_pose is pinned 1:1 to the
        # bag's own timeline with zero drift.
        self.tf_sub = self.create_subscription(
            TFMessage, '/tf', self.on_tf, 50
        )

        self.get_logger().info(
            f'Publishing {self.camera_frame} pose in {self.reference_frame} '
            f'frame on /camera_pose, triggered by {self.trigger_child_frame} updates'
        )

    def on_tf(self, msg: TFMessage):
        stamp = None
        for t in msg.transforms:
            if t.child_frame_id == self.trigger_child_frame:
                stamp = t.header.stamp
                break
        if stamp is None:
            return  # this /tf message didn't contain our trigger frame

        try:
            # Look up the transform AT THE TRIGGER'S OWN TIMESTAMP, with a
            # short timeout so we block briefly if lidar_top/cam_front's
            # static transform hasn't been registered by the listener yet
            # (can happen for a few ms right at startup).
            tf = self.tf_buffer.lookup_transform(
                self.reference_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
        except (LookupException, ExtrapolationException) as e:
            self.get_logger().warn(f'tf lookup failed: {e}', throttle_duration_sec=2.0)
            return

        msg_out = PoseStamped()
        msg_out.header.stamp = stamp  # exactly the bag's own timestamp
        msg_out.header.frame_id = self.reference_frame
        msg_out.pose.position.x = tf.transform.translation.x
        msg_out.pose.position.y = tf.transform.translation.y
        msg_out.pose.position.z = tf.transform.translation.z
        msg_out.pose.orientation = tf.transform.rotation

        self.pub.publish(msg_out)


def main():
    rclpy.init()
    node = CameraPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
