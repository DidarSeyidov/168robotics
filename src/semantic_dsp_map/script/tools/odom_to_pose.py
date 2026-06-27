#!/usr/bin/env python3
"""Convert nav_msgs/Odometry to geometry_msgs/PoseStamped."""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class OdomToPose(Node):
    def __init__(self):
        super().__init__('odom_to_pose')
        self.declare_parameter('input_topic',  '/bnk/omninav_combine/out_lo')
        self.declare_parameter('output_topic', '/camera/pose_from_odom')

        input_topic  = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.pub = self.create_publisher(PoseStamped, output_topic, 10)
        self.sub = self.create_subscription(Odometry, input_topic, self._cb, 10)
        self.get_logger().info(f"Relaying {input_topic} → {output_topic}")

    def _cb(self, msg: Odometry):
        out = PoseStamped()
        out.header       = msg.header
        out.pose         = msg.pose.pose
        self.pub.publish(out)


def main():
    rclpy.init()
    node = OdomToPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
