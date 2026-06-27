#!/usr/bin/env python3
"""Publish an empty MaskGroup in sync with each incoming depth image.

This lets semantic_dsp_map/mapping run in static-environment mode (no tracked
objects) when no segmentation pipeline is available — useful for quick tests
with a raw LiDAR/odometry bag.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from mask_kpts_msgs.msg import MaskGroup


class DummyMaskPublisher(Node):
    def __init__(self):
        super().__init__('dummy_mask_publisher')

        self.declare_parameter('depth_topic',  '/camera/depth_from_lidar')
        self.declare_parameter('output_topic', '/mask_group_bag')

        depth_topic  = self.get_parameter('depth_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.pub = self.create_publisher(MaskGroup, output_topic, 1)
        self.sub = self.create_subscription(Image, depth_topic, self._cb, 1)
        self.get_logger().info(
            f"Publishing empty MaskGroup on {output_topic} "
            f"(triggered by {depth_topic})")

    def _cb(self, depth_msg: Image):
        out = MaskGroup()
        out.header  = depth_msg.header
        out.objects = []
        self.pub.publish(out)


def main():
    rclpy.init()
    node = DummyMaskPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
