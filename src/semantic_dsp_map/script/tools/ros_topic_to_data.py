#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import message_filters
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

folder1 = "/media/clarence/Clarence/semantic_dsp_bags/data_for_Symphonies/zed2/depth"
folder2 = "/media/clarence/Clarence/semantic_dsp_bags/data_for_Symphonies/zed2/rgb"

os.makedirs(folder1, exist_ok=True)
os.makedirs(folder2, exist_ok=True)


class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()

        depth_sub = message_filters.Subscriber(self, Image, "/camera/depth_repub")
        rgb_sub   = message_filters.Subscriber(self, Image, "/zed2/left/rgb/image")

        ts = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, rgb_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.callback)

        self.get_logger().info("Started synchronized image saver node")

    def callback(self, depth_msg, rgb_msg):
        # Build a nanosecond timestamp from sec + nanosec (replaces ROS1 .to_nsec())
        depth_nsec = depth_msg.header.stamp.sec * 1_000_000_000 + depth_msg.header.stamp.nanosec
        rgb_nsec   = rgb_msg.header.stamp.sec   * 1_000_000_000 + rgb_msg.header.stamp.nanosec

        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            depth_image = cv2.resize(depth_image, (1408, 376), interpolation=cv2.INTER_NEAREST)
            depth_filename = os.path.join(folder1, f"depth_{depth_nsec}.npy")
            np.save(depth_filename, depth_image)
            self.get_logger().info(f"Depth image saved to {depth_filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save depth image: {e}")

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            rgb_image = cv2.resize(rgb_image, (1408, 376))
            rgb_filename = os.path.join(folder2, f"rgb_{rgb_nsec}.png")
            cv2.imwrite(rgb_filename, rgb_image)
            self.get_logger().info(f"RGB image saved to {rgb_filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save RGB image: {e}")


def main():
    rclpy.init()
    node = ImageSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
