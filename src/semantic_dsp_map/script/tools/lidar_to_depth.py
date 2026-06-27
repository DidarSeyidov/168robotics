#!/usr/bin/env python3
"""Project a LiDAR PointCloud2 onto a camera plane to generate a 32FC1 depth image.

Requires TF to be available between the point cloud's frame and the camera frame.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import numpy as np
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import Buffer, TransformListener


class LidarToDepthNode(Node):
    def __init__(self):
        super().__init__('lidar_to_depth')

        self.declare_parameter('lidar_topic',       '/lidar/merged/points')
        self.declare_parameter('camera_info_topic', '/cam/left/camera_info')
        self.declare_parameter('output_topic',      '/camera/depth_from_lidar')
        # Frame of the camera (destination for TF lookup).
        # Set to '' to use the frame_id from the CameraInfo message.
        self.declare_parameter('camera_frame', '')
        self.declare_parameter('max_depth', 80.0)

        lidar_topic       = self.get_parameter('lidar_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        output_topic      = self.get_parameter('output_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.max_depth    = self.get_parameter('max_depth').value

        self.K = None
        self.img_w = 0
        self.img_h = 0
        self.bridge = CvBridge()

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.depth_pub = self.create_publisher(Image, output_topic, 1)
        self.info_sub  = self.create_subscription(
            CameraInfo, camera_info_topic, self._info_cb, 1)
        self.lidar_sub = self.create_subscription(
            PointCloud2, lidar_topic, self._lidar_cb, 1)

        self.get_logger().info(f"lidar_to_depth: {lidar_topic} → {output_topic}")

    def _info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.K     = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.img_w = msg.width
            self.img_h = msg.height
            if not self.camera_frame:
                self.camera_frame = msg.header.frame_id
            self.get_logger().info(
                f"Camera intrinsics set. frame={self.camera_frame} "
                f"res={self.img_w}x{self.img_h}")

    def _lidar_cb(self, msg: PointCloud2):
        if self.K is None:
            return

        # ── TF lookup (use latest available transform to avoid timing issues) ──
        try:
            tf = self.tf_buffer.lookup_transform(
                self.camera_frame,
                msg.header.frame_id,
                Time(),                         # latest
                timeout=Duration(seconds=0.1))
        except Exception as exc:
            self.get_logger().warn(
                f"TF {msg.header.frame_id}→{self.camera_frame}: {exc}",
                throttle_duration_sec=2.0)
            return

        t = tf.transform.translation
        q = tf.transform.rotation
        tx, ty, tz = t.x, t.y, t.z
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        # Quaternion → 3×3 rotation matrix
        R = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)
        T = np.array([tx, ty, tz], dtype=np.float64)

        # ── Read XYZ from point cloud ──
        raw = list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
        if not raw:
            return
        pts = np.array(raw, dtype=np.float64)       # (N, 3)

        # ── Transform to camera frame ──
        pts_cam = (R @ pts.T).T + T                 # (N, 3)

        # Keep only points in front of the image plane
        mask = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[mask]
        if len(pts_cam) == 0:
            return

        # ── Project ──
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        u = (pts_cam[:, 0] / pts_cam[:, 2] * fx + cx).astype(np.int32)
        v = (pts_cam[:, 1] / pts_cam[:, 2] * fy + cy).astype(np.int32)
        d = pts_cam[:, 2].astype(np.float32)

        valid = (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h) & (d <= self.max_depth)
        u, v, d = u[valid], v[valid], d[valid]

        # ── Build depth image ──
        depth = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        # Where multiple points project to the same pixel, keep the nearest
        order = np.argsort(d)[::-1]                 # far → near
        depth[v[order], u[order]] = d[order]

        out = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        out.header = msg.header
        out.header.frame_id = self.camera_frame
        self.depth_pub.publish(out)


def main():
    rclpy.init()
    node = LidarToDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
