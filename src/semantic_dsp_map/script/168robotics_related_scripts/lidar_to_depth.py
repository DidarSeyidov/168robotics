#!/usr/bin/env python3
"""
ROS2 Humble — LidarToDepth node (fixed)
Fixes:
  1. cv_bridge NumPy 2.x incompatibility — replaced cv2_to_imgmsg with
     manual Image message construction (avoids cv_bridge entirely)
  2. CameraInfo intrinsics — reads from P matrix (rectified) not K matrix (raw fisheye)
  3. PointCloud2 structured array — properly unpacked before numpy cast
"""

import numpy as np
import cv2
import struct
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2


# ── Extrinsic: lidar_top -> cam_front (from colleague's YAML, confirmed direction) ──
def _quat_to_rot(qx, qy, qz, qw):
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),  1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),  1-2*(qx**2+qy**2)]
    ])

_R = _quat_to_rot(
    -0.596487210506752,
     0.582829651789514,
    -0.378265530109409,
     0.401780777822727
)
T_LIDAR_TO_CAM = np.eye(4, dtype=np.float64)
T_LIDAR_TO_CAM[:3, :3] = _R
T_LIDAR_TO_CAM[:3,  3] = [0.19644791362537, 0.129130179925657, -0.310924551167351]

# ── Depth range ────────────────────────────────────────────────────────────────
DEPTH_MIN = 0.3
DEPTH_MAX = 20.0
DILATION_KERNEL = 5


def numpy_to_image_msg(depth_np, header):
    """
    Convert float32 numpy array to sensor_msgs/Image without cv_bridge.
    Avoids the NumPy 2.x / cv_bridge ABI incompatibility entirely.
    """
    msg = Image()
    msg.header = header
    msg.height = depth_np.shape[0]
    msg.width = depth_np.shape[1]
    msg.encoding = '32FC1'
    msg.is_bigendian = False
    msg.step = depth_np.shape[1] * 4  # 4 bytes per float32
    msg.data = depth_np.astype(np.float32).tobytes()
    return msg


class LidarToDepth(Node):
    def __init__(self):
        super().__init__('lidar_to_depth')
        self.w = None
        self.h = None
        # Intrinsics filled from CameraInfo P matrix (rectified)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.create_subscription(
            CameraInfo, '/cam/front/camera_info', self.info_cb, 10)
        self.create_subscription(
            PointCloud2, '/lidar/merged/points', self.cloud_cb, 10)
        self.depth_pub = self.create_publisher(Image, '/camera_depth_image', 10)
        self.get_logger().info('LidarToDepth started. Waiting for CameraInfo...')

    def info_cb(self, msg):
        if self.w is not None:
            return
        self.w = msg.width
        self.h = msg.height

        # ── Use P matrix (rectified intrinsics), NOT K matrix (raw fisheye) ──
        # P is a 3x4 matrix stored row-major as 12 floats.
        # P[0,0]=fx  P[0,2]=cx  P[1,1]=fy  P[1,2]=cy
        if len(msg.p) >= 12 and msg.p[0] > 0:
            self.fx = msg.p[0]
            self.fy = msg.p[5]
            self.cx = msg.p[2]
            self.cy = msg.p[6]
            self.get_logger().info(
                f'Camera info received: {self.w}x{self.h} '
                f'(from P matrix) fx={self.fx:.2f} fy={self.fy:.2f} '
                f'cx={self.cx:.2f} cy={self.cy:.2f}')
        else:
            # Fallback to K matrix if P is empty
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().warn(
                f'P matrix empty, falling back to K matrix: '
                f'fx={self.fx:.2f} — verify image is rectified!')

    def cloud_cb(self, msg):
        if self.w is None or self.fx is None:
            self.get_logger().warn('No CameraInfo yet — skipping frame.',
                                   throttle_duration_sec=2.0)
            return

        # ── 1. Extract XYZ from PointCloud2 — handle structured array correctly ──
        pts_gen = pc2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)
        pts_list = list(pts_gen)
        if len(pts_list) == 0:
            return

        # pts_list is a list of named tuples or structured scalars — unpack explicitly
        pts = np.array(
            [(p[0], p[1], p[2]) for p in pts_list],
            dtype=np.float64
        )  # (N, 3)

        # ── 2. Transform lidar points into camera frame ──
        N = pts.shape[0]
        pts_h = np.hstack([pts, np.ones((N, 1), dtype=np.float64)])  # (N, 4)
        pts_cam = (T_LIDAR_TO_CAM @ pts_h.T).T                       # (N, 4)
        X = pts_cam[:, 0]
        Y = pts_cam[:, 1]
        Z = pts_cam[:, 2]

        # ── 3. Filter by depth range ──
        valid = (Z > DEPTH_MIN) & (Z < DEPTH_MAX)
        X, Y, Z = X[valid], Y[valid], Z[valid]
        if len(Z) == 0:
            return

        # ── 4. Project onto image plane ──
        u = (self.fx * X / Z + self.cx).astype(np.int32)
        v = (self.fy * Y / Z + self.cy).astype(np.int32)

        in_bounds = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        u, v, Z = u[in_bounds], v[in_bounds], Z[in_bounds]

        # ── 5. Fill depth image — nearest point wins ──
        depth = np.zeros((self.h, self.w), dtype=np.float32)
        order = np.argsort(Z)[::-1]   # far first, near overwrites
        depth[v[order], u[order]] = Z[order].astype(np.float32)

        # ── 6. Densify sparse lidar projection ──
        kernel = np.ones((DILATION_KERNEL, DILATION_KERNEL), np.uint8)
        depth = cv2.dilate(depth, kernel)

        # ── 7. Publish (no cv_bridge — avoids NumPy 2.x ABI issue) ──
        img_msg = numpy_to_image_msg(depth, msg.header)
        img_msg.header.frame_id = 'cam_front_optical_frame'
        self.depth_pub.publish(img_msg)


def main():
    rclpy.init()
    node = LidarToDepth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()