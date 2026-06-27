#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import cv2
import os
import fnmatch
import argparse
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as SciRot  # replaces tf.transformations
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2          # replaces sensor_msgs.point_cloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import open3d as o3d


def generate_point_cloud(depth_image, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    rows, cols = depth_image.shape
    x = np.linspace(0, cols - 1, cols)
    y = np.linspace(0, rows - 1, rows)
    x, y = np.meshgrid(x, y)

    x = (x - cx) * depth_image / fx
    y = (y - cy) * depth_image / fy
    z = depth_image

    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def generate_point_cloud_with_rgb(depth_image, rgb_image, K,
                                   translation=None, quaternion=None,
                                   max_depth=1000.0):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    rows, cols = depth_image.shape
    x = np.linspace(0, cols - 1, cols)
    y = np.linspace(0, rows - 1, rows)
    x, y = np.meshgrid(x, y)

    x = (x - cx) * depth_image / fx
    y = (y - cy) * depth_image / fy
    z = depth_image

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid_mask = z.flatten() <= max_depth
    points = points[valid_mask]

    if translation is not None and quaternion is not None:
        # quaternion is [x, y, z, w] — same convention as tf.transformations
        R = SciRot.from_quat(quaternion).as_matrix()
        points = (R @ points.T).T + translation

    rgb_flat = rgb_image.reshape(-1, 3)[valid_mask]
    r = rgb_flat[:, 2].astype(np.uint32)
    g = rgb_flat[:, 1].astype(np.uint32)
    b = rgb_flat[:, 0].astype(np.uint32)
    rgb_packed = ((r << 16) | (g << 8) | b).view(np.float32)

    return np.column_stack((points, rgb_packed))


def remove_outliers(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    pcd = pcd.select_by_index(ind)
    return np.asarray(pcd.points)


def read_pose_txt(pose_txt):
    with open(pose_txt, 'r') as f:
        lines = f.readlines()

    camera_to_imu = np.array([
        [ 0.0371783278, -0.0986182135,  0.9944306009, 1.5752681039],
        [ 0.9992675562, -0.0053553387, -0.0378902567, 0.0043914093],
        [ 0.0090621821,  0.9951109327,  0.0983468786, -0.6500000000],
        [0.0, 0.0, 0.0, 1.0],
    ])

    poses = []
    for line in lines:
        pose = line.split()

        if len(pose) == 13:
            frame_idx = int(pose[0])
            pose += ['0', '0', '0', '1']
            imu_to_world = np.array(pose[1:], dtype=np.float32).reshape(4, 4)
            cam0_to_world = imu_to_world @ camera_to_imu
        elif len(pose) == 17:
            frame_idx = int(pose[0])
            cam0_to_world = np.array(pose[1:], dtype=np.float32).reshape(4, 4)
        else:
            raise ValueError(f"Invalid number of elements in pose line: {len(pose)}")

        translation = cam0_to_world[:3, 3]
        # SciRot.from_matrix returns [x, y, z, w] via .as_quat()
        quaternion = SciRot.from_matrix(cam0_to_world[:3, :3]).as_quat()
        poses.append([frame_idx, translation, quaternion])

    return poses


class KittiDataReaderNode(Node):
    def __init__(self, args):
        super().__init__('kitti360_data_reader')
        self.args = args
        self.bridge = CvBridge()

        self.rgb_image_pub          = self.create_publisher(Image,       args.rgb_image_topic,            1)
        self.depth_image_pub        = self.create_publisher(Image,       args.depth_image_topic,          1)
        self.camera_pose_pub        = self.create_publisher(PoseStamped, args.camera_pose_topic,          1)
        self.semantic_seg_image_pub = self.create_publisher(Image,       args.semantic_seg_image_topic,   1)
        self.semantic_point_cloud_pub = self.create_publisher(
            PointCloud2, args.semantic_point_cloud_topic, 1)

        self.pose_data = read_pose_txt(args.pose_txt)
        self.get_logger().info(f"Number of poses = {len(self.pose_data)}")

        self.publish_pose_idx = 0
        self.first_frame_repeat_count = 0

        period = 1.0 / max(args.loop_rate, 0.001)
        self.timer = self.create_timer(period, self._publish_next_frame)

    def _publish_next_frame(self):
        if self.publish_pose_idx >= len(self.pose_data):
            self.get_logger().info("All frames published. Shutting down.")
            self.timer.cancel()
            return

        pose = self.pose_data[self.publish_pose_idx]

        if self.publish_pose_idx < self.args.starting_frame_idx:
            self.publish_pose_idx += 1
            return

        if self.publish_pose_idx > self.args.stop_frame_idx:
            self.timer.cancel()
            return

        # Repeat the first frame for initialisation
        if (self.publish_pose_idx == self.args.starting_frame_idx + 1 and
                self.first_frame_repeat_count < self.args.repeat_first_frame):
            self.publish_pose_idx -= 1
            self.first_frame_repeat_count += 1

        frame_idx, translation, quaternion = pose

        rgb_path   = os.path.join(self.args.rgb_dir,   str(frame_idx).zfill(10) + '.png')
        depth_path = os.path.join(self.args.depth_dir, str(frame_idx).zfill(10) + '.npy')

        rgb_image   = cv2.imread(rgb_path)
        depth_image = np.load(depth_path)

        stamp = self.get_clock().now().to_msg()

        if rgb_image is None:
            raise ValueError("RGB Image is None")
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image)
        rgb_msg.header.stamp = stamp
        self.rgb_image_pub.publish(rgb_msg)

        if depth_image is None:
            raise ValueError("Depth Image is None")
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
        depth_msg.header.stamp = stamp
        self.depth_image_pub.publish(depth_msg)

        pose_msg = PoseStamped()
        pose_msg.header.stamp    = stamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(translation[0])
        pose_msg.pose.position.y = float(translation[1])
        pose_msg.pose.position.z = float(translation[2])
        pose_msg.pose.orientation.x = float(quaternion[0])
        pose_msg.pose.orientation.y = float(quaternion[1])
        pose_msg.pose.orientation.z = float(quaternion[2])
        pose_msg.pose.orientation.w = float(quaternion[3])
        self.camera_pose_pub.publish(pose_msg)

        if self.args.publish_semantic_seg:
            seg_path = os.path.join(self.args.semantic_seg_dir,
                                    str(frame_idx).zfill(10) + '.png')
            semantic_seg_image = cv2.imread(seg_path)
            if semantic_seg_image is None:
                raise ValueError("Semantic Segmentation Image is None")

            seg_msg = self.bridge.cv2_to_imgmsg(semantic_seg_image)
            seg_msg.header.stamp = stamp
            self.semantic_seg_image_pub.publish(seg_msg)

            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored    = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(semantic_seg_image, 0.7, depth_colored, 0.3, 0)
            cv2.imshow('Overlay', overlay)
            cv2.waitKey(1)

            if self.args.publish_semantic_pointcloud:
                K = np.array([
                    [552.554261, 0.000000,   682.049453],
                    [0.000000,   552.554261, 238.769549],
                    [0.000000,   0.000000,   1.000000],
                ])
                points = generate_point_cloud_with_rgb(
                    depth_image, semantic_seg_image, K,
                    translation, quaternion, max_depth=30.0)

                header = Header()
                header.stamp    = stamp
                header.frame_id = "map"
                fields = [
                    PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
                    PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
                    PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
                    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
                ]
                pc2_msg = pc2.create_cloud(header, fields, points)
                self.semantic_point_cloud_pub.publish(pc2_msg)

        if self.publish_pose_idx % 10 == 0:
            self.get_logger().info(
                f"Progress: {self.publish_pose_idx} / {len(self.pose_data)}")

        self.publish_pose_idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dir',
        default='/media/cc/Elements/KITTI-360/data_2d_test_slam/test_2/'
                '2013_05_28_drive_0004_sync/image_00/data_rect')
    parser.add_argument('--depth_dir',
        default='/media/cc/Elements/KITTI-360/data_2d_test_slam/depth_sgm/test_2/'
                '2013_05_28_drive_0004_sync/depth')
    parser.add_argument('--semantic_seg_dir',
        default='/media/cc/Elements/KITTI-360/data_2d_test_slam/segmentation_cmnext/test_2/'
                '2013_05_28_drive_0004_sync')
    parser.add_argument('--pose_txt',
        default='/media/cc/Elements/KITTI-360/data_2d_test_slam/poses/test_2_poses.txt')
    parser.add_argument('--starting_frame_idx', type=int, default=0)
    parser.add_argument('--stop_frame_idx',     type=int, default=1000000)
    parser.add_argument('--rgb_image_topic',          default='/kitti360/cam0/rgb')
    parser.add_argument('--depth_image_topic',        default='/kitti360/cam0/depth')
    parser.add_argument('--camera_pose_topic',        default='/kitti360/pose_cam')
    parser.add_argument('--semantic_seg_image_topic', default='/kitti360/cam0/semantic')
    parser.add_argument('--semantic_point_cloud_topic', default='/kitti360/semantic_point')
    parser.add_argument('--loop_rate',              type=float, default=0.4)
    parser.add_argument('--publish_semantic_seg',   type=bool,  default=True)
    parser.add_argument('--publish_semantic_pointcloud', type=bool, default=True)
    parser.add_argument('--repeat_first_frame',     type=int,   default=2)
    args = parser.parse_args()

    rclpy.init()
    node = KittiDataReaderNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
