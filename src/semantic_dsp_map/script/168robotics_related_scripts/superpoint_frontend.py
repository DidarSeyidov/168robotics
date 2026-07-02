#!/usr/bin/env python3
"""
superpoint_frontend.py  —  ROS2 Humble
Superpoint Mode front-end for semantic_dsp_map.

Subscribes:
  /cam/front/image_raw          sensor_msgs/Image   (decompressed RGB)
  /camera_depth_image           sensor_msgs/Image   (32FC1 metres, from lidar_to_depth.py)
  /camera_pose                  geometry_msgs/PoseStamped  (world frame camera pose)

Publishes:
  /mask_group_super_glued       mask_kpts_msgs/MaskGroup

CRITICAL CORRECTNESS NOTE (from reading semantic_dsp_map.h directly):
  For SETTING==2 (Superpoint mode), kpts_current and kpts_previous must be
  3D points in the WORLD/GLOBAL coordinate frame — NOT pixel coords, NOT
  camera frame.  The mapping node runs RANSAC on these 3D pairs directly.
  Pipeline per keypoint match:
    (u,v) pixel  +  depth[v,u]  →  3D camera frame  →  world frame via camera_pose

REQUIREMENTS:
  pip install ultralytics lightglue torch torchvision
  (your mask_kpts_msgs ROS2 package must be built in your workspace)

CONFIGURATION (set these to match your setup):
  IMAGE_TOPIC, DEPTH_TOPIC, POSE_TOPIC  — input topics
  FX, FY, CX, CY                        — from /cam/front/camera_info
  YOLO_MODEL                             — path or name of yolov8*-seg.pt
  MAX_KEYPOINTS                          — SuperPoint keypoint limit per frame
  MIN_DEPTH, MAX_DEPTH                   — valid depth range in metres
  MIN_KPTS_FOR_DYNAMIC                   — minimum matched kpts to trust motion estimation
"""

import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from ultralytics import YOLO
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from mask_kpts_msgs.msg import MaskGroup, MaskKpts, Keypoint


# ── Topic names ────────────────────────────────────────────────────────────────
IMAGE_TOPIC = '/cam/front/image_raw'
DEPTH_TOPIC = '/camera_depth_image'
POSE_TOPIC  = '/camera_pose'
OUTPUT_TOPIC = '/mask_group_super_glued'  # exact name expected by mapping node

# ── Camera intrinsics (from /cam/front/camera_info) ───────────────────────────
# These match your cam_front.json
FX = 516.9271402796466
FY = 516.9271402796466
CX = 611.541205740157
CY = 328.2243027445102

# ── Depth range ────────────────────────────────────────────────────────────────
MIN_DEPTH = 0.3    # metres — matches lidar_to_depth.py
MAX_DEPTH = 20.0

# ── Model settings ─────────────────────────────────────────────────────────────
YOLO_MODEL      = 'yolov8n-seg.pt'   # nano=fastest; yolov8m-seg.pt = better accuracy
MAX_KEYPOINTS   = 512
MIN_KPTS_FOR_DYNAMIC = 4  # RANSAC in mapping node needs >=4 valid 3D point pairs

# ── COCO class → your semantic label mapping ───────────────────────────────────
# dynamic (class 5 in your spec): people and vehicles
DYNAMIC_COCO = {0, 1, 2, 3, 4, 5, 6, 7, 8}   # person, bicycle, car, motorcycle,
                                                # airplane, bus, train, truck, boat
# static obstacle (class 3): furniture, objects
STATIC_COCO  = {56, 57, 58, 59, 60, 61, 62, 63, 64}

# grass/ground (class 4): COCO has no direct grass class.
# Use a SegFormer ADE20K model for this — placeholder for now.
# If you add SegFormer, label its "grass"/"ground" output as label="grass"
# with track_id=65535 (static sentinel), mask = pixel-value label image.

def coco_to_label(cls_id: int):
    if cls_id in DYNAMIC_COCO: return "dynamic"
    if cls_id in STATIC_COCO:  return "static_obj"  # non-background static obstacle
    return None  # background — skip


# ── Geometry helpers ───────────────────────────────────────────────────────────

def pose_to_matrix(pose_msg: PoseStamped) -> np.ndarray:
    """Convert PoseStamped to 4x4 camera-to-world transform matrix."""
    p = pose_msg.pose.position
    q = pose_msg.pose.orientation
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [p.x, p.y, p.z]
    return T


def unproject_pixel(u: float, v: float, depth: float,
                    camera_to_world: np.ndarray) -> np.ndarray | None:
    """
    Lift a single 2D pixel + depth value into a 3D world-frame point.
    Returns None if depth is invalid.
    """
    if depth <= MIN_DEPTH or depth >= MAX_DEPTH or np.isnan(depth):
        return None
    # Camera frame (pinhole inverse)
    X = (u - CX) * depth / FX
    Y = (v - CY) * depth / FY
    Z = depth
    pt_cam = np.array([X, Y, Z, 1.0], dtype=np.float64)
    # World frame
    pt_world = camera_to_world @ pt_cam
    return pt_world[:3]


# ── Simple IoU tracker (fallback when YOLO track() IDs are unavailable) ────────

class SimpleTracker:
    def __init__(self, iou_thresh=0.4):
        self.next_id = 1
        self.prev_boxes = {}
        self.iou_thresh = iou_thresh

    def update(self, boxes_xyxy: np.ndarray) -> list:
        if len(boxes_xyxy) == 0:
            self.prev_boxes = {}
            return []
        ids, used = [], set()
        for box in boxes_xyxy:
            best_id, best_iou = None, self.iou_thresh
            for tid, pb in self.prev_boxes.items():
                if tid in used: continue
                iou = self._iou(box, pb)
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used.add(best_id)
            ids.append(best_id)
        self.prev_boxes = {tid: box for tid, box in zip(ids, boxes_xyxy)}
        return ids

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0: return 0.0
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


# ── Main node ──────────────────────────────────────────────────────────────────

class SuperpointFrontend(Node):
    def __init__(self):
        super().__init__('superpoint_frontend')
        self.bridge  = CvBridge()
        self.tracker = SimpleTracker()

        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_str)
        self.get_logger().info(f'Using device: {device_str}')

        # YOLOv8 instance segmentation + built-in tracking
        self.yolo = YOLO(YOLO_MODEL)

        # SuperPoint + LightGlue
        self.extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(self.device)
        self.matcher   = LightGlue(features='superpoint').eval().to(self.device)

        # State from previous frame
        self.prev_feats       = None   # SuperPoint features
        self.prev_depth       = None   # depth image (np.ndarray float32)
        self.prev_cam_to_world = None  # 4x4 camera-to-world matrix

        self.pub = self.create_publisher(MaskGroup, OUTPUT_TOPIC, 10)

        # Synchronise RGB + depth + pose — approximate time sync
        # (depth and RGB are published at different times since lidar and
        #  camera are not hardware-synced)
        self.rgb_sub   = Subscriber(self, Image,        IMAGE_TOPIC)
        self.depth_sub = Subscriber(self, Image,        DEPTH_TOPIC)
        self.pose_sub  = Subscriber(self, PoseStamped,  POSE_TOPIC)

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.pose_sub],
            queue_size=10, slop=0.05)  # 50ms tolerance
        self.sync.registerCallback(self.callback)

        self.get_logger().info('SuperpointFrontend ready.')

    # ── Synchronized callback ──────────────────────────────────────────────────
    def callback(self, rgb_msg: Image, depth_msg: Image, pose_msg: PoseStamped):

        # 1. Convert messages to numpy
        bgr   = self.bridge.imgmsg_to_cv2(rgb_msg,   desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        cam_to_world = pose_to_matrix(pose_msg)  # 4x4 current frame

        # ── 2. Instance segmentation ──────────────────────────────────────────
        results    = self.yolo.track(rgb, persist=True, verbose=False,
                                     conf=0.3, iou=0.5)
        result     = results[0]
        detections = []  # list of (mask_binary, label, track_id, box_xyxy)

        if result.masks is not None and result.boxes is not None:
            seg_masks  = result.masks.data.cpu().numpy()        # (N, H, W)
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()        # (N, 4)
            cls_ids    = result.boxes.cls.cpu().numpy().astype(int)
            yolo_ids   = result.boxes.id
            if yolo_ids is not None:
                track_ids = yolo_ids.cpu().numpy().astype(int).tolist()
            else:
                track_ids = self.tracker.update(boxes_xyxy)

            H, W = bgr.shape[:2]
            for mask, cls_id, tid, box in zip(seg_masks, cls_ids,
                                               track_ids, boxes_xyxy):
                label = coco_to_label(cls_id)
                if label is None:
                    continue
                mask_resized = cv2.resize(mask, (W, H),
                                          interpolation=cv2.INTER_NEAREST)
                mask_bin = (mask_resized > 0.5).astype(np.uint8)
                detections.append((mask_bin, label, int(tid), box))

        # ── 3. SuperPoint feature extraction ─────────────────────────────────
        gray_t = (torch.from_numpy(gray).float() / 255.0
                  ).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H,W)
        with torch.no_grad():
            curr_feats = self.extractor.extract(gray_t)

        # ── 4. LightGlue matching (current vs previous frame) ─────────────────
        # matched_2d_curr/last: Nx2 pixel coords in current/previous frame
        matched_2d_curr = np.zeros((0, 2), dtype=np.float32)
        matched_2d_last = np.zeros((0, 2), dtype=np.float32)

        if self.prev_feats is not None:
            with torch.no_grad():
                match_out = self.matcher({'image0': self.prev_feats,
                                          'image1': curr_feats})
            pf = rbd(self.prev_feats)
            cf = rbd(curr_feats)
            mr = rbd(match_out)
            m  = mr['matches']  # (M, 2)
            if m.shape[0] > 0:
                matched_2d_last = pf['keypoints'][m[:, 0]].cpu().numpy()
                matched_2d_curr = cf['keypoints'][m[:, 1]].cpu().numpy()

        # ── 5. Lift matched 2D keypoints to 3D world frame ────────────────────
        # This is the critical step that the mapping node requires.
        # kpts_curr[n] and kpts_last[n] must be the SAME physical point
        # expressed in world coordinates at their respective frames.
        matched_3d_curr = []
        matched_3d_last = []

        for kc, kl in zip(matched_2d_curr, matched_2d_last):
            uc, vc = int(round(kc[0])), int(round(kc[1]))
            ul, vl = int(round(kl[0])), int(round(kl[1]))

            # Bounds check
            if not (0 <= vc < depth.shape[0] and 0 <= uc < depth.shape[1]):
                continue
            if self.prev_depth is None:
                continue
            if not (0 <= vl < self.prev_depth.shape[0] and
                    0 <= ul < self.prev_depth.shape[1]):
                continue

            depth_c = float(depth[vc, uc])
            depth_l = float(self.prev_depth[vl, ul])

            # Unproject to 3D world frame using respective camera poses
            pt3d_curr = unproject_pixel(uc, vc, depth_c, cam_to_world)
            pt3d_last = unproject_pixel(ul, vl, depth_l, self.prev_cam_to_world)

            if pt3d_curr is None or pt3d_last is None:
                continue

            matched_3d_curr.append(pt3d_curr)
            matched_3d_last.append(pt3d_last)

        # ── 6. Assemble MaskGroup message ─────────────────────────────────────
        group_msg         = MaskGroup()
        group_msg.header  = rgb_msg.header

        for mask_bin, label, track_id, box in detections:

            # For each instance: find 3D matched kpts whose CURRENT pixel
            # falls inside this instance's mask
            inst_kpts_curr_3d = []
            inst_kpts_last_3d = []

            for k3c, k3l, kc in zip(matched_3d_curr,
                                      matched_3d_last,
                                      matched_2d_curr):
                uc = int(round(kc[0]))
                vc = int(round(kc[1]))
                if (0 <= vc < mask_bin.shape[0] and
                        0 <= uc < mask_bin.shape[1] and
                        mask_bin[vc, uc] > 0):
                    inst_kpts_curr_3d.append(k3c)
                    inst_kpts_last_3d.append(k3l)

            # Build MaskKpts
            obj          = MaskKpts()
            obj.label    = label
            obj.track_id = track_id % 65535  # uint16 clamp

            # Mask: binary uint8 (255 = object, 0 = background)
            obj.mask        = self.bridge.cv2_to_imgmsg(
                mask_bin * 255, encoding='mono8')
            obj.mask.header = rgb_msg.header

            # Bounding box (pixel coords, z unused)
            obj.bbox_tl = Keypoint(x=float(box[0]), y=float(box[1]), z=0.0)
            obj.bbox_br = Keypoint(x=float(box[2]), y=float(box[3]), z=0.0)

            # 3D keypoints in WORLD frame — what the mapping node actually needs
            obj.kpts_curr = [
                Keypoint(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                for p in inst_kpts_curr_3d
            ]
            obj.kpts_last = [
                Keypoint(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                for p in inst_kpts_last_3d
            ]

            group_msg.objects.append(obj)

        # ── 7. Build static background mask ───────────────────────────────────
        # For the "static" entry: one MaskKpts with label="static",
        # track_id=65535, mask = label-ID image (pixel value = label id).
        # For now we use a simple "everything not detected as dynamic" mask.
        # Replace with SegFormer output for proper semantic labelling.
        static_mask = np.zeros(depth.shape[:2], dtype=np.uint8)
        # Mark all non-detected pixels as static background (label id = 3
        # for static obstacle in your 5-class scheme — adjust as needed)
        static_mask[:] = 3
        for mask_bin, label, _, _ in detections:
            if label == 'dynamic':
                static_mask[mask_bin > 0] = 0  # clear dynamic pixels

        static_obj          = MaskKpts()
        static_obj.label    = "static"
        static_obj.track_id = 65535
        static_obj.mask     = self.bridge.cv2_to_imgmsg(
            static_mask, encoding='mono8')
        static_obj.mask.header = rgb_msg.header
        # bbox and kpts not required for static label (from custom_files.md)
        static_obj.bbox_tl  = Keypoint(x=0.0, y=0.0, z=0.0)
        static_obj.bbox_br  = Keypoint(x=0.0, y=0.0, z=0.0)
        group_msg.objects.append(static_obj)

        self.pub.publish(group_msg)

        self.get_logger().debug(
            f'MaskGroup: {len(group_msg.objects)-1} dynamic + 1 static | '
            f'3D kpt pairs this frame: {len(matched_3d_curr)}',
            throttle_duration_sec=1.0)

        # ── 8. Save state for next frame ──────────────────────────────────────
        self.prev_feats        = curr_feats
        self.prev_depth        = depth.copy()
        self.prev_cam_to_world = cam_to_world.copy()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = SuperpointFrontend()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
