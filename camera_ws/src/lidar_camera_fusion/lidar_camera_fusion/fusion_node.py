#!/home/user/miniconda3/envs/yolov12/bin/python

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

from detect_msgs.msg import DetectedArray, DetectedObject, YoloObjects


@dataclass
class TrackState:
    track_id: int
    position_xy: np.ndarray
    miss_count: int = 0


class LidarCameraFusionNode(Node):
    def __init__(self) -> None:
        super().__init__('lidar_camera_fusion')

        self.declare_parameter('lidar_topic', '/car1/livox/points')
        self.declare_parameter('camera_topic', '/car1/camera/image_raw')
        self.declare_parameter('yolo_topic', '/yolov12_pub')
        self.declare_parameter('detected_topic', '/detected_array')
        self.declare_parameter('roi_cloud_topic', '/yolo_roi')
        self.declare_parameter('sync_slop_sec', 0.10)
        self.declare_parameter('bbox_scale_ratio', 1.05)
        self.declare_parameter('min_bbox_edge_px', 8.0)
        self.declare_parameter('tracker_match_distance_m', 1.5)
        self.declare_parameter('tracker_max_miss', 5)
        self.declare_parameter(
            'camera_matrix',
            [
                1108.5, 0.0, 640.0,
                0.0, 1108.5, 360.0,
                0.0, 0.0, 1.0,
            ],
        )
        self.declare_parameter(
            'lidar_to_camera_matrix',
            [
                # MATLAB rigidtform3d.A
                -0.0017, -1.0000, 0.0039, 0.0046,
                0.0523, -0.0040, -0.9986, -0.0614,
                0.9986, -0.0015, 0.0523, -0.0466,
                0.0, 0.0, 0.0, 1.0,
            ],
        )

        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.yolo_topic = self.get_parameter('yolo_topic').value
        self.detected_topic = self.get_parameter('detected_topic').value
        self.roi_cloud_topic = self.get_parameter('roi_cloud_topic').value
        self.sync_slop = Duration(seconds=float(self.get_parameter('sync_slop_sec').value))
        self.bbox_scale_ratio = float(self.get_parameter('bbox_scale_ratio').value)
        self.min_bbox_edge_px = float(self.get_parameter('min_bbox_edge_px').value)
        self.tracker_match_distance = float(self.get_parameter('tracker_match_distance_m').value)
        self.tracker_max_miss = int(self.get_parameter('tracker_max_miss').value)

        self.camera_matrix = np.array(self.get_parameter('camera_matrix').value, dtype=np.float64).reshape(3, 3)
        self.lidar_to_camera = np.array(
            self.get_parameter('lidar_to_camera_matrix').value,
            dtype=np.float64,
        ).reshape(4, 4)

        self.bridge = CvBridge()
        self.latest_image_msg: Optional[Image] = None
        self.latest_yolo_msg: Optional[YoloObjects] = None
        self.tracks: Dict[int, TrackState] = {}
        self.next_track_id = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(PointCloud2, self.lidar_topic, self.lidar_callback, qos)
        self.create_subscription(Image, self.camera_topic, self.image_callback, qos)
        self.create_subscription(YoloObjects, self.yolo_topic, self.yolo_callback, qos)

        self.detected_pub = self.create_publisher(DetectedArray, self.detected_topic, 10)
        self.roi_cloud_pub = self.create_publisher(PointCloud2, self.roi_cloud_topic, 10)

        self.get_logger().info(f'lidar topic: {self.lidar_topic}')
        self.get_logger().info(f'camera topic: {self.camera_topic}')
        self.get_logger().info(f'yolo topic: {self.yolo_topic}')
        self.get_logger().info(f'roi cloud topic: {self.roi_cloud_topic}')
        self.get_logger().info('Using provided LiDAR-to-camera extrinsic matrix and camera intrinsics.')

    def image_callback(self, msg: Image) -> None:
        self.latest_image_msg = msg

    def yolo_callback(self, msg: YoloObjects) -> None:
        self.latest_yolo_msg = msg

    def lidar_callback(self, lidar_msg: PointCloud2) -> None:
        image_msg, image = self._get_latest_image()
        if image_msg is None or image is None:
            return

        if not self._is_time_close(lidar_msg.header.stamp, image_msg.header.stamp):
            return

        points_xyz = self._pointcloud2_to_xyz(lidar_msg)
        if points_xyz.size == 0:
            return

        roi_cloud_xyz = np.empty((0, 3), dtype=np.float64)
        fused_objects: List[DetectedObject] = []
        if self.latest_yolo_msg is not None and self._is_time_close(lidar_msg.header.stamp, self.latest_yolo_msg.header.stamp):
            fused_objects, roi_cloud_xyz = self._fuse(
                points_xyz,
                self.latest_yolo_msg,
                lidar_msg.header.frame_id,
            )

        output = DetectedArray()
        output.header = lidar_msg.header
        output.objects = fused_objects
        self.detected_pub.publish(output)
        self._publish_roi_cloud(roi_cloud_xyz, lidar_msg)

    def _get_latest_image(self) -> Tuple[Optional[object], Optional[np.ndarray]]:
        try:
            if self.latest_image_msg is None:
                return None, None
            image = self.bridge.imgmsg_to_cv2(self.latest_image_msg, desired_encoding='bgr8')
            return self.latest_image_msg, image
        except Exception as exc:
            self.get_logger().error(f'image conversion failed: {exc}')
            return None, None

    def _is_time_close(self, lhs, rhs) -> bool:
        lhs_ns = int(lhs.sec) * 1_000_000_000 + int(lhs.nanosec)
        rhs_ns = int(rhs.sec) * 1_000_000_000 + int(rhs.nanosec)
        diff = abs(lhs_ns - rhs_ns)
        return diff <= self.sync_slop.nanoseconds

    def _pointcloud2_to_xyz(self, cloud_msg: PointCloud2) -> np.ndarray:
        raw_points = point_cloud2.read_points(
            cloud_msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True,
        )
        points = np.asarray(raw_points)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        if points.dtype.names is not None:
            return np.column_stack(
                [
                    points['x'].astype(np.float64, copy=False),
                    points['y'].astype(np.float64, copy=False),
                    points['z'].astype(np.float64, copy=False),
                ]
            )

        return np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def _project_points(self, points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
        lidar_h = np.hstack([points_xyz, ones])
        camera_h = (self.lidar_to_camera @ lidar_h.T).T
        camera_xyz = camera_h[:, :3]

        positive_depth = camera_xyz[:, 2] > 1e-6
        pixels = np.full((points_xyz.shape[0], 2), np.nan, dtype=np.float64)

        if np.any(positive_depth):
            camera_valid = camera_xyz[positive_depth]
            uvw = (self.camera_matrix @ camera_valid.T).T
            pixels_valid = uvw[:, :2] / uvw[:, 2:3]
            pixels[positive_depth] = pixels_valid

        return pixels, camera_xyz, positive_depth

    def _expand_bbox(self, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        half_w = 0.5 * (x2 - x1) * self.bbox_scale_ratio
        half_h = 0.5 * (y2 - y1) * self.bbox_scale_ratio
        ex1 = max(0, int(round(cx - half_w)))
        ey1 = max(0, int(round(cy - half_h)))
        ex2 = min(width - 1, int(round(cx + half_w)))
        ey2 = min(height - 1, int(round(cy + half_h)))
        return ex1, ey1, ex2, ey2

    def _compute_overlay_mask(self, points_xyz: np.ndarray, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        pixels, _, positive_depth = self._project_points(points_xyz)

        finite_pixels = np.isfinite(pixels).all(axis=1)
        in_image = (
            positive_depth
            & finite_pixels
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] < height)
        )
        return pixels, in_image

    def _fuse(
        self,
        points_xyz: np.ndarray,
        yolo_msg: YoloObjects,
        frame_id: str,
    ) -> Tuple[List[DetectedObject], np.ndarray]:
        image = self.bridge.imgmsg_to_cv2(self.latest_image_msg, desired_encoding='bgr8')
        height, width = image.shape[:2]
        pixels, overlay_mask = self._compute_overlay_mask(points_xyz, image)

        candidates: List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]] = []
        roi_cloud_parts: List[np.ndarray] = []

        for bbox in yolo_msg.yolo_objects:
            ex1, ey1, ex2, ey2 = self._expand_bbox(bbox.x1, bbox.y1, bbox.x2, bbox.y2, width, height)
            if (ex2 - ex1) < self.min_bbox_edge_px or (ey2 - ey1) < self.min_bbox_edge_px:
                continue

            bbox_mask = (
                overlay_mask
                & (pixels[:, 0] >= ex1)
                & (pixels[:, 0] <= ex2)
                & (pixels[:, 1] >= ey1)
                & (pixels[:, 1] <= ey2)
            )
            matched_points = points_xyz[bbox_mask]
            if matched_points.shape[0] == 0:
                continue
            roi_cloud_parts.append(matched_points)

            # Robust 3D point estimate: use the median after rejecting the furthest 20% by range.
            ranges = np.linalg.norm(matched_points, axis=1)
            keep_count = max(1, int(np.ceil(matched_points.shape[0] * 0.8)))
            keep_idx = np.argsort(ranges)[:keep_count]
            filtered = matched_points[keep_idx]
            centroid = np.median(filtered, axis=0)

            candidates.append((centroid, (ex1, ey1, ex2, ey2)))

        assigned_ids = self._update_tracks([item[0] for item in candidates])

        fused_objects: List[DetectedObject] = []
        for track_id, (centroid, rect) in zip(assigned_ids, candidates):
            pose = Pose()
            pose.position.x = float(centroid[0])
            pose.position.y = float(centroid[1])
            pose.position.z = float(centroid[2])
            pose.orientation.w = 1.0

            obj = DetectedObject()
            obj.id = int(track_id)
            obj.world_point = pose
            fused_objects.append(obj)

        if roi_cloud_parts:
            roi_cloud_xyz = np.vstack(roi_cloud_parts)
        else:
            roi_cloud_xyz = np.empty((0, 3), dtype=np.float64)

        return fused_objects, roi_cloud_xyz

    def _publish_roi_cloud(self, roi_cloud_xyz: np.ndarray, lidar_msg: PointCloud2) -> None:
        roi_points = roi_cloud_xyz.astype(np.float32, copy=False).tolist()
        roi_msg = point_cloud2.create_cloud_xyz32(lidar_msg.header, roi_points)
        self.roi_cloud_pub.publish(roi_msg)

    def _update_tracks(self, detections_xyz: List[np.ndarray]) -> List[int]:
        if not detections_xyz:
            for track in self.tracks.values():
                track.miss_count += 1
            self._prune_tracks()
            return []

        unmatched_track_ids = set(self.tracks.keys())
        assigned_ids: List[int] = []

        for det in detections_xyz:
            det_xy = det[:2]
            best_id = None
            best_distance = None

            for track_id in unmatched_track_ids:
                distance = float(np.linalg.norm(self.tracks[track_id].position_xy - det_xy))
                if distance > self.tracker_match_distance:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_id = track_id

            if best_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = TrackState(track_id=track_id, position_xy=det_xy.copy())
                assigned_ids.append(track_id)
                continue

            unmatched_track_ids.remove(best_id)
            self.tracks[best_id].position_xy = det_xy.copy()
            self.tracks[best_id].miss_count = 0
            assigned_ids.append(best_id)

        for track_id in unmatched_track_ids:
            self.tracks[track_id].miss_count += 1

        self._prune_tracks()
        return assigned_ids

    def _prune_tracks(self) -> None:
        expired = [track_id for track_id, track in self.tracks.items() if track.miss_count > self.tracker_max_miss]
        for track_id in expired:
            del self.tracks[track_id]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarCameraFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down lidar_camera_fusion node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
