#!/home/user/miniconda3/envs/yolov12/bin/python

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import message_filters
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
import numpy as np
import rclpy
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
        self.declare_parameter('object_point_topic', '/object_point')
        self.declare_parameter('sync_slop_sec', 0.10)
        self.declare_parameter('sync_queue_size', 20)
        self.declare_parameter('bbox_scale_ratio', 1.05)
        self.declare_parameter('bbox_inner_ratio', 0.8)
        self.declare_parameter('min_bbox_edge_px', 8.0)
        self.declare_parameter('roi_radius_px', 80.0)
        self.declare_parameter('cluster_tolerance_m', 0.6)
        self.declare_parameter('cluster_min_size', 5)
        self.declare_parameter('cluster_max_size', 20000)
        self.declare_parameter('ground_threshold_m', 0.12)
        self.declare_parameter('ground_ransac_iters', 30)
        self.declare_parameter('tracker_match_distance_m', 1.5)
        self.declare_parameter('tracker_max_miss', 5)
        self.declare_parameter('debug_log_every_n_frames', 10)
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
        self.object_point_topic = self.get_parameter('object_point_topic').value
        self.sync_slop_sec = float(self.get_parameter('sync_slop_sec').value)
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value)
        self.bbox_scale_ratio = float(self.get_parameter('bbox_scale_ratio').value)
        self.bbox_inner_ratio = float(self.get_parameter('bbox_inner_ratio').value)
        self.min_bbox_edge_px = float(self.get_parameter('min_bbox_edge_px').value)
        self.roi_radius_px = float(self.get_parameter('roi_radius_px').value)
        self.cluster_tolerance = float(self.get_parameter('cluster_tolerance_m').value)
        self.cluster_min_size = int(self.get_parameter('cluster_min_size').value)
        self.cluster_max_size = int(self.get_parameter('cluster_max_size').value)
        self.ground_threshold = float(self.get_parameter('ground_threshold_m').value)
        self.ground_ransac_iters = int(self.get_parameter('ground_ransac_iters').value)
        self.tracker_match_distance = float(self.get_parameter('tracker_match_distance_m').value)
        self.tracker_max_miss = int(self.get_parameter('tracker_max_miss').value)
        self.debug_log_every_n_frames = max(1, int(self.get_parameter('debug_log_every_n_frames').value))

        self.camera_matrix = np.array(self.get_parameter('camera_matrix').value, dtype=np.float64).reshape(3, 3)
        self.lidar_to_camera = np.array(
            self.get_parameter('lidar_to_camera_matrix').value,
            dtype=np.float64,
        ).reshape(4, 4)

        self.bridge = CvBridge()
        self.tracks: Dict[int, TrackState] = {}
        self.next_track_id = 0
        self.frame_counter = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub_lidar = message_filters.Subscriber(self, PointCloud2, self.lidar_topic, qos_profile=qos)
        self.sub_camera = message_filters.Subscriber(self, Image, self.camera_topic, qos_profile=qos)
        self.sub_yolo = message_filters.Subscriber(self, YoloObjects, self.yolo_topic, qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_lidar, self.sub_camera, self.sub_yolo],
            self.sync_queue_size,
            self.sync_slop_sec,
        )
        self.sync.registerCallback(self.synced_callback)

        self.detected_pub = self.create_publisher(DetectedArray, self.detected_topic, 10)
        self.roi_cloud_pub = self.create_publisher(PointCloud2, self.roi_cloud_topic, 10)
        self.object_point_pub = self.create_publisher(PointCloud2, self.object_point_topic, 10)

        self.get_logger().info(f'lidar topic: {self.lidar_topic}')
        self.get_logger().info(f'camera topic: {self.camera_topic}')
        self.get_logger().info(f'yolo topic: {self.yolo_topic}')
        self.get_logger().info(f'roi cloud topic: {self.roi_cloud_topic}')
        self.get_logger().info(f'object point topic: {self.object_point_topic}')
        self.get_logger().info(f'approx sync queue: {self.sync_queue_size}, slop: {self.sync_slop_sec:.3f}s')
        self.get_logger().info('Using provided LiDAR-to-camera extrinsic matrix and camera intrinsics.')

    def synced_callback(
        self,
        lidar_msg: PointCloud2,
        image_msg: Image,
        yolo_msg: YoloObjects,
    ) -> None:
        self.frame_counter += 1
        image = self._image_msg_to_cv2(image_msg)
        if image is None:
            return

        points_xyz = self._pointcloud2_to_xyz(lidar_msg)
        if points_xyz.size == 0:
            return

        roi_cloud_xyz = np.empty((0, 3), dtype=np.float64)
        fused_objects: List[DetectedObject] = []
        debug_stats = None
        fused_objects, roi_cloud_xyz, debug_stats = self._fuse(
            points_xyz,
            image,
            yolo_msg,
            lidar_msg.header.frame_id,
        )

        output = DetectedArray()
        output.header = lidar_msg.header
        output.objects = fused_objects
        self.detected_pub.publish(output)
        self._publish_roi_cloud(roi_cloud_xyz, lidar_msg)
        self._publish_object_points(fused_objects, lidar_msg)
        self._log_debug_stats(points_xyz.shape[0], image.shape, fused_objects, roi_cloud_xyz, debug_stats)

    def _image_msg_to_cv2(self, image_msg: Image) -> Optional[np.ndarray]:
        try:
            return self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'image conversion failed: {exc}')
            return None

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

    def _inner_bbox(self, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        half_w = 0.5 * (x2 - x1) * self.bbox_inner_ratio
        half_h = 0.5 * (y2 - y1) * self.bbox_inner_ratio
        ix1 = max(0, int(round(cx - half_w)))
        iy1 = max(0, int(round(cy - half_h)))
        ix2 = min(width - 1, int(round(cx + half_w)))
        iy2 = min(height - 1, int(round(cy + half_h)))
        return ix1, iy1, ix2, iy2

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
        image: np.ndarray,
        yolo_msg: YoloObjects,
        frame_id: str,
    ) -> Tuple[List[DetectedObject], np.ndarray, Dict[str, int]]:
        height, width = image.shape[:2]
        pixels, overlay_mask = self._compute_overlay_mask(points_xyz, image)

        candidates: List[np.ndarray] = []
        roi_cloud_parts: List[np.ndarray] = []
        stats = {
            'projected_in_image': int(np.count_nonzero(overlay_mask)),
            'bbox_count': len(yolo_msg.yolo_objects),
            'bbox_matched_points': 0,
            'center_roi_points': 0,
            'non_ground_points': 0,
            'cluster_points': 0,
            'valid_boxes': 0,
        }

        for bbox in yolo_msg.yolo_objects:
            ex1, ey1, ex2, ey2 = self._expand_bbox(bbox.x1, bbox.y1, bbox.x2, bbox.y2, width, height)
            if (ex2 - ex1) < self.min_bbox_edge_px or (ey2 - ey1) < self.min_bbox_edge_px:
                continue
            ix1, iy1, ix2, iy2 = self._inner_bbox(ex1, ey1, ex2, ey2, width, height)

            bbox_mask = (
                overlay_mask
                & (pixels[:, 0] >= ix1)
                & (pixels[:, 0] <= ix2)
                & (pixels[:, 1] >= iy1)
                & (pixels[:, 1] <= iy2)
            )
            matched_pixels = pixels[bbox_mask]
            matched_points = points_xyz[bbox_mask]
            stats['bbox_matched_points'] += int(matched_points.shape[0])
            if matched_points.shape[0] < self.cluster_min_size:
                continue

            center_u = 0.5 * (bbox.x1 + bbox.x2)
            center_v = 0.5 * (bbox.y1 + bbox.y2)
            dists = np.linalg.norm(matched_pixels - np.array([center_u, center_v]), axis=1)
            roi_points = matched_points[dists <= self.roi_radius_px]
            stats['center_roi_points'] += int(roi_points.shape[0])
            if roi_points.shape[0] < self.cluster_min_size:
                continue

            non_ground_points = self._remove_ground_ransac(roi_points)
            stats['non_ground_points'] += int(non_ground_points.shape[0])
            if non_ground_points.shape[0] < self.cluster_min_size:
                continue

            roi_cloud_parts.append(non_ground_points)

            clusters = self._euclidean_clusters(non_ground_points)
            if not clusters:
                continue

            largest = max(clusters, key=len)
            largest_cluster = non_ground_points[largest]
            stats['cluster_points'] += int(largest_cluster.shape[0])
            centroid = np.mean(largest_cluster, axis=0)
            candidates.append(centroid)
            stats['valid_boxes'] += 1

        assigned_ids = self._update_tracks(candidates)

        fused_objects: List[DetectedObject] = []
        for track_id, centroid in zip(assigned_ids, candidates):
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

        return fused_objects, roi_cloud_xyz, stats

    def _remove_ground_ransac(self, points_xyz: np.ndarray) -> np.ndarray:
        if points_xyz.shape[0] < 10:
            return points_xyz

        rng = np.random.default_rng()
        best_inliers = -1
        best_plane = None

        for _ in range(self.ground_ransac_iters):
            sample_idx = rng.choice(points_xyz.shape[0], size=3, replace=False)
            p1, p2, p3 = points_xyz[sample_idx]

            den = p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
            if abs(den) < 1e-6:
                continue

            a = (p1[2] * (p2[1] - p3[1]) + p2[2] * (p3[1] - p1[1]) + p3[2] * (p1[1] - p2[1])) / den
            b = (p1[0] * (p2[2] - p3[2]) + p2[0] * (p3[2] - p1[2]) + p3[0] * (p1[2] - p2[2])) / den
            c = p1[2] - a * p1[0] - b * p1[1]

            residuals = np.abs(points_xyz[:, 2] - (a * points_xyz[:, 0] + b * points_xyz[:, 1] + c))
            inliers = int(np.count_nonzero(residuals < self.ground_threshold))
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = (a, b, c)

        if best_plane is None:
            return points_xyz

        a, b, c = best_plane
        residuals = np.abs(points_xyz[:, 2] - (a * points_xyz[:, 0] + b * points_xyz[:, 1] + c))
        return points_xyz[residuals > self.ground_threshold]

    def _euclidean_clusters(self, points_xyz: np.ndarray) -> List[np.ndarray]:
        num_points = points_xyz.shape[0]
        if num_points < self.cluster_min_size:
            return []

        visited = np.zeros(num_points, dtype=bool)
        clusters: List[np.ndarray] = []

        for start_idx in range(num_points):
            if visited[start_idx]:
                continue

            queue = [start_idx]
            visited[start_idx] = True
            cluster = []

            while queue:
                idx = queue.pop()
                cluster.append(idx)

                deltas = points_xyz - points_xyz[idx]
                neighbors = np.flatnonzero(np.linalg.norm(deltas, axis=1) <= self.cluster_tolerance)
                for neighbor_idx in neighbors:
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        queue.append(int(neighbor_idx))

            if self.cluster_min_size <= len(cluster) <= self.cluster_max_size:
                clusters.append(np.array(cluster, dtype=np.int32))

        return clusters

    def _publish_roi_cloud(self, roi_cloud_xyz: np.ndarray, lidar_msg: PointCloud2) -> None:
        roi_points = roi_cloud_xyz.astype(np.float32, copy=False).tolist()
        roi_msg = point_cloud2.create_cloud_xyz32(lidar_msg.header, roi_points)
        self.roi_cloud_pub.publish(roi_msg)

    def _publish_object_points(self, fused_objects: List[DetectedObject], lidar_msg: PointCloud2) -> None:
        object_points = [
            [
                float(obj.world_point.position.x),
                float(obj.world_point.position.y),
                float(obj.world_point.position.z),
            ]
            for obj in fused_objects
        ]
        object_msg = point_cloud2.create_cloud_xyz32(lidar_msg.header, object_points)
        self.object_point_pub.publish(object_msg)

    def _log_debug_stats(
        self,
        lidar_point_count: int,
        image_shape: Tuple[int, int, int],
        fused_objects: List[DetectedObject],
        roi_cloud_xyz: np.ndarray,
        debug_stats: Optional[Dict[str, int]],
    ) -> None:
        if self.frame_counter % self.debug_log_every_n_frames != 0:
            return

        if debug_stats is None:
            self.get_logger().info(
                f'frame={self.frame_counter} lidar_points={lidar_point_count} image={image_shape[1]}x{image_shape[0]} '
                'sync=no_match'
            )
            return

        self.get_logger().info(
            f'frame={self.frame_counter} lidar_points={lidar_point_count} '
            f'projected_in_image={debug_stats["projected_in_image"]} '
            f'bboxes={debug_stats["bbox_count"]} '
            f'bbox_matched={debug_stats["bbox_matched_points"]} '
            f'center_roi={debug_stats["center_roi_points"]} '
            f'non_ground={debug_stats["non_ground_points"]} '
            f'largest_cluster={debug_stats["cluster_points"]} '
            f'valid_boxes={debug_stats["valid_boxes"]} '
            f'roi_pub_points={roi_cloud_xyz.shape[0]} '
            f'detected_objects={len(fused_objects)}'
        )

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
