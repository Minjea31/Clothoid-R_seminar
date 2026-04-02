#!/home/user/miniconda3/envs/yolov12/bin/python

import os
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO


class PrunedYoloDetectorViewer(Node):
    def __init__(self):
        super().__init__('pruned_yolo_detector_viewer')

        workspace_root = Path(__file__).resolve().parents[4]
        default_yaml_path = str(workspace_root / 'camera_ws/best.yaml')
        default_weights_path = str(workspace_root / 'camera_ws/best.pt')

        self.declare_parameter('image_topic', '/car1/camera/image_raw')
        self.declare_parameter('yaml_path', default_yaml_path)
        self.declare_parameter('weights_path', default_weights_path)
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('window_name', 'Pruned YOLO Detection')

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.yaml_path = self.get_parameter('yaml_path').get_parameter_value().string_value
        self.weights_path = self.get_parameter('weights_path').get_parameter_value().string_value
        self.confidence_threshold = (
            self.get_parameter('confidence_threshold').get_parameter_value().double_value
        )
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value

        self._validate_model_files()

        self.bridge = CvBridge()

        # Allow trusted Ultralytics checkpoints saved before torch 2.6 defaults changed.
        os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
        self.model = YOLO(self.yaml_path, task='detect').load(self.weights_path)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile,
        )

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info(f'Subscribed to: {self.image_topic}')
        self.get_logger().info(f'Loaded pruned YOLO yaml: {self.yaml_path}')
        self.get_logger().info(f'Loaded pruned YOLO weights: {self.weights_path}')
        self.get_logger().info(f'Confidence threshold: {self.confidence_threshold:.2f}')

    def _validate_model_files(self):
        if not Path(self.yaml_path).is_file():
            raise FileNotFoundError(f'YOLO yaml file not found: {self.yaml_path}')

        if not Path(self.weights_path).is_file():
            raise FileNotFoundError(f'YOLO weights file not found: {self.weights_path}')

    def _resolve_label(self, names, cls_id: int) -> str:
        if isinstance(names, dict):
            label = names.get(cls_id, str(cls_id))
        elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            label = names[cls_id]
        else:
            label = str(cls_id)

        if label in {'car', '0'}:
            return 'ERP-42'
        return label

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge conversion failed: {exc}')
            return

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().error(f'YOLO inference failed: {exc}')
            return

        annotated = frame.copy()
        result = results[0]
        names = result.names

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            label = self._resolve_label(names, cls_id)

            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))

            cv2.rectangle(annotated, pt1, pt2, (0, 0, 255), 2)
            text_origin = (pt1[0], max(pt1[1] - 10, 20))
            cv2.putText(
                annotated,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(self.window_name, annotated)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = PrunedYoloDetectorViewer()
    except Exception as exc:
        rclpy.logging.get_logger('pruned_yolo_detector_viewer').error(str(exc))
        rclpy.shutdown()
        raise

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down pruned YOLO detector viewer.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
