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


class YoloDetectorViewer(Node):
    def __init__(self):
        super().__init__('yolo_detector_viewer')

        default_model_path = str(Path('/home/user/seminar/camera_ws/raw_best.pt'))

        self.declare_parameter('image_topic', '/car1/camera/image_raw')
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('window_name', 'YOLO Detection')

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = (
            self.get_parameter('confidence_threshold').get_parameter_value().double_value
        )
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value

        self.bridge = CvBridge()

        # PyTorch 2.6 changed torch.load() to weights_only=True by default.
        # Ultralytics checkpoints often need the older behavior to load trusted .pt files.
        os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
        self.model = YOLO(self.model_path)

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
        self.get_logger().info(f'Loaded YOLO model: {self.model_path}')
        self.get_logger().info(f'Confidence threshold: {self.confidence_threshold:.2f}')

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
    node = YoloDetectorViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down YOLO detector viewer.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
