#!/home/user/miniconda3/envs/yolov12/bin/python

import sys
import logging
import os
from pathlib import Path
import warnings

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO

from detect_msgs.msg import Objects, YoloObjects


logging.getLogger('ultralytics').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)


class YoloPublisher(Node):
    def __init__(self, check_mode: bool = False) -> None:
        super().__init__('yolo_publisher')

        workspace_root = Path(__file__).resolve().parents[4]
        default_yaml_path = str(workspace_root / 'camera_ws/best.yaml')
        default_weights_path = str(workspace_root / 'camera_ws/best.pt')

        self.declare_parameter('image_topic', '/car1/camera/image_raw')
        self.declare_parameter('yaml_path', default_yaml_path)
        self.declare_parameter('weights_path', default_weights_path)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('target_class_ids', [0])
        self.declare_parameter('publish_topic', '/yolov12_pub')

        self.image_topic = self.get_parameter('image_topic').value
        self.yaml_path = self.get_parameter('yaml_path').value
        self.weights_path = self.get_parameter('weights_path').value
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.target_class_ids = {int(value) for value in self.get_parameter('target_class_ids').value}
        self.publish_topic = self.get_parameter('publish_topic').value
        self.check_mode = check_mode

        self._validate_model_files()

        self.bridge = CvBridge()
        os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
        self.model = YOLO(self.yaml_path, task='detect').load(self.weights_path)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Image, self.image_topic, self.image_callback, qos)

        self.pub = self.create_publisher(YoloObjects, self.publish_topic, 10)

        self.class_names = {
            0: 'ERP-42',
            1: 'drum',
            2: 'robbercone',
        }

        if self.check_mode:
            cv2.namedWindow('YOLO Check', cv2.WINDOW_NORMAL)

        self.get_logger().info(f'Subscribed to: {self.image_topic}')
        self.get_logger().info(f'Publishing detections to: {self.publish_topic}')
        self.get_logger().info(f'Publishing array message type: {YoloObjects.__module__}.{YoloObjects.__name__}')
        if self.check_mode:
            self.get_logger().info('OpenCV check window enabled.')

    def _validate_model_files(self) -> None:
        if not Path(self.yaml_path).is_file():
            raise FileNotFoundError(f'YOLO yaml file not found: {self.yaml_path}')
        if not Path(self.weights_path).is_file():
            raise FileNotFoundError(f'YOLO weights file not found: {self.weights_path}')

    def image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self._run_inference(frame, msg.header)

    def _run_inference(self, frame, header) -> None:
        if frame is None:
            return

        h0, w0 = frame.shape[:2]
        try:
            results = self.model(frame, imgsz=(h0, w0), conf=self.confidence_threshold, verbose=False)[0]
        except Exception as exc:
            self.get_logger().error(f'YOLO inference failed: {exc}')
            return

        out = YoloObjects()
        out.header = header
        debug_frame = frame.copy() if self.check_mode else None

        idx_counter = 0
        for box in results.boxes:
            cls_id = int(box.cls.cpu().item())
            if cls_id not in self.target_class_ids:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())

            obj = Objects()
            obj.id = idx_counter
            obj.class_id = cls_id
            obj.x1 = x1
            obj.x2 = x2
            obj.y1 = y1
            obj.y2 = y2
            out.yolo_objects.append(obj)
            idx_counter += 1

            if self.check_mode:
                label = self.class_names.get(cls_id, f'class_{cls_id}')
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    debug_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        self.pub.publish(out)

        if self.check_mode and debug_frame is not None:
            cv2.imshow('YOLO Check', debug_frame)
            cv2.waitKey(1)

    def destroy_node(self):
        if self.check_mode:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None) -> None:
    argv = list(sys.argv[1:] if args is None else args)
    check_mode = False
    filtered_args = []
    for arg in argv:
        if arg == '--check':
            check_mode = True
            continue
        filtered_args.append(arg)

    rclpy.init(args=filtered_args)
    node = YoloPublisher(check_mode=check_mode)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down yolo_publisher.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
