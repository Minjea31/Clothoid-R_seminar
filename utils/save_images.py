#!/usr/bin/env python3

import argparse
import os
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image


DEFAULT_TOPIC_NAME = '/car1/camera/image_raw'
DEFAULT_SAVE_DIR = './extracted_images'
DEFAULT_SAVE_INTERVAL = 0.1


class ImageSaver(Node):
    def __init__(self, topic_name: str, save_dir: str, save_interval: float):
        super().__init__('image_saver')

        self.topic_name = topic_name
        self.save_dir = save_dir
        self.save_interval = save_interval

        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.last_saved_time = 0.0
        self.image_count = 0

        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.image_callback,
            10
        )

        self.get_logger().info(f'Subscribed to: {self.topic_name}')
        self.get_logger().info(f'Saving images to: {self.save_dir}')
        self.get_logger().info(f'Save interval: {self.save_interval} sec')

    def image_callback(self, msg: Image):
        now = time.time()

        if now - self.last_saved_time < self.save_interval:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        filename = os.path.join(self.save_dir, f'frame_{self.image_count:04d}.jpg')

        ok = cv2.imwrite(filename, cv_image)
        if not ok:
            self.get_logger().error(f'Failed to save image: {filename}')
            return

        self.last_saved_time = now
        self.get_logger().info(f'Saved: {filename}')
        self.image_count += 1


def parse_args(args=None):
    cli_args = remove_ros_args(args=args)[1:]

    parser = argparse.ArgumentParser(description='Save ROS image messages as frame images.')
    parser.add_argument(
        '--topic-name',
        default=DEFAULT_TOPIC_NAME,
        help=f'Image topic to subscribe to (default: {DEFAULT_TOPIC_NAME})'
    )
    parser.add_argument(
        '--save-dir',
        default=DEFAULT_SAVE_DIR,
        help=f'Directory to save extracted images (default: {DEFAULT_SAVE_DIR})'
    )
    parser.add_argument(
        '--save-interval',
        type=float,
        default=DEFAULT_SAVE_INTERVAL,
        help=f'Seconds between saved frames (default: {DEFAULT_SAVE_INTERVAL})'
    )
    return parser.parse_args(cli_args)


def main(args=None):
    parsed_args = parse_args(args)
    rclpy.init(args=args)
    node = ImageSaver(
        topic_name=parsed_args.topic_name,
        save_dir=parsed_args.save_dir,
        save_interval=parsed_args.save_interval,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down image saver.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
