# ros2_ws/src/obstacle_detector/obstacle_detector/camera_publisher.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.get_logger().info('Starting CameraPublisher…')

        # Declare parameters
        self.declare_parameter('camera_device', 0)          # default to /dev/video0
        self.declare_parameter('publish_topic', '/nuimage/image')
        self.declare_parameter('frame_rate', 30.0)          # frames per second

        camera_device = self.get_parameter('camera_device').get_parameter_value().integer_value
        self.topic_name = self.get_parameter('publish_topic').get_parameter_value().string_value
        frame_rate     = self.get_parameter('frame_rate').get_parameter_value().double_value

        # OpenCV video capture
        self.cap = cv2.VideoCapture(camera_device)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera device {camera_device}')
            rclpy.shutdown()
            return

        # Publisher
        self.pub = self.create_publisher(ROSImage, self.topic_name, 10)
        self.bridge = CvBridge()

        # Timer for periodic frame grabbing
        timer_period = 1.0 / frame_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame from camera')
            return

        # Convert BGR (OpenCV) → ROS Image (bgr8)
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        self.pub.publish(msg)

    def destroy_node(self):
        # Release the camera on shutdown
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
