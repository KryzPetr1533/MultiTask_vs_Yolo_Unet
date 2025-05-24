import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector')
        self.get_logger().info('Starting Detector…')

        self.bridge = CvBridge()
        self.sub    = self.create_subscription(
            ROSImage, 'nuimage/image', self.image_callback, 1)
        self.pub    = self.create_publisher(
            Detection2DArray, 'nuimage/detections_pred', 1)

        model_path = os.path.join(
            os.getenv('ROS_WS', '/ros2_ws'),
            'yolo_best.pt'
        )
        self.model = YOLO(model_path)
        self.model.model.conf = 0.25
        self.get_logger().info(f'Loaded YOLOv8 model from {model_path}')

    def image_callback(self, msg: ROSImage):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        frame  = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        results = self.model(frame)[0] 
        preds   = results.boxes.xyxy.cpu().numpy() 

        out = Detection2DArray()
        out.header.stamp    = msg.header.stamp
        out.header.frame_id = msg.header.frame_id

        for x1, y1, x2, y2, conf, cls in preds:
            bb = BoundingBox2D()
            bb.center.x = float((x1 + x2) / 2.0)
            bb.center.y = float((y1 + y2) / 2.0)
            bb.size_x   = float(x2 - x1)
            bb.size_y   = float(y2 - y1)

            det = Detection2D()
            det.bbox = bb

            out.detections.append(det)

        self.pub.publish(out)
        self.get_logger().debug(
            f'Published {len(out.detections)} detections at {msg.header.stamp.sec}.{msg.header.stamp.nanosec}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
