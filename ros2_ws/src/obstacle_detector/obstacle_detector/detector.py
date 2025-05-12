import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D
from cv_bridge import CvBridge
import numpy as np

class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector')
        self.get_logger().info('Starting Detector…')

        # Bridge to convert ROS Image ↔ OpenCV
        self.bridge = CvBridge()

        # Subscribe to the raw image stream
        self.sub = self.create_subscription(
            ROSImage,
            'nuimage/image',
            self.image_callback,
            1)

        # Publish predicted detections
        self.pub = self.create_publisher(
            Detection2DArray,
            'nuimage/detections_pred',
            1)

        # TODO: load your detection model here
        # e.g. self.model = load_my_detector(...)

    def image_callback(self, msg: ROSImage):
        # 1) Convert to OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        # 2) Run your detector (here we fake one box in center)
        h, w, _ = cv_img.shape
        # TODO: replace with inference: boxes, scores, classes = self.model(cv_img)
        bbox = BoundingBox2D()
        bbox.center.x = w / 2.0
        bbox.center.y = h / 2.0
        bbox.size_x   = w / 4.0
        bbox.size_y   = h / 4.0

        det = Detection2D()
        det.bbox = bbox
        # (optional) det.results = [ObjectHypothesisWithPose(...)]

        # 3) Package into Detection2DArray
        out = Detection2DArray()
        # keep the same timestamp and frame as the input — this is crucial
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id
        out.detections = [det]

        # 4) Publish
        self.pub.publish(out)
        self.get_logger().debug(f'Published {len(out.detections)} detections for stamp {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')

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
