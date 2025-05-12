import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2

class SegmentorNode(Node):
    def __init__(self):
        super().__init__('segmentor')
        self.get_logger().info('Starting Segmentor…')
        self.bridge = CvBridge()
        # Subscribe to raw images
        self.sub = self.create_subscription(
            ROSImage,
            'nuimage/image',
            self.image_callback,
            1,
        )
        # Publish predicted masks
        self.pub = self.create_publisher(
            ROSImage,
            'nuimage/mask_pred',
            1,
        )
        # TODO: load your segmentation model here (e.g. a neural net)
        # self.model = load_segmentation_model(...)

    def image_callback(self, msg: ROSImage):
        # 1) Convert ROS Image to OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        # 2) Run your segmentation model (dummy: grayscale + threshold)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # 3) Convert mask back to ROS Image
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = msg.header

        # 4) Publish
        self.pub.publish(mask_msg)
        self.get_logger().debug(
            f'Published mask for stamp {msg.header.stamp.sec}.{msg.header.stamp.nanosec}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = SegmentorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
