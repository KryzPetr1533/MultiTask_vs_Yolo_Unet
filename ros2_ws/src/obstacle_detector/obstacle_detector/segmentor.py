import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import torch
import numpy as np
from torchvision import transforms

class SegmentorNode(Node):
    def __init__(self):
        super().__init__('segmentor')
        self.get_logger().info('Starting Segmentor…')

        # Declare parameters for model path, device, and threshold
        self.declare_parameter('model_path', '/ros2_ws/unet_best.pt')
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('threshold', 0.5)

        model_path = self.get_parameter('model_path').value
        device     = self.get_parameter('device').value
        threshold  = self.get_parameter('threshold').value

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Subscribers and publishers
        self.sub = self.create_subscription(
            ROSImage,
            'nuimage/image',
            self.image_callback,
            1
        )
        self.pub = self.create_publisher(
            ROSImage,
            'nuimage/mask_pred',
            1
        )

        # Load U-Net model
        self.model = torch.load(model_path, map_location=device)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold

        # Preprocessing transform
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def image_callback(self, msg: ROSImage):
        # Convert ROS Image to CV2 BGR
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        # Preprocess for model input
        input_tensor = self.preprocess(cv_img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            # Handle model outputs that may be tuple or list
            if isinstance(output, (tuple, list)):
                output = output[0]
            # Expect output shape [1, 1, H, W] or [1, H, W]
            if output.dim() == 4:
                mask = output[:, 0, :, :]
            else:
                mask = output
            # Apply sigmoid and threshold
            mask = torch.sigmoid(mask)
            mask = (mask > self.threshold).cpu().numpy().astype(np.uint8) * 255
            mask = mask[0]

        # Convert mask to ROS Image
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = msg.header

        # Publish mask
        self.pub.publish(mask_msg)
        self.get_logger().debug(
            f'Published mask for stamp ' +
            f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec}"
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
