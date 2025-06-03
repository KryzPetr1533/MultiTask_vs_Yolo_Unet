import os
import numpy as np
from PIL import Image
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D
from std_msgs.msg import Header
from nuimages.nuimages import NuImages

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('image_retriever')
        self.get_logger().info('Starting ImageRetriever…')


        self.declare_parameter('dataroot', '/var/tmp/MultiTask_vs_Yolo_Unet/full_nuImages')
        self.declare_parameter('version', 'v1.0-test')

        # Read parameters
        dataroot = self.get_parameter('dataroot').value
        version  = self.get_parameter('version').value

        self.nuim = NuImages(dataroot=dataroot, version=version, verbose=False)


        self.anns_by_token = {}
        for ann in self.nuim.object_ann:
            token = ann['sample_data_token']
            self.anns_by_token.setdefault(token, []).append(ann)

        self.sd_tokens = [
            sd['token']
            for sd in self.nuim.sample_data
            if sd['is_key_frame'] and
               self.nuim.shortcut('sample_data','sensor', sd['token'])['channel']
                   in ('CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT')
        ]
        if not self.sd_tokens:
            self.get_logger().error('No key‐frame camera data found!')
            rclpy.shutdown()
            return


        self.img_pub        = self.create_publisher(ROSImage,         'nuimage/image',      1)
        self.mask_pub       = self.create_publisher(ROSImage,         'nuimage/mask',       1)
        self.detections_pub = self.create_publisher(Detection2DArray, 'nuimage/detections', 1)


        self.publish_sample(0)

    def publish_sample(self, idx):
        token = self.sd_tokens[idx]
        sd    = self.nuim.get('sample_data', token)
        img_path = os.path.join(self.nuim.dataroot, sd['filename'])
        pil_img  = Image.open(img_path).convert('RGB')
        np_img   = np.array(pil_img)
        img_msg  = self.numpy_to_ros_image(np_img, encoding='rgb8')


        sem_mask, _ = self.nuim.get_segmentation(token)
        mask_msg    = self.numpy_to_ros_image(sem_mask.astype(np.uint8), encoding='mono8')


        det_array = Detection2DArray()
        det_array.header = Header(stamp=self.get_clock().now().to_msg(),
                                  frame_id='camera')
        for ann in self.anns_by_token.get(token, []):
            xmin, ymin, xmax, ymax = ann['bbox']
            w = sd['width']; h = sd['height']
            bb = BoundingBox2D()
            bb.center.x = (xmin + xmax)/2.0
            bb.center.y = (ymin + ymax)/2.0
            bb.size_x   = (xmax - xmin)*1.0
            bb.size_y   = (ymax - ymin)*1.0
            det = Detection2D()
            det.bbox = bb
            det_array.detections.append(det)

        self.img_pub.publish(img_msg)
        self.mask_pub.publish(mask_msg)
        self.detections_pub.publish(det_array)
        self.get_logger().info(f'Published sample {token}')

    @staticmethod
    def numpy_to_ros_image(np_array, encoding='rgb8'):
        msg = ROSImage()
        msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        msg.height = np_array.shape[0]
        msg.width  = np_array.shape[1]
        msg.encoding     = encoding
        msg.is_bigendian = 0
        msg.step = np_array.strides[0]
        msg.data = np_array.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
