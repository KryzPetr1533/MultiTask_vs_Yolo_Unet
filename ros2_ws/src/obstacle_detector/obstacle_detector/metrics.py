import threading

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

class MetricsNode(Node):
    def __init__(self):
        super().__init__('metrics')
        self.get_logger().info('Starting MetricsNode…')

        # Subscribe to ground-truth and predicted detections
        self.gt_sub = self.create_subscription(
            Detection2DArray,
            'nuimage/detections',
            self.gt_cb,
            10
        )
        self.pr_sub = self.create_subscription(
            Detection2DArray,
            'nuimage/detections_pred',
            self.pr_cb,
            10
        )

        # Buffers: timestamp tuple -> message
        self.buffer_gt = {}
        self.buffer_pr = {}
        self.lock = threading.Lock()

    def gt_cb(self, msg: Detection2DArray):
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        with self.lock:
            self.buffer_gt[stamp] = msg
            if stamp in self.buffer_pr:
                pr_msg = self.buffer_pr.pop(stamp)
                gt_msg = self.buffer_gt.pop(stamp)
                self.compute_metrics(gt_msg, pr_msg)

    def pr_cb(self, msg: Detection2DArray):
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        with self.lock:
            self.buffer_pr[stamp] = msg
            if stamp in self.buffer_gt:
                gt_msg = self.buffer_gt.pop(stamp)
                pr_msg = self.buffer_pr.pop(stamp)
                self.compute_metrics(gt_msg, pr_msg)

    def compute_metrics(self, gt_msg: Detection2DArray, pr_msg: Detection2DArray):
        """
        Naively compute the mean IoU between predicted and ground-truth bounding boxes.
        """
        def iou(bb1, bb2):
            x1_min = bb1.center.x - bb1.size_x / 2.0
            y1_min = bb1.center.y - bb1.size_y / 2.0
            x1_max = bb1.center.x + bb1.size_x / 2.0
            y1_max = bb1.center.y + bb1.size_y / 2.0

            x2_min = bb2.center.x - bb2.size_x / 2.0
            y2_min = bb2.center.y - bb2.size_y / 2.0
            x2_max = bb2.center.x + bb2.size_x / 2.0
            y2_max = bb2.center.y + bb2.size_y / 2.0

            xi_min = max(x1_min, x2_min)
            yi_min = max(y1_min, y2_min)
            xi_max = min(x1_max, x2_max)
            yi_max = min(y1_max, y2_max)

            inter_w = max(0.0, xi_max - xi_min)
            inter_h = max(0.0, yi_max - yi_min)
            inter_area = inter_w * inter_h

            area1 = bb1.size_x * bb1.size_y
            area2 = bb2.size_x * bb2.size_y
            union = area1 + area2 - inter_area
            return inter_area / union if union > 0 else 0.0

        ious = []
        for pred in pr_msg.detections:
            best = 0.0
            for gt in gt_msg.detections:
                val = iou(pred.bbox, gt.bbox)
                if val > best:
                    best = val
            ious.append(best)
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        self.get_logger().info(
            f"[Metrics] stamp={gt_msg.header.stamp.sec}.{gt_msg.header.stamp.nanosec} "
            f"mean IoU={mean_iou:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MetricsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
