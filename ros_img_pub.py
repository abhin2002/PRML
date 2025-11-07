#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
# from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from sensor_msgs.msg import RegionOfInterest


class ImageSubscriber(Node):
    def __init__(self, show_image=True):
        super().__init__('image_subscriber')

        # --- Subscriber for compressed image ---
        self.subscription = self.create_subscription(
            CompressedImage,
            'image_topic',
            self.listener_callback,
            10
        )

        # --- ROI publisher ---
        self.roi_publisher = self.create_publisher(RegionOfInterest, 'roi_topic', 10)


        # --- State ---
        self.bridge = CvBridge()
        self.show_image = show_image
        self.roi = None
        self.roi_selected = False
        self.selecting = False
        self.selection_start = None
        self.selection_end = None
        self.current_frame = None

        # Since compressed image has no initframe, treat first frame as init
        self.first_frame = True

        if self.show_image:
            cv2.namedWindow('received', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('received', self._on_mouse)

        self.get_logger().info("ImageSubscriber initialized. Waiting for compressed frames...")

    # ----------------------------------------------------------
    # Mouse callback for ROI selection
    # ----------------------------------------------------------
    def _on_mouse(self, event, x, y, flags, param):
        if not self.show_image or not self.selecting:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection_start = (x, y)
            self.selection_end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.selection_start is not None:
            self.selection_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.selection_start is not None:
            self.selection_end = (x, y)
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            x_min, x_max = sorted((int(x1), int(x2)))
            y_min, y_max = sorted((int(y1), int(y2)))

            if (x_max - x_min) > 0 and (y_max - y_min) > 0:
                self.roi = (x_min, y_min, x_max, y_max)
                self.roi_selected = True
                self.get_logger().info(
                    f"Selected ROI: (x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})"
                )

                roi_msg = RegionOfInterest()
                roi_msg.x_offset = x_min
                roi_msg.y_offset = y_min
                roi_msg.width = x_max - x_min
                roi_msg.height = y_max - y_min
                roi_msg.do_rectify = False

                self.roi_publisher.publish(roi_msg)
                self.get_logger().info(
                    f"Published ROI on /roi_topic: (x={x_min}, y={y_min}, w={roi_msg.width}, h={roi_msg.height})"
                )


            else:
                self.get_logger().warning("Invalid ROI (zero area). Ignoring selection.")

            # Reset selection mode
            self.selecting = False
            self.selection_start = None
            self.selection_end = None

    # ----------------------------------------------------------
    # Process incoming compressed image
    # ----------------------------------------------------------
    def listener_callback(self, msg: CompressedImage):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to decode compressed image: {e}")
            return

        self.current_frame = cv_image.copy()

        # Treat *first received frame* as initframe
        if self.first_frame:
            self.get_logger().info("First 
frame received — start ROI selection.")
            self.roi = None
            self.roi_selected = False
            self.selecting = True
            self.selection_start = None
            self.selection_end = None
            self.first_frame = False

        # Display frame
        display_frame = self.current_frame.copy()

        # Temporary draw while dragging
        if self.selecting and self.selection_start and self.selection_end:
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Persistent ROI
        if self.roi_selected and self.roi:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if self.show_image:
            cv2.imshow('received', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("User pressed 'q' — closing viewer.")

    # ----------------------------------------------------------
    def destroy_node(self):
        if self.show_image:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        return super().destroy_node()


# ----------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber(show_image=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
