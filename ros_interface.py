# ros_interface.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import math
import cv2
import threading


class ROSInterface(Node):
    def __init__(self):
        super().__init__('rl_ros_interface')

        self.laser_ranges = np.ones(360) * 10.0
        self.position = (0.0, 0.0)
        self.yaw = 0.0
        self.rgb_image = np.zeros((128, 128, 3), dtype=np.uint8)

        self.bridge = CvBridge()

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.image_sub = self.create_subscription(Image, '/rgb', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # :흰색_확인_표시: Spin thread 시작
        self._start_spin_thread()

    def _start_spin_thread(self):
        thread = threading.Thread(target=self._spin_forever, daemon=True)
        thread.start()

    def _spin_forever(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

    def scan_callback(self, msg: LaserScan):
        self.laser_ranges = np.array(msg.ranges)

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.position = (pos.x, pos.y)
        self.yaw = self.quaternion_to_yaw(ori.x, ori.y, ori.z, ori.w)

    def image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            resized = cv2.resize(cv_img, (128, 128))
            self.rgb_image = resized
        except:
            pass

    def send_velocity(self, linear_x: float, angular_z: float):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def get_laser_scan(self, num_points=24):
        step = len(self.laser_ranges) // num_points
        return self.laser_ranges[::step][:num_points]
    
    def get_position_and_yaw(self):
        return self.position, self.yaw
    
    def get_rgb_image(self):
        return self.rgb_image