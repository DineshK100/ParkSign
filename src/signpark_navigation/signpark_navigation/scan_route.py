"""Drive once down the aisle so the side cameras can read every sign, then stop.

    ros2 run signpark_navigation scan_route --ros-args -p lane_y:=-0.5 -p end_x:=44.0
"""
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from signpark_description.car import load_car
from signpark_navigation.tracker import SegmentFollower, TrackerConfig, to_twist


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ScanRoute(Node):
    def __init__(self):
        super().__init__('scan_route')
        self.lane_y = self.declare_parameter('lane_y', -0.5).value
        self.end_x = self.declare_parameter('end_x', 44.0).value
        self.speed = self.declare_parameter('speed', 1.5).value
        self.car = load_car()
        self.odom = None
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status = self.create_publisher(String, '/signpark/scan_status', 10)

    def on_odom(self, m):
        p = m.pose.pose
        self.odom = (p.position.x, p.position.y, yaw_of(p.orientation))

    def run(self):
        while self.odom is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.2)
        x0, y0, _ = self.odom
        # straight lane from slightly ahead of the car to end_x (pure pursuit merges onto it)
        xs = np.arange(x0 + 2.0, self.end_x, 0.2)
        path = np.stack([xs, np.full_like(xs, self.lane_y), np.zeros_like(xs)], axis=1)
        f = SegmentFollower(path, 1, self.car, TrackerConfig(v_max=self.speed, lookahead=3.0))
        self.get_logger().info(f'scanning the lot along y={self.lane_y} to x={self.end_x}')
        self.status.publish(String(data='scanning'))
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            v, steer, done = f.step(*self.odom)
            if done:
                break
            lin, ang = to_twist(v, steer, self.car.wheelbase)
            t = Twist()
            t.linear.x, t.angular.z = lin, ang
            self.cmd.publish(t)
        for _ in range(20):
            self.cmd.publish(Twist())
            time.sleep(0.05)
        self.status.publish(String(data='done'))
        self.get_logger().info('scan done')


def main():
    rclpy.init()
    node = ScanRoute()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
