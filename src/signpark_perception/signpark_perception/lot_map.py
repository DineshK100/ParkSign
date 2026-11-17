"""Lot map node: spot geometry + sign classes learned from the sign reader + occupancy.

Publishes /signpark/lot_state (LotState, latched) and /signpark/lot_markers (RViz).
Params: reveal_signs (bool) - start with ground-truth sign classes (skip the scan; for
        testing parking on its own).
Occupancy currently comes from the scenario file (ground truth).
TODO: estimate occupancy from the lidar instead.
"""
import math

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from signpark_msgs.msg import LotState, SignReading, SpotInfo
from signpark_perception.common import declare_scenario

COLORS = {
    'unknown': (0.6, 0.6, 0.6), 'head_in': (0.1, 0.8, 0.2), 'back_in': (0.2, 0.4, 1.0),
    'parallel': (0.0, 0.8, 0.8), 'angled': (0.7, 0.3, 0.9), 'no_parking': (0.9, 0.1, 0.1),
}


class LotMap(Node):
    def __init__(self):
        super().__init__('lot_map')
        self.lot, _ = declare_scenario(self)
        reveal = self.declare_parameter('reveal_signs', False).value
        self.info = {}
        for s in self.lot.spots:
            self.info[s.id] = dict(cls=s.sign_class if reveal else 'unknown', text='', conf=1.0 if reveal else 0.0)
        self.highlight = ''
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(LotState, '/signpark/lot_state', latched)
        self.mpub = self.create_publisher(MarkerArray, '/signpark/lot_markers', latched)
        self.create_subscription(SignReading, '/signpark/sign_readings', self.on_reading, 50)
        from std_msgs.msg import String
        self.create_subscription(String, '/signpark/selected_spot', self.on_selected, 10)
        self.create_timer(1.0, self.publish)
        self.publish()

    def on_reading(self, m: SignReading):
        if m.spot_id in self.info and m.sign_class != 'unknown':
            self.info[m.spot_id] = dict(cls=m.sign_class, text=m.text_read, conf=m.confidence)
            self.publish()

    def on_selected(self, m):
        self.highlight = m.data
        self.publish()

    def publish(self):
        st = LotState()
        st.header.stamp = self.get_clock().now().to_msg()
        st.header.frame_id = 'map'
        for s in self.lot.spots:
            i = self.info[s.id]
            sp = SpotInfo()
            sp.id, sp.geometry = s.id, s.geometry
            sp.center.x, sp.center.y, sp.center.theta = s.cx, s.cy, s.yaw
            sp.width, sp.length = s.width, s.length
            sp.sign_class, sp.sign_text, sp.sign_confidence = i['cls'], i['text'], float(i['conf'])
            sp.occupied = s.occupied
            sp.dist_to_entrance = float(self.lot.dist_to_entrance(s))
            st.spots.append(sp)
        self.pub.publish(st)
        self.mpub.publish(self.markers(st))

    def markers(self, st):
        arr = MarkerArray()
        for k, sp in enumerate(st.spots):
            r, g, b = COLORS.get(sp.sign_class, COLORS['unknown'])
            m = Marker()
            m.header = st.header
            m.ns, m.id, m.type, m.action = 'spots', k, Marker.LINE_STRIP, Marker.ADD
            m.scale.x = 0.25 if sp.id == self.highlight else 0.1
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            c, s_ = math.cos(sp.center.theta), math.sin(sp.center.theta)
            for dx, dy in ((1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1)):
                x, y = dx * sp.length / 2, dy * sp.width / 2
                m.points.append(Point(x=sp.center.x + c * x - s_ * y, y=sp.center.y + s_ * x + c * y, z=0.05))
            arr.markers.append(m)
            # arrow = required heading once parked (from the sign class)
            if sp.sign_class in ('head_in', 'back_in', 'parallel', 'angled'):
                yaw = sp.center.theta + (math.pi if sp.sign_class == 'back_in' else 0.0)
                a = Marker()
                a.header = st.header
                a.ns, a.id, a.type = 'heading', k, Marker.ARROW
                a.points = [Point(x=sp.center.x - 1.2 * math.cos(yaw), y=sp.center.y - 1.2 * math.sin(yaw), z=0.1),
                            Point(x=sp.center.x + 1.2 * math.cos(yaw), y=sp.center.y + 1.2 * math.sin(yaw), z=0.1)]
                a.scale.x, a.scale.y, a.scale.z = 0.15, 0.4, 0.4
                a.color.r, a.color.g, a.color.b, a.color.a = r, g, b, 0.9
                arr.markers.append(a)
            t = Marker()
            t.header = st.header
            t.ns, t.id, t.type = 'labels', k, Marker.TEXT_VIEW_FACING
            t.pose.position.x, t.pose.position.y, t.pose.position.z = sp.center.x, sp.center.y, 1.0
            t.scale.z = 0.6
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f'{sp.id}\n{sp.sign_class}' + (' [occ]' if sp.occupied else '')
            arr.markers.append(t)
        return arr


def main():
    rclpy.init()
    node = LotMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
