"""Sign reader node.

Sign-post positions come from the lot map (the car knows *where* the signs are, not what
they say). While the car drives, each post is projected into the camera image; when the
board is large enough in view (or about to leave the view), the best crop is sent to the
classifier once and the result is published as a SignReading.

Params: backend (claude|ocr|ground_truth), model, min_width_px, good_width_px,
        max_distance, min_facing, max_attempts, save_crops_dir
"""
import os
import threading
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from signpark_description.car import load_car
from signpark_msgs.msg import SignReading
from signpark_perception.classifiers import GroundTruthClassifier, make_classifier
from signpark_perception.common import declare_scenario, yaw_from_quat
from signpark_perception.projection import crop, sign_view


class SignReader(Node):
    def __init__(self):
        super().__init__('sign_reader')
        self.lot, self.scenario_dir = declare_scenario(self)
        p = self.declare_parameter
        self.backend = p('backend', 'claude').value
        model = p('model', 'claude-opus-5-5').value
        self.min_w = p('min_width_px', 60.0).value
        self.good_w = p('good_width_px', 130.0).value
        self.max_dist = p('max_distance', 16.0).value
        self.min_facing = p('min_facing', 0.3).value
        self.max_attempts = p('max_attempts', 3).value
        self.save_dir = p('save_crops_dir', os.path.join(self.scenario_dir, 'crops')).value
        os.makedirs(self.save_dir, exist_ok=True)

        kw = {'model': model} if self.backend == 'claude' else {}
        self.clf = make_classifier(self.backend, **kw)
        car = load_car()
        self.cams = car.cameras
        s = self.lot.sign
        self.board = {'center_z': s['post_height'], 'width': s['board_width'], 'height': s['board_height']}
        self.K = {}
        self.pose = None
        self.best = {}        # spot_id -> (width_px, crop)
        self.done = set()     # spot ids with a final answer
        self.attempts = {}
        self.busy = set()
        self.lock = threading.Lock()

        self.pub = self.create_publisher(SignReading, '/signpark/sign_readings', 50)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        for name in self.cams:
            self.create_subscription(CameraInfo, f'/camera/{name}/camera_info',
                                     lambda m, n=name: self.K.__setitem__(n, list(m.k)), 1)
            self.create_subscription(Image, f'/camera/{name}/image_raw',
                                     lambda m, n=name: self.on_image(m, n), 1)
        self.get_logger().info(f'sign reader up: backend={self.backend}, {len(self.lot.spots)} sign posts, '
                               f'crops -> {self.save_dir}')

    def on_odom(self, m):
        q = m.pose.pose.orientation
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_from_quat(q))

    def on_image(self, m, cam_name):
        if self.pose is None:
            return
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        for spot in self.lot.spots:
            sid = spot.id
            if sid in self.done or sid in self.busy:
                continue
            v = sign_view(spot, self.pose, self.cams[cam_name], self.board, m.width, m.height,
                          self.K.get(cam_name))
            ok = (v is not None and v['distance'] <= self.max_dist and v['facing'] >= self.min_facing
                  and v['width_px'] >= self.min_w)
            key = (sid, cam_name)
            if ok:
                if key not in self.best or v['width_px'] > self.best[key][0]:
                    self.best[key] = (v['width_px'], crop(img, v['bbox']))
                if v['width_px'] >= self.good_w:
                    self.submit(spot, key)
            elif key in self.best:
                self.submit(spot, key)  # sign left this camera's view: classify the best crop we got

    def submit(self, spot, key):
        sid = spot.id
        _, img = self.best.pop(key)
        for k in [k for k in self.best if k[0] == sid]:
            self.best.pop(k)
        self.busy.add(sid)
        self.attempts[sid] = self.attempts.get(sid, 0) + 1
        threading.Thread(target=self.classify, args=(spot, img), daemon=True).start()

    def classify(self, spot, img):
        sid = spot.id
        n = self.attempts[sid]
        PILImage.fromarray(img).save(os.path.join(self.save_dir, f'{sid}_{spot.sign_class}_{n}.png'))
        if isinstance(self.clf, GroundTruthClassifier):
            r = self.clf.classify(img, spot.sign_class)
        else:
            r = self.clf.classify(img)
        if r.sign_class != 'unknown' or n >= self.max_attempts:
            self.done.add(sid)
        self.busy.discard(sid)
        msg = SignReading()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.spot_id, msg.sign_class, msg.text_read = sid, r.sign_class, r.text_read
        msg.confidence, msg.backend, msg.latency_s = float(r.confidence), self.backend, float(r.latency_s)
        msg.ground_truth_class = spot.sign_class
        self.pub.publish(msg)
        mark = 'OK ' if r.sign_class == spot.sign_class else 'ERR'
        self.get_logger().info(f'[{mark}] {sid}: {r.sign_class} (truth {spot.sign_class}) '
                               f'"{r.text_read}" {r.latency_s:.1f}s{"" if r.raw[:5] != "API e" else " " + r.raw}')


def main():
    rclpy.init()
    node = SignReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
