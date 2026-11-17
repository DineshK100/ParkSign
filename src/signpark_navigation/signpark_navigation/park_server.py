"""Park action server (/signpark/park).

Goal: spot_id + method ("planner" | "rl") + ignore_sign (sign-blind baseline for E3).
  planner: Nav2 Smac Hybrid-A* (Reeds-Shepp) -> segment-wise pure pursuit -> final alignment
  rl:      hands the goal to the RL policy node (/signpark/rl_command) and supervises it
Both methods are measured the same way (odometry + exact collision check against the lot
geometry) and every trial is appended to results/park_trials.csv.
"""
import csv
import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point32, PolygonStamped, PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from signpark_description.car import load_car
from signpark_gazebo.collision import ObstacleSet, car_polygon
from signpark_gazebo.lot import default_scenario_dir, load_scenario
from signpark_gazebo.world_gen import obstacles
from signpark_msgs.action import Park
from signpark_msgs.msg import LotState
from signpark_navigation.maneuver import (ParkGoal, complies, default_class, heading_error_deg, park_goal,
                                         wrap)
from signpark_navigation.tracker import SegmentFollower, TrackerConfig, split_segments, to_twist


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class ParkServer(Node):
    def __init__(self):
        super().__init__('park_server')
        p = self.declare_parameter
        seed = p('seed', 0).value
        sdir = p('scenario_dir', '').value or default_scenario_dir(seed)
        self.lot = load_scenario(os.path.join(sdir, 'scenario.yaml'))
        self.pos_tol = p('position_tolerance', 0.25).value
        self.yaw_tol = p('heading_tolerance_deg', 5.0).value
        self.rl_timeout = p('rl_timeout', 60.0).value
        self.approach = p('straight_approach', 2.5).value   # m, 0 disables the staging pose
        self.max_replans = p('max_replans', 4).value         # replan at each cusp
        self.reverse_gain = p('reverse_steer_gain', 1.35).value
        self.csv_path = os.path.expanduser(p('results_csv', 'results/park_trials.csv').value)
        self.car = load_car()
        self.obs = ObstacleSet(obstacles(self.lot))
        self.cfg = TrackerConfig()
        self.cb = ReentrantCallbackGroup()

        self.odom = None
        self.lot_state = {}
        self.create_subscription(Odometry, '/odom', self.on_odom, 20, callback_group=self.cb)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(LotState, '/signpark/lot_state', self.on_lot, latched, callback_group=self.cb)
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/signpark/planned_path', latched)
        self.fp_pub = self.create_publisher(PolygonStamped, '/signpark/footprint', 10)
        self.rl_pub = self.create_publisher(String, '/signpark/rl_command', 10)
        self.say_pub = self.create_publisher(String, '/signpark/say', 10)
        self.planner = ActionClient(self, ComputePathToPose, 'compute_path_to_pose', callback_group=self.cb)
        self.busy = threading.Lock()
        ActionServer(self, Park, '/signpark/park', self.execute, callback_group=self.cb,
                     goal_callback=self.on_goal, cancel_callback=lambda g: CancelResponse.ACCEPT)
        self.create_timer(0.1, self.publish_footprint, callback_group=self.cb)
        self.get_logger().info('park server ready (/signpark/park)')

    # ---------------------------------------------------------------- inputs
    def on_odom(self, m):
        p = m.pose.pose
        self.odom = (p.position.x, p.position.y, yaw_of(p.orientation), m.twist.twist.linear.x)

    def on_lot(self, m):
        self.lot_state = {s.id: s for s in m.spots}

    def on_goal(self, goal):
        return GoalResponse.REJECT if self.busy.locked() else GoalResponse.ACCEPT

    def publish_footprint(self):
        if self.odom is None:
            return
        msg = PolygonStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in car_polygon(self.car, *self.odom[:3]):
            msg.polygon.points.append(Point32(x=float(x), y=float(y)))
        self.fp_pub.publish(msg)

    # ---------------------------------------------------------------- helpers
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def drive(self, v, steer):
        if v < 0:  # measured: in Gazebo the car reaches only ~70% of the commanded steer when reversing
            steer = max(-self.car.max_steer, min(self.car.max_steer, steer * self.reverse_gain))
        lin, ang = to_twist(v, steer, self.car.wheelbase)
        t = Twist()
        t.linear.x, t.angular.z = float(lin), float(ang)
        self.cmd.publish(t)

    def stop(self, hold=0.5):
        t_end = time.time() + hold
        while time.time() < t_end:
            self.drive(0.0, 0.0)
            time.sleep(0.05)

    def collided(self):
        return self.obs.collides(car_polygon(self.car, *self.odom[:3]))

    def plan(self, goal):
        if not self.planner.wait_for_server(timeout_sec=5.0):
            return None, 'Nav2 planner (compute_path_to_pose) not available - is navigation.launch.py running?'
        g = ComputePathToPose.Goal()
        g.goal.header.frame_id = 'map'
        g.goal.pose.position.x, g.goal.pose.position.y = goal.x, goal.y
        q = quat(goal.yaw)
        g.goal.pose.orientation.x, g.goal.pose.orientation.y, g.goal.pose.orientation.z, g.goal.pose.orientation.w = q
        g.planner_id = 'GridBased'
        g.use_start = False
        fut = self.planner.send_goal_async(g)
        while not fut.done():
            time.sleep(0.02)
        gh = fut.result()
        if not gh.accepted:
            return None, 'planner rejected the goal'
        rf = gh.get_result_async()
        while not rf.done():
            time.sleep(0.02)
        res = rf.result()
        if res.status != GoalStatus.STATUS_SUCCEEDED or len(res.result.path.poses) < 2:
            return None, f'no path found (status {res.status})'
        pts = np.array([(ps.pose.position.x, ps.pose.position.y, yaw_of(ps.pose.orientation))
                        for ps in res.result.path.poses])
        self.publish_path(pts)
        return pts, ''

    def publish_path(self, pts):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y, yaw in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            _, _, ps.pose.orientation.z, ps.pose.orientation.w = quat(yaw)
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    # ---------------------------------------------------------------- execute
    def execute(self, gh):
        with self.busy:
            return self._execute(gh)

    def _execute(self, gh):
        req = gh.request
        res = Park.Result()
        t0 = self.now()

        def fail(msg):
            self.drive(0.0, 0.0)
            res.success, res.message = False, msg
            self.get_logger().warn(f'park {req.spot_id}: {msg}')
            gh.abort()
            return res

        if self.odom is None:
            return fail('no odometry')
        try:
            spot = self.lot.spot(req.spot_id)
        except KeyError:
            return fail(f'unknown spot {req.spot_id}')
        known = self.lot_state.get(req.spot_id)
        cls = known.sign_class if known else 'unknown'
        if req.ignore_sign:
            cls = default_class(spot.geometry)
        if cls == 'unknown':
            return fail('sign of this spot has not been read yet')
        if cls == 'no_parking':
            return fail('spot is marked NO PARKING')
        if spot.occupied:
            return fail('spot is occupied')
        goal = park_goal(spot, cls, self.car)
        self.say_pub.publish(String(data=f'Parking in {spot.id}: the sign says {cls.replace("_", "-")}, so I will {goal.description}.'))
        self.get_logger().info(f'{spot.id}: {cls} -> goal ({goal.x:.2f}, {goal.y:.2f}, {math.degrees(goal.yaw):.0f} deg) via {req.method}')

        stats = dict(dist=0.0, last=self.odom[:2], dir_changes=0, last_dir=0, collision=False)

        def account():
            x, y, _, v = self.odom
            stats['dist'] += math.hypot(x - stats['last'][0], y - stats['last'][1])
            stats['last'] = (x, y)
            d = 1 if v > 0.1 else (-1 if v < -0.1 else 0)
            if d and stats['last_dir'] and d != stats['last_dir']:
                stats['dir_changes'] += 1
            if d:
                stats['last_dir'] = d
            if self.collided():
                stats['collision'] = True

        if req.method == 'rl':
            ok, msg = self.run_rl(gh, spot, goal, account)
        else:
            ok, msg = self.run_planner(gh, goal, account)
        self.stop(0.5)

        x, y, yaw, _ = self.odom
        pos_err = math.hypot(x - goal.x, y - goal.y)
        yaw_err = heading_error_deg(yaw, goal.yaw)
        true_cls = spot.sign_class
        res.success = bool(ok and not stats['collision'] and pos_err <= self.pos_tol and yaw_err <= self.yaw_tol)
        res.message = msg or ('parked' if res.success else 'outside tolerance')
        res.sign_class_used = cls
        res.position_error, res.heading_error = float(pos_err), float(yaw_err)
        res.direction_changes = stats['dir_changes']
        res.path_length = float(stats['dist'])
        res.duration = float(self.now() - t0)
        res.collision = stats['collision']
        compliant = true_cls in ('head_in', 'back_in', 'parallel', 'angled') and complies(yaw, spot, true_cls)
        self.log_csv(req, spot, res, compliant)
        self.get_logger().info(
            f'{spot.id} {req.method}: success={res.success} pos_err={pos_err:.2f}m yaw_err={yaw_err:.1f}deg '
            f'dir_changes={res.direction_changes} time={res.duration:.1f}s collision={res.collision} '
            f'sign_compliant={compliant}')
        if res.success:
            gh.succeed()
        else:
            gh.abort()
        return res

    def run_planner(self, gh, goal, account):
        fb = Park.Feedback()
        fb.phase = 'planning'
        gh.publish_feedback(fb)
        # Perpendicular / angled spots: plan to a staging pose in front of the spot and finish
        # with a straight move in, like a human driver (much more precise than ending on an arc).
        straight, target = None, goal
        if self.approach > 0 and goal.geometry != 'parallel':
            d = -1.0 if goal.sign_class == 'back_in' else 1.0   # +1: drive in forwards, -1: reverse in
            ux, uy = math.cos(goal.yaw), math.sin(goal.yaw)
            target = ParkGoal(goal.x - d * self.approach * ux, goal.y - d * self.approach * uy, goal.yaw,
                              goal.sign_class, goal.description, goal.geometry)
            s = np.linspace(0.0, self.approach, int(self.approach / 0.1) + 1)[1:]
            straight = np.stack([target.x + d * s * ux, target.y + d * s * uy, np.full_like(s, goal.yaw)], axis=1)
        seg_count = 0
        for attempt in range(self.max_replans + 1):
            path, msg = self.plan(target)
            if path is None and straight is not None and attempt == 0:
                self.get_logger().info(f'staging pose unreachable ({msg}); planning straight to the goal')
                straight, target = None, goal
                path, msg = self.plan(target)
            if path is None:
                return False, msg
            full = np.vstack([path, straight]) if straight is not None else path
            self.publish_path(full)
            segs = split_segments(full)
            last = len(segs) == 1 or attempt == self.max_replans
            self.get_logger().info(f'plan {attempt}: {len(full)} poses, {len(segs)} segment(s)')
            # Follow only up to the next cusp, then replan from where the car really is.
            for pts, direction in (segs if last else segs[:1]):
                ok, msg = self.follow(gh, pts, direction, seg_count, goal, account, fb)
                if not ok:
                    return False, msg
                seg_count += 1
            if last:
                break
        return self.align(gh, goal, account), ''

    def follow(self, gh, pts, direction, k, goal, account, fb):
        f = SegmentFollower(pts, direction, self.car, self.cfg)
        t_seg = time.time()
        while True:
            if gh.is_cancel_requested:
                gh.canceled()
                return False, 'canceled'
            x, y, yaw, _ = self.odom
            v, steer, done = f.step(x, y, yaw)
            account()
            if done:
                ex, ey, eyaw = f.end
                self.get_logger().info(
                    f'segment {k} ({"fwd" if direction > 0 else "rev"}) done: pos err '
                    f'{math.hypot(ex - x, ey - y):.2f} m, yaw err {math.degrees(wrap(eyaw - yaw)):.1f} deg')
                break
            if time.time() - t_seg > 60.0:
                return False, f'segment {k} timed out'
            self.drive(v, steer)
            fb.phase, fb.segment = 'driving', k
            fb.distance_to_goal = float(math.hypot(goal.x - x, goal.y - y))
            gh.publish_feedback(fb)
            time.sleep(0.05)
        self.stop(self.cfg.stop_time)
        return True, ''

    def align(self, gh, goal, account):
        """Final alignment: fix the remaining error along the goal heading with a slow straight move."""
        fb = Park.Feedback()
        fb.phase = 'aligning'
        gh.publish_feedback(fb)
        t_end = time.time() + 8.0
        while time.time() < t_end:
            x, y, yaw, _ = self.odom
            along = (goal.x - x) * math.cos(goal.yaw) + (goal.y - y) * math.sin(goal.yaw)
            lateral = -(goal.x - x) * math.sin(goal.yaw) + (goal.y - y) * math.cos(goal.yaw)
            if abs(along) < 0.05:
                break
            steer = max(-0.3, min(0.3, 1.5 * wrap(goal.yaw - yaw) * (1 if along > 0 else -1) + 0.5 * lateral * (1 if along > 0 else -1)))
            self.drive(math.copysign(max(0.15, min(0.4, abs(along))), along), steer)
            account()
            time.sleep(0.05)
        return True

    def run_rl(self, gh, spot, goal, account):
        cmd = {'active': True, 'spot_id': spot.id, 'goal': [goal.x, goal.y, goal.yaw]}
        self.rl_pub.publish(String(data=json.dumps(cmd)))
        fb = Park.Feedback()
        fb.phase = 'driving'
        t0 = time.time()
        settled = None
        ok, msg = False, 'timeout'
        while time.time() - t0 < self.rl_timeout:
            if gh.is_cancel_requested:
                gh.canceled()
                msg = 'canceled'
                break
            account()
            x, y, yaw, v = self.odom
            pe = math.hypot(x - goal.x, y - goal.y)
            ye = heading_error_deg(yaw, goal.yaw)
            fb.distance_to_goal = float(pe)
            gh.publish_feedback(fb)
            if pe <= self.pos_tol and ye <= self.yaw_tol and abs(v) < 0.05:
                settled = settled or time.time()
                if time.time() - settled > 1.0:
                    ok, msg = True, ''
                    break
            else:
                settled = None
            if self.collided():
                msg = 'collision'
                break
            time.sleep(0.05)
        self.rl_pub.publish(String(data=json.dumps({'active': False})))
        return ok, msg

    def log_csv(self, req, spot, res, compliant):
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['time', 'scenario_seed', 'spot', 'geometry', 'true_sign', 'sign_used', 'method',
                            'ignore_sign', 'success', 'message', 'pos_err_m', 'heading_err_deg',
                            'direction_changes', 'path_length_m', 'duration_s', 'collision', 'sign_compliant'])
            w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), self.get_parameter('seed').value, spot.id,
                        spot.geometry, spot.sign_class, res.sign_class_used, req.method, req.ignore_sign,
                        res.success, res.message, f'{res.position_error:.3f}', f'{res.heading_error:.2f}',
                        res.direction_changes, f'{res.path_length:.2f}', f'{res.duration:.1f}',
                        res.collision, compliant])


def main():
    rclpy.init()
    node = ParkServer()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
