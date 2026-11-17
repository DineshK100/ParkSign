"""Send one Park goal from the command line.

    ros2 run signpark_navigation park P03                 # Hybrid-A*
    ros2 run signpark_navigation park P03 --method rl     # learned policy
    ros2 run signpark_navigation park P03 --ignore-sign   # sign-blind baseline (E3)
"""
import argparse

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from signpark_msgs.action import Park


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('spot')
    ap.add_argument('--method', default='planner', choices=['planner', 'rl'])
    ap.add_argument('--ignore-sign', action='store_true')
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = Node('park_cli')
    client = ActionClient(node, Park, '/signpark/park')
    if not client.wait_for_server(timeout_sec=10.0):
        print('park server not available')
        return
    goal = Park.Goal(spot_id=a.spot, method=a.method, ignore_sign=a.ignore_sign)
    last = ['']

    def fb(m):
        f = m.feedback
        s = f'{f.phase} segment={f.segment} dist={f.distance_to_goal:.2f}m'
        if s.split(' dist')[0] != last[0]:
            print(s)
            last[0] = s.split(' dist')[0]

    fut = client.send_goal_async(goal, feedback_callback=fb)
    rclpy.spin_until_future_complete(node, fut)
    gh = fut.result()
    if not gh.accepted:
        print('goal rejected (park server busy?)')
        return
    rf = gh.get_result_async()
    rclpy.spin_until_future_complete(node, rf)
    r = rf.result().result
    print(f'success={r.success} ({r.message}) sign={r.sign_class_used} pos_err={r.position_error:.2f}m '
          f'heading_err={r.heading_error:.1f}deg direction_changes={r.direction_changes} '
          f'path={r.path_length:.1f}m time={r.duration:.1f}s collision={r.collision}')
    node.destroy_node()
    rclpy.try_shutdown()
