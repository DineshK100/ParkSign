"""Small helpers shared by the SignPark ROS nodes."""
import math
import os

from signpark_gazebo.lot import default_scenario_dir, load_scenario


def declare_scenario(node):
    node.declare_parameter('seed', 0)
    node.declare_parameter('scenario_dir', '')
    d = node.get_parameter('scenario_dir').value or default_scenario_dir(node.get_parameter('seed').value)
    path = os.path.join(d, 'scenario.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found - start the simulation first (it generates the scenario)')
    return load_scenario(path), d


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
