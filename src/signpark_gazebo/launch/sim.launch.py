"""Gazebo parking lot + Ackermann car + ROS bridge.

    ros2 launch signpark_gazebo sim.launch.py seed:=3 gui:=true rviz:=true

A new scenario (sign classes, sign photos, parked cars) is generated from `seed`
into ~/.ros/signpark/scenario_<seed>/ each launch.
"""
import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from signpark_description.car import load_car
from signpark_gazebo.world_gen import generate


def _setup(context):
    seed = int(LaunchConfiguration('seed').perform(context))
    gui = LaunchConfiguration('gui').perform(context).lower() == 'true'
    rviz = LaunchConfiguration('rviz').perform(context).lower() == 'true'
    force = LaunchConfiguration('force').perform(context).strip()
    force = dict(kv.split('=') for kv in force.split(',')) if force else None
    out = generate(seed, force=force)
    world = os.path.join(out, 'world.sdf')
    share = get_package_share_directory('signpark_gazebo')
    car = load_car()

    gz_args = ['gz', 'sim', '-r', world] if gui else ['gz', 'sim', '-r', '-s', '--headless-rendering', world]
    actions = [ExecuteProcess(cmd=gz_args, output='screen', additional_env={'GZ_SIM_RESOURCE_PATH': out})]

    actions.append(Node(package='ros_gz_bridge', executable='parameter_bridge', name='gz_bridge',
                        parameters=[{'config_file': os.path.join(share, 'config', 'bridge.yaml'),
                                     'use_sim_time': True}], output='screen'))

    def static_tf(name, parent, child, xyz, rpy):
        return Node(package='tf2_ros', executable='static_transform_publisher', name=name,
                    arguments=['--x', str(xyz[0]), '--y', str(xyz[1]), '--z', str(xyz[2]),
                               '--roll', str(rpy[0]), '--pitch', str(rpy[1]), '--yaw', str(rpy[2]),
                               '--frame-id', parent, '--child-frame-id', child],
                    parameters=[{'use_sim_time': True}])

    # gz OdometryPublisher reports the world pose (ground truth), so map == odom.
    actions += [
        static_tf('map_to_odom', 'map', 'odom', (0, 0, 0), (0, 0, 0)),
        static_tf('base_to_lidar', 'base_link', 'lidar_link', car.lidar['xyz'], (0, 0, 0)),
    ]
    for n, cam in car.cameras.items():
        actions += [
            static_tf(f'base_to_{n}_camera', 'base_link', f'camera_{n}_link', cam['xyz'], (0, 0, cam['yaw'])),
            static_tf(f'{n}_camera_optical', f'camera_{n}_link', f'camera_{n}_optical_frame', (0, 0, 0),
                      (-math.pi / 2, 0, -math.pi / 2)),
        ]
    if rviz:
        actions.append(Node(package='rviz2', executable='rviz2',
                            arguments=['-d', os.path.join(share, 'rviz', 'signpark.rviz')],
                            parameters=[{'use_sim_time': True}]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('gui', default_value='false', description='Gazebo GUI (false = headless)'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('force', default_value='', description='spot=class overrides, e.g. P03=back_in,L01=parallel'),
        OpaqueFunction(function=_setup),
    ])
