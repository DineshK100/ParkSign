"""Nav2 map server + Smac Hybrid-A* planner + SignPark park server.

    ros2 launch signpark_navigation navigation.launch.py seed:=1
"""
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from signpark_description.car import load_car
from signpark_gazebo.lot import default_scenario_dir


def _setup(context):
    seed = int(LaunchConfiguration('seed').perform(context))
    results = LaunchConfiguration('results_csv').perform(context)
    sdir = default_scenario_dir(seed)
    car = load_car()
    share = get_package_share_directory('signpark_navigation')
    with open(os.path.join(share, 'config', 'nav2_params.yaml')) as f:
        params = f.read()
    params = params.replace('__FOOTPRINT__', car.footprint_str(padding=0.1))
    # +0.5 m: the simulated car understeers ~10% (measured), plan within what it can really do
    params = params.replace('__MIN_RADIUS__', f'{car.min_turning_radius + 0.5:.2f}')
    tmp = tempfile.NamedTemporaryFile('w', suffix='_nav2.yaml', delete=False)
    tmp.write(params)
    tmp.close()
    common = {'use_sim_time': True}
    return [
        Node(package='nav2_map_server', executable='map_server', name='map_server', output='screen',
             parameters=[tmp.name, {'yaml_filename': os.path.join(sdir, 'map.yaml')}, common]),
        Node(package='nav2_planner', executable='planner_server', name='planner_server', output='screen',
             parameters=[tmp.name, common]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_planning',
             output='screen', parameters=[common, {'autostart': True,
                                                   'node_names': ['map_server', 'planner_server']}]),
        Node(package='signpark_navigation', executable='park_server', output='screen',
             parameters=[common, {'seed': seed, 'results_csv': results}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('results_csv', default_value='results/park_trials.csv'),
        OpaqueFunction(function=_setup),
    ])
