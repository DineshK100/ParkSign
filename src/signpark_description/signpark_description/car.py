"""Car parameters and Gazebo (SDF) model generation.

Everything that needs the car's geometry (Gazebo model, Nav2 footprint, path
tracker, RL environment) imports it from here so the numbers never drift apart.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import yaml


def _default_config_path() -> str:
    # Installed: <share>/signpark_description/config/car.yaml
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('signpark_description'), 'config', 'car.yaml')
    except Exception:  # running from source (e.g. RL training without sourcing ROS)
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'car.yaml')


@dataclass
class CarParams:
    name: str = 'signpark_car'
    length: float = 4.5
    width: float = 1.8
    height: float = 1.2
    wheelbase: float = 2.7
    track: float = 1.55
    rear_overhang: float = 0.9
    wheel_radius: float = 0.33
    wheel_width: float = 0.22
    max_steer: float = 0.6
    max_speed: float = 2.0
    max_accel: float = 1.5
    mass: float = 1200.0
    cameras: dict = field(default_factory=dict)
    lidar: dict = field(default_factory=dict)

    @property
    def min_turning_radius(self) -> float:
        return self.wheelbase / math.tan(self.max_steer)

    @property
    def center_offset(self) -> float:
        """Distance from base_link (rear axle) forward to the geometric center."""
        return self.length / 2.0 - self.rear_overhang

    def footprint(self, padding: float = 0.0):
        """Footprint polygon in base_link frame (counter-clockwise)."""
        xr = -self.rear_overhang - padding
        xf = self.length - self.rear_overhang + padding
        hw = self.width / 2.0 + padding
        return [(xf, hw), (xr, hw), (xr, -hw), (xf, -hw)]

    def footprint_str(self, padding: float = 0.0) -> str:
        return '[' + ', '.join(f'[{x:.3f}, {y:.3f}]' for x, y in self.footprint(padding)) + ']'


def load_car(path: str | None = None) -> CarParams:
    with open(path or _default_config_path()) as f:
        data = yaml.safe_load(f)
    return CarParams(**data)


# ----------------------------------------------------------------------------
# SDF generation
# ----------------------------------------------------------------------------

def _box_inertia(m, x, y, z):
    return (m / 12.0 * (y * y + z * z), m / 12.0 * (x * x + z * z), m / 12.0 * (x * x + y * y))


def _inertial(m, ixx, iyy, izz):
    return (f'<inertial><mass>{m}</mass><inertia><ixx>{ixx:.4f}</ixx><iyy>{iyy:.4f}</iyy>'
            f'<izz>{izz:.4f}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>')


def _wheel_link(name, x, y, c: CarParams, parent_pose=''):
    m = 20.0
    r, w = c.wheel_radius, c.wheel_width
    ia = 0.5 * m * r * r
    ib = m * (3 * r * r + w * w) / 12.0
    return f"""
    <link name="{name}">
      <pose>{x} {y} {r} -1.5707963 0 0</pose>
      {_inertial(m, ib, ib, ia)}
      <visual name="visual">
        <geometry><cylinder><radius>{r}</radius><length>{w}</length></cylinder></geometry>
        <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material>
      </visual>
      <collision name="collision">
        <geometry><cylinder><radius>{r}</radius><length>{w}</length></cylinder></geometry>
        <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
      </collision>
    </link>"""


def _steer_link(name, x, y, c: CarParams):
    return f"""
    <link name="{name}">
      <pose>{x} {y} {c.wheel_radius} 0 0 0</pose>
      {_inertial(5.0, 0.05, 0.05, 0.05)}
    </link>"""


def _revolute(name, parent, child, axis='0 0 1', lower=None, upper=None, effort=1e4):
    lim = ''
    if lower is not None:
        lim = f'<limit><lower>{lower}</lower><upper>{upper}</upper><effort>{effort}</effort><velocity>2.0</velocity></limit>'
    else:
        lim = '<limit><lower>-1e16</lower><upper>1e16</upper></limit>'
    return f"""
    <joint name="{name}" type="revolute">
      <parent>{parent}</parent><child>{child}</child>
      <axis><xyz>{axis}</xyz>{lim}</axis>
    </joint>"""


def _camera(name, cam):
    x, y, z = cam['xyz']
    return f"""
      <sensor name="{name}_camera" type="camera">
        <pose>{x} {y} {z} 0 0 {cam['yaw']}</pose>
        <always_on>1</always_on>
        <update_rate>{cam['rate']}</update_rate>
        <topic>/camera/{name}/image_raw</topic>
        <gz_frame_id>camera_{name}_optical_frame</gz_frame_id>
        <camera>
          <horizontal_fov>{cam['hfov']}</horizontal_fov>
          <image><width>{cam['width']}</width><height>{cam['height']}</height><format>R8G8B8</format></image>
          <clip><near>0.1</near><far>60</far></clip>
          <camera_info_topic>/camera/{name}/camera_info</camera_info_topic>
        </camera>
      </sensor>"""


def car_sdf(c: CarParams, color=(0.85, 0.15, 0.1)) -> str:
    """Ackermann car as an SDF <model> string. Model frame = base_link (rear axle, ground)."""
    L, T = c.wheelbase, c.track
    body_len, body_w = c.length, c.width
    clearance = 0.2
    body_h = c.height - clearance
    cx = c.center_offset
    ixx, iyy, izz = _box_inertia(c.mass, body_len, body_w, body_h)
    lid = c.lidar
    cams = ''.join(_camera(n, cfg) for n, cfg in c.cameras.items())
    r, g, b = color
    rad = c.max_steer + 0.25  # inner wheel turns more than the bicycle-model angle
    return f"""<model name="{c.name}">
    <link name="base_link">
      <pose>0 0 0 0 0 0</pose>
      <inertial><pose>{cx} 0 {clearance + body_h / 2} 0 0 0</pose><mass>{c.mass}</mass>
        <inertia><ixx>{ixx:.2f}</ixx><iyy>{iyy:.2f}</iyy><izz>{izz:.2f}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <visual name="body">
        <pose>{cx} 0 {clearance + body_h / 2} 0 0 0</pose>
        <geometry><box><size>{body_len} {body_w} {body_h}</size></box></geometry>
        <material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>
      </visual>
      <visual name="windshield">
        <pose>{cx + 0.9} 0 {c.height - 0.05} 0 0 0</pose>
        <geometry><box><size>0.8 {body_w - 0.1} 0.12</size></box></geometry>
        <material><ambient>0.2 0.3 0.4 1</ambient><diffuse>0.2 0.3 0.4 1</diffuse></material>
      </visual>
      <collision name="body_collision">
        <pose>{cx} 0 {clearance + body_h / 2} 0 0 0</pose>
        <geometry><box><size>{body_len} {body_w} {body_h}</size></box></geometry>
      </collision>
      {cams}
      <sensor name="lidar" type="gpu_lidar">
        <pose>{' '.join(map(str, lid['xyz']))} 0 0 0</pose>
        <always_on>1</always_on>
        <update_rate>{lid['rate']}</update_rate>
        <topic>/scan</topic>
        <gz_frame_id>lidar_link</gz_frame_id>
        <lidar>
          <scan><horizontal><samples>{lid['samples']}</samples><resolution>1</resolution>
            <min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal></scan>
          <range><min>0.3</min><max>{lid['range_max']}</max><resolution>0.01</resolution></range>
        </lidar>
      </sensor>
    </link>
    {_steer_link('front_left_steer', L, T / 2, c)}
    {_steer_link('front_right_steer', L, -T / 2, c)}
    {_wheel_link('front_left_wheel', L, T / 2, c)}
    {_wheel_link('front_right_wheel', L, -T / 2, c)}
    {_wheel_link('rear_left_wheel', 0, T / 2, c)}
    {_wheel_link('rear_right_wheel', 0, -T / 2, c)}
    {_revolute('front_left_steer_joint', 'base_link', 'front_left_steer', lower=-rad, upper=rad)}
    {_revolute('front_right_steer_joint', 'base_link', 'front_right_steer', lower=-rad, upper=rad)}
    {_revolute('front_left_wheel_joint', 'front_left_steer', 'front_left_wheel')}
    {_revolute('front_right_wheel_joint', 'front_right_steer', 'front_right_wheel')}
    {_revolute('rear_left_wheel_joint', 'base_link', 'rear_left_wheel')}
    {_revolute('rear_right_wheel_joint', 'base_link', 'rear_right_wheel')}

    <plugin filename="gz-sim-ackermann-steering-system" name="gz::sim::systems::AckermannSteering">
      <left_joint>front_left_wheel_joint</left_joint>
      <left_joint>rear_left_wheel_joint</left_joint>
      <right_joint>front_right_wheel_joint</right_joint>
      <right_joint>rear_right_wheel_joint</right_joint>
      <left_steering_joint>front_left_steer_joint</left_steering_joint>
      <right_steering_joint>front_right_steer_joint</right_steering_joint>
      <kingpin_width>{T}</kingpin_width>
      <steering_limit>{rad}</steering_limit>
      <wheel_base>{L}</wheel_base>
      <wheel_separation>{T}</wheel_separation>
      <wheel_radius>{c.wheel_radius}</wheel_radius>
      <min_velocity>{-c.max_speed}</min_velocity>
      <max_velocity>{c.max_speed}</max_velocity>
      <min_acceleration>{-3 * c.max_accel}</min_acceleration>
      <max_acceleration>{c.max_accel}</max_acceleration>
      <topic>/cmd_vel</topic>
      <odom_topic>/wheel_odom_gz</odom_topic>
      <tf_topic>/wheel_tf_gz</tf_topic>
    </plugin>
    <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
      <odom_frame>odom</odom_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <odom_publish_frequency>50</odom_publish_frequency>
      <odom_topic>/odom</odom_topic>
      <tf_topic>/tf_gz</tf_topic>
      <dimensions>2</dimensions>
    </plugin>
    <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher">
      <topic>/joint_states</topic>
    </plugin>
  </model>"""


if __name__ == '__main__':
    car = load_car()
    print(f'min turning radius: {car.min_turning_radius:.2f} m, footprint: {car.footprint_str()}')
