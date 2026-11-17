"""Generate the Gazebo world (SDF) and the Nav2 occupancy map for one scenario.

    ros2 run signpark_gazebo make_scenario --seed 3
writes ~/.ros/signpark/scenario_3/{world.sdf, scenario.yaml, map.pgm, map.yaml}.
The sim launch file calls the same functions, so you rarely need to run it by hand.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil

import numpy as np

from signpark_description.car import car_sdf, load_car
from signpark_gazebo.lot import (Lot, default_scenario_dir, load_layout, make_scenario,
                                 save_scenario)

PARKED_CAR = dict(length=4.5, width=1.8, height=1.5)
WALL_H = 2.0
WALL_T = 0.3
CURB_H = 0.15


def _box(name, x, y, z, yaw, sx, sy, sz, rgba, collide=True, static=True):
    r, g, b, a = rgba
    col = f'<collision name="c"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>' if collide else ''
    return f"""
  <model name="{name}"><static>{'true' if static else 'false'}</static>
    <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>
    <link name="link">
      <visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse></material></visual>
      {col}
    </link>
  </model>"""


def obstacles(lot: Lot):
    """Static obstacles as rotated rectangles (cx, cy, yaw, sx, sy, kind). Shared by world + map + RL env."""
    b = lot.bounds
    w, h = b['xmax'] - b['xmin'], b['ymax'] - b['ymin']
    mx, my = (b['xmax'] + b['xmin']) / 2, (b['ymax'] + b['ymin']) / 2
    obs = [
        (mx, b['ymax'], 0.0, w, WALL_T, 'wall'),
        (mx, b['ymin'], 0.0, w, WALL_T, 'wall'),
        (b['xmin'], my, 0.0, WALL_T, h, 'wall'),
        (b['xmax'], my, 0.0, WALL_T, h, 'wall'),
    ]
    # curb behind the parallel spots
    par = [s for s in lot.spots if s.geometry == 'parallel']
    if par:
        x0 = min(s.cx - s.length / 2 for s in par)
        x1 = max(s.cx + s.length / 2 for s in par)
        y = par[0].cy - par[0].width / 2 - 0.15
        obs.append(((x0 + x1) / 2, y, 0.0, x1 - x0, 0.3, 'curb'))
    for s in lot.spots:
        obs.append((s.sign_x, s.sign_y, 0.0, 0.12, 0.12, 'post'))
        if s.occupied:
            obs.append((s.cx, s.cy, s.yaw, PARKED_CAR['length'], PARKED_CAR['width'], 'car'))
    return obs


def _stripes(lot: Lot):
    """White painted lines (visual only)."""
    out = []
    for s in lot.spots:
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        if s.geometry == 'parallel':
            # short ticks at both ends
            for sign in (-1, 1):
                x = s.cx + sign * s.length / 2 * c
                y = s.cy + sign * s.length / 2 * sn
                out.append((x, y, s.yaw + math.pi / 2, s.width, 0.12))
        else:
            # two side lines along the spot
            for sign in (-1, 1):
                x = s.cx - sign * s.width / 2 * sn
                y = s.cy + sign * s.width / 2 * c
                out.append((x, y, s.yaw, s.length, 0.12))
    return out


def _sign_post(s, lot: Lot):
    sg = lot.sign
    ph, bw, bh = sg['post_height'], sg['board_width'], sg['board_height']
    img = s.sign_image
    face = ''
    if img:
        face = f"""
      <visual name="face">
        <pose>0.111 0 {ph} 0 0 0</pose>
        <geometry><box><size>0.002 {bw} {bh}</size></box></geometry>
        <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><specular>0.1 0.1 0.1 1</specular>
          <pbr><metal><albedo_map>{img}</albedo_map><metalness>0.0</metalness><roughness>0.8</roughness></metal></pbr>
        </material>
      </visual>"""
    return f"""
  <model name="sign_{s.id}"><static>true</static>
    <pose>{s.sign_x:.3f} {s.sign_y:.3f} 0 0 0 {s.sign_yaw:.4f}</pose>
    <link name="link">
      <visual name="pole"><pose>0 0 {ph / 2} 0 0 0</pose>
        <geometry><cylinder><radius>0.04</radius><length>{ph}</length></cylinder></geometry>
        <material><ambient>0.5 0.5 0.5 1</ambient><diffuse>0.5 0.5 0.5 1</diffuse></material></visual>
      <collision name="pole_c"><pose>0 0 {ph / 2} 0 0 0</pose>
        <geometry><cylinder><radius>0.06</radius><length>{ph}</length></cylinder></geometry></collision>
      <visual name="board"><pose>0.08 0 {ph} 0 0 0</pose>
        <geometry><box><size>0.06 {bw + 0.04} {bh + 0.04}</size></box></geometry>
        <material><ambient>0.3 0.3 0.3 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material></visual>
      {face}
    </link>
  </model>"""


def world_sdf(lot: Lot, car_xml: str) -> str:
    b = lot.bounds
    parts = []
    gx, gy = (b['xmax'] + b['xmin']) / 2, (b['ymax'] + b['ymin']) / 2
    parts.append(_box('asphalt', gx, gy, -0.048, 0, b['xmax'] - b['xmin'] + 10, b['ymax'] - b['ymin'] + 10, 0.1,
                      (0.22, 0.22, 0.24, 1)))
    for i, (x, y, yaw, sx, sy) in enumerate(_stripes(lot)):
        parts.append(_box(f'stripe_{i}', x, y, 0.006, yaw, sx, sy, 0.006, (0.95, 0.95, 0.95, 1), collide=False))
    ex, ey = lot.entrance
    parts.append(_box('entrance_marker', ex, ey, 0.006, 0, 1.0, 7.0, 0.006, (0.95, 0.8, 0.1, 1), collide=False))
    palette = [(0.2, 0.3, 0.7, 1), (0.8, 0.8, 0.8, 1), (0.1, 0.1, 0.1, 1), (0.3, 0.5, 0.3, 1), (0.6, 0.6, 0.65, 1)]
    k = 0
    for (x, y, yaw, sx, sy, kind) in obstacles(lot):
        if kind == 'wall':
            parts.append(_box(f'wall_{k}', x, y, WALL_H / 2, yaw, sx, sy, WALL_H, (0.6, 0.55, 0.5, 1)))
        elif kind == 'curb':
            parts.append(_box(f'curb_{k}', x, y, CURB_H / 2, yaw, sx, sy, CURB_H, (0.75, 0.75, 0.72, 1)))
        elif kind == 'car':
            col = palette[k % len(palette)]
            parts.append(_box(f'parked_{k}', x, y, 0.2 + PARKED_CAR['height'] / 2 - 0.1, yaw,
                              sx, sy, PARKED_CAR['height'] - 0.2, col))
        k += 1
    for s in lot.spots:
        parts.append(_sign_post(s, lot))
    sp = lot.spawn
    car_xml = car_xml.replace('<model name=', f'<model name=', 1)
    car_inc = car_xml.replace('<link name="base_link">',
                              f'<pose>{sp["x"]} {sp["y"]} 0.02 0 0 {sp["yaw"]}</pose>\n    <link name="base_link">', 1)
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
<world name="signpark_lot">
  <physics name="2ms" type="ignored"><max_step_size>0.002</max_step_size><real_time_factor>1.0</real_time_factor></physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
  <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.7 0.8 0.95 1</background><shadows>false</shadows></scene>
  <light type="directional" name="sun"><cast_shadows>false</cast_shadows><pose>0 0 30 0 0 0</pose>
    <diffuse>0.9 0.9 0.85 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.4 0.3 -0.9</direction></light>
  <model name="ground"><static>true</static><link name="link">
    <collision name="c"><geometry><plane><normal>0 0 1</normal><size>300 300</size></plane></geometry>
      <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
    <visual name="v"><geometry><plane><normal>0 0 1</normal><size>300 300</size></plane></geometry>
      <material><ambient>0.4 0.5 0.35 1</ambient><diffuse>0.4 0.5 0.35 1</diffuse></material></visual>
  </link></model>
  {''.join(parts)}
  {car_inc}
</world>
</sdf>
"""


def rasterize(lot: Lot, resolution=0.1, kinds=('wall', 'curb', 'post', 'car')):
    """Occupancy grid (uint8, 1 = occupied), row 0 = ymin. Returns (grid, origin_x, origin_y)."""
    b = lot.bounds
    m = 1.0
    x0, y0 = b['xmin'] - m, b['ymin'] - m
    nx = int(math.ceil((b['xmax'] - b['xmin'] + 2 * m) / resolution))
    ny = int(math.ceil((b['ymax'] - b['ymin'] + 2 * m) / resolution))
    xs = x0 + (np.arange(nx) + 0.5) * resolution
    ys = y0 + (np.arange(ny) + 0.5) * resolution
    X, Y = np.meshgrid(xs, ys)
    grid = np.zeros((ny, nx), dtype=np.uint8)
    for (cx, cy, yaw, sx, sy, kind) in obstacles(lot):
        if kind not in kinds:
            continue
        c, s = math.cos(yaw), math.sin(yaw)
        lx = (X - cx) * c + (Y - cy) * s
        ly = -(X - cx) * s + (Y - cy) * c
        grid[(np.abs(lx) <= sx / 2) & (np.abs(ly) <= sy / 2)] = 1
    return grid, x0, y0


def write_map(lot: Lot, out_dir: str, resolution=0.1):
    grid, x0, y0 = rasterize(lot, resolution)
    img = np.where(grid[::-1] > 0, 0, 254).astype(np.uint8)  # PGM row 0 = top = ymax
    pgm = os.path.join(out_dir, 'map.pgm')
    with open(pgm, 'wb') as f:
        f.write(f'P5\n{img.shape[1]} {img.shape[0]}\n255\n'.encode())
        f.write(img.tobytes())
    with open(os.path.join(out_dir, 'map.yaml'), 'w') as f:
        f.write(f'image: map.pgm\nmode: trinary\nresolution: {resolution}\n'
                f'origin: [{x0}, {y0}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')


def generate(seed: int, out_dir: str | None = None, layout: str | None = None,
             signs_dir: str | None = None, force: dict | None = None) -> str:
    out_dir = out_dir or default_scenario_dir(seed)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    lot = make_scenario(load_layout(layout), seed, signs_dir, force)
    car = load_car()
    with open(os.path.join(out_dir, 'world.sdf'), 'w') as f:
        f.write(world_sdf(lot, car_sdf(car)))
    save_scenario(lot, os.path.join(out_dir, 'scenario.yaml'), seed)
    write_map(lot, out_dir)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--signs-dir', default=None)
    ap.add_argument('--force', nargs='*', default=[], help='spot=class overrides, e.g. P03=back_in')
    a = ap.parse_args()
    force = dict(kv.split('=') for kv in a.force)
    out = generate(a.seed, a.out, signs_dir=a.signs_dir, force=force)
    print(out)


if __name__ == '__main__':
    main()
