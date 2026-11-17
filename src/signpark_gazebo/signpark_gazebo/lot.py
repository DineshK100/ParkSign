"""Parking-lot geometry and scenario randomization.

A *layout* (config/lot.yaml) describes where the spots are. A *scenario* adds
what changes between runs: which sign class / which photo each spot's post
carries and which spots are occupied by parked cars.
"""
from __future__ import annotations

import glob
import math
import os
import random
from dataclasses import asdict, dataclass, field

import yaml

SIGN_CLASSES = ['head_in', 'back_in', 'parallel', 'angled', 'no_parking']


def share_dir() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('signpark_gazebo')
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class Spot:
    id: str
    geometry: str            # perpendicular | parallel | angled
    cx: float                # spot center
    cy: float
    yaw: float               # "into the spot" heading (rad)
    width: float
    length: float
    sign_x: float = 0.0      # sign post position
    sign_y: float = 0.0
    sign_yaw: float = 0.0    # direction the sign face points (toward the aisle)
    allowed_classes: dict = field(default_factory=dict)
    # scenario fields
    sign_class: str = 'unknown'
    sign_image: str = ''
    occupied: bool = False

    def corners(self):
        """Spot rectangle corners (map frame)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        hl, hw = self.length / 2, self.width / 2
        pts = [(hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)]
        return [(self.cx + c * x - s * y, self.cy + s * x + c * y) for x, y in pts]


@dataclass
class Lot:
    bounds: dict
    entrance: tuple
    spawn: dict
    aisle: dict
    sign: dict
    occupancy_prob: float
    spots: list

    def spot(self, sid: str) -> Spot:
        for s in self.spots:
            if s.id == sid:
                return s
        raise KeyError(sid)

    def dist_to_entrance(self, s: Spot) -> float:
        return math.hypot(s.cx - self.entrance[0], s.cy - self.entrance[1])


def load_layout(path: str | None = None) -> Lot:
    path = path or os.path.join(share_dir(), 'config', 'lot.yaml')
    with open(path) as f:
        cfg = yaml.safe_load(f)
    spots = []
    sign_cfg = cfg['sign']
    for row in cfg['rows']:
        yaw = math.radians(row['into_yaw_deg'])
        dx, dy = math.cos(yaw), math.sin(yaw)
        for i in range(row['count']):
            if 'first_entry' in row:  # angled: given entry midpoint, center is half a length inside
                ex = row['first_entry'][0] + i * row['step'][0]
                ey = row['first_entry'][1] + i * row['step'][1]
                cx, cy = ex + dx * row['length'] / 2, ey + dy * row['length'] / 2
            else:
                cx = row['first_center'][0] + i * row['step'][0]
                cy = row['first_center'][1] + i * row['step'][1]
            if row['geometry'] == 'parallel':
                # sign on the curb side (to the right of the parked heading), facing the aisle
                nx, ny = math.sin(yaw), -math.cos(yaw)
                off = row['width'] / 2 + sign_cfg['setback']
            else:
                nx, ny = dx, dy
                off = row['length'] / 2 + sign_cfg['setback']
            spots.append(Spot(
                id=f"{row['prefix']}{i + 1:02d}", geometry=row['geometry'],
                cx=cx, cy=cy, yaw=yaw, width=row['width'], length=row['length'],
                sign_x=cx + nx * off, sign_y=cy + ny * off,
                sign_yaw=math.atan2(-ny, -nx),
                allowed_classes=dict(row['sign_classes'])))
    return Lot(bounds=cfg['bounds'], entrance=tuple(cfg['entrance']), spawn=cfg['spawn'],
               aisle=cfg['aisle'], sign=sign_cfg, occupancy_prob=cfg['occupancy_prob'], spots=spots)


def find_sign_images(signs_dir: str | None = None) -> dict:
    """{class: [image paths]}. Real photos go in signs/<class>/; synthetic placeholders are
    used only for classes that have no real photos yet."""
    signs_dir = signs_dir or os.path.join(share_dir(), 'signs')
    out = {}
    for cls in SIGN_CLASSES:
        files = sorted(f for ext in ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')
                       for f in glob.glob(os.path.join(signs_dir, cls, f'*.{ext}')))
        real = [f for f in files if not os.path.basename(f).startswith('synthetic_')]
        out[cls] = real or files
    return out


def make_scenario(lot: Lot, seed: int, signs_dir: str | None = None,
                  force: dict | None = None) -> Lot:
    """Randomize sign classes, sign photos and occupancy. `force` = {spot_id: class}."""
    rng = random.Random(seed)
    images = find_sign_images(signs_dir)
    for s in lot.spots:
        classes, weights = zip(*s.allowed_classes.items())
        s.sign_class = rng.choices(classes, weights=weights)[0]
        if force and s.id in force:
            s.sign_class = force[s.id]
        pool = images.get(s.sign_class) or []
        s.sign_image = rng.choice(pool) if pool else ''
        s.occupied = rng.random() < lot.occupancy_prob
    # always keep at least one legal free spot of each geometry
    for geom in ('perpendicular', 'parallel', 'angled'):
        legal = [s for s in lot.spots if s.geometry == geom and s.sign_class != 'no_parking']
        if legal and all(s.occupied for s in legal):
            rng.choice(legal).occupied = False
    return lot


def save_scenario(lot: Lot, path: str, seed: int):
    data = {'seed': seed, 'entrance': list(lot.entrance), 'spawn': lot.spawn,
            'spots': [asdict(s) for s in lot.spots]}
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_scenario(path: str, layout_path: str | None = None) -> Lot:
    """Layout + the scenario fields saved by save_scenario()."""
    lot = load_layout(layout_path)
    with open(path) as f:
        data = yaml.safe_load(f)
    by_id = {d['id']: d for d in data['spots']}
    for s in lot.spots:
        d = by_id[s.id]
        s.sign_class, s.sign_image, s.occupied = d['sign_class'], d['sign_image'], d['occupied']
    return lot


def default_scenario_dir(seed: int) -> str:
    return os.path.join(os.path.expanduser('~'), '.ros', 'signpark', f'scenario_{seed}')
