"""Rectangle collision checks (separating axis theorem) for the car footprint vs. lot obstacles."""
import math

import numpy as np


def rect_corners(cx, cy, yaw, sx, sy):
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = sx / 2, sy / 2
    return np.array([(cx + c * x - s * y, cy + s * x + c * y)
                     for x, y in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))])


def _axes(p):
    e = np.roll(p, -1, axis=0) - p
    n = np.stack([-e[:, 1], e[:, 0]], axis=1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def polygons_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    for axis in np.vstack([_axes(a), _axes(b)]):
        pa, pb = a @ axis, b @ axis
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False
    return True


def car_polygon(car, x, y, yaw, padding=0.0):
    """Car footprint corners in the map frame for base_link pose (x, y, yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([(x + c * px - s * py, y + s * px + c * py) for px, py in car.footprint(padding)])


class ObstacleSet:
    def __init__(self, rects):
        """rects: iterable of (cx, cy, yaw, sx, sy, kind)."""
        self.rects = [r for r in rects if r[5] != 'curb_visual']
        self.polys = [rect_corners(*r[:5]) for r in self.rects]
        self.centers = np.array([[r[0], r[1]] for r in self.rects]) if self.rects else np.zeros((0, 2))
        self.radii = np.array([math.hypot(r[3], r[4]) / 2 for r in self.rects])

    def collides(self, poly: np.ndarray) -> bool:
        c = poly.mean(axis=0)
        r = np.max(np.linalg.norm(poly - c, axis=1))
        near = np.linalg.norm(self.centers - c, axis=1) <= self.radii + r
        return any(polygons_overlap(poly, self.polys[i]) for i in np.nonzero(near)[0])
