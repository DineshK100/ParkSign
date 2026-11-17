"""Pure-pursuit path tracker that handles reversing (Reeds-Shepp paths with cusps).

The path is split into segments wherever the driving direction changes. Each segment is
tracked with pure pursuit (same curvature law forwards and backwards for a rear-axle
reference point); the car stops at every cusp before switching direction.
ROS-free, so it is reused by the scan route and unit-testable.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TrackerConfig:
    lookahead: float = 1.6
    lookahead_min: float = 0.8
    v_max: float = 1.2
    v_max_reverse: float = 0.7
    v_min: float = 0.25
    slow_dist: float = 2.0        # start slowing this far from a segment end
    end_tol: float = 0.12         # segment end reached
    stop_time: float = 0.8        # s to stand still at a cusp


def _raw_segments(path):
    d = np.diff(path[:, :2], axis=0)
    dirs = np.sign(d[:, 0] * np.cos(path[:-1, 2]) + d[:, 1] * np.sin(path[:-1, 2]))
    dirs[dirs == 0] = 1
    segs, start = [], 0
    for i in range(1, len(dirs)):
        if dirs[i] != dirs[i - 1]:
            segs.append([path[start:i + 1], int(dirs[i - 1])])
            start = i
    segs.append([path[start:], int(dirs[-1])])
    return segs


def split_segments(path: np.ndarray, min_len: float = 0.4):
    """path: (N,3) x, y, yaw. Returns list of (points (M,3), direction +1/-1).
    Segments shorter than min_len (smoother artifacts) are dropped and their neighbours joined."""
    if len(path) < 2:
        return []
    segs = _raw_segments(path)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (pts, dr) in enumerate(segs):
            if path_length(pts) >= min_len:
                continue
            if 0 < i < len(segs) - 1 and segs[i - 1][1] == segs[i + 1][1]:
                segs[i - 1] = [np.vstack([segs[i - 1][0], segs[i + 1][0][1:]]), segs[i - 1][1]]
                del segs[i:i + 2]
            elif i == 0:
                segs[1][0] = np.vstack([pts[:1], segs[1][0][1:]]) if len(segs[1][0]) > 1 else segs[1][0]
                del segs[0]
            else:
                segs[i - 1][0] = np.vstack([segs[i - 1][0], pts[-1:]])
                del segs[i]
            changed = True
            break
    return [(pts, dr) for pts, dr in segs]


class SegmentFollower:
    def __init__(self, pts: np.ndarray, direction: int, car, cfg: TrackerConfig):
        self.dir = direction
        self.car, self.cfg = car, cfg
        # extend past the end along the final heading so pure pursuit aligns the car at the end
        end = pts[-1]
        ext = [end[:2] + direction * k * np.array([math.cos(end[2]), math.sin(end[2])])
               for k in np.arange(0.2, cfg.lookahead + 0.2, 0.2)]
        self.pts = np.vstack([pts[:, :2], ext]) if ext else pts[:, :2]
        self.n_real = len(pts)
        self.end = end
        self.idx = 0

    def remaining(self, x, y):
        """Signed distance to the end along the end heading (positive = not reached yet)."""
        ex, ey, eyaw = self.end
        return self.dir * ((ex - x) * math.cos(eyaw) + (ey - y) * math.sin(eyaw))

    def step(self, x, y, yaw):
        """Returns (v, steer, done)."""
        cfg = self.cfg
        rem = self.remaining(x, y)
        if rem < cfg.end_tol and self.idx >= self.n_real - 5:
            return 0.0, 0.0, True
        # closest point (search forward only)
        win = self.pts[self.idx:min(self.idx + 60, len(self.pts))]
        self.idx += int(np.argmin(np.linalg.norm(win - [x, y], axis=1)))
        ld = max(cfg.lookahead_min, min(cfg.lookahead, abs(rem) + cfg.lookahead_min))
        j = self.idx
        while j < len(self.pts) - 1 and np.hypot(*(self.pts[j] - [x, y])) < ld:
            j += 1
        tx, ty = self.pts[j]
        c, s = math.cos(yaw), math.sin(yaw)
        lx = c * (tx - x) + s * (ty - y)
        ly = -s * (tx - x) + c * (ty - y)
        L2 = max(lx * lx + ly * ly, 1e-6)
        kappa = 2.0 * ly / L2
        steer = math.atan(self.car.wheelbase * kappa)
        steer = max(-self.car.max_steer, min(self.car.max_steer, steer))
        v_max = cfg.v_max if self.dir > 0 else cfg.v_max_reverse
        v = cfg.v_min + (v_max - cfg.v_min) * min(1.0, max(0.0, rem) / cfg.slow_dist)
        return self.dir * v, steer, False


def to_twist(v, steer, wheelbase):
    """(speed, steering angle) -> (linear.x, angular.z) for the Ackermann plugin."""
    return v, v * math.tan(steer) / wheelbase


def path_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))) if len(path) > 1 else 0.0
