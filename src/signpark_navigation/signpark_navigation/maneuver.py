"""Maneuver mapping: (spot, sign class) -> goal pose of base_link (rear axle) and required heading."""
import math
from dataclasses import dataclass

PARKABLE = ('head_in', 'back_in', 'parallel', 'angled')


@dataclass
class ParkGoal:
    x: float
    y: float
    yaw: float
    sign_class: str
    description: str
    geometry: str = ''


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def default_class(geometry: str) -> str:
    """What a sign-blind system does: the 'natural' maneuver for the geometry (E3 baseline)."""
    return {'perpendicular': 'head_in', 'parallel': 'parallel', 'angled': 'angled'}[geometry]


def park_goal(spot, sign_class: str, car) -> ParkGoal:
    """spot: anything with cx, cy, yaw (into-spot heading) and geometry.
    The car's geometric center is placed on the spot center."""
    if sign_class not in PARKABLE:
        raise ValueError(f'cannot park in a spot marked {sign_class!r}')
    yaw = spot.yaw + (math.pi if sign_class == 'back_in' else 0.0)
    yaw = wrap(yaw)
    d = car.center_offset
    x = spot.cx - d * math.cos(yaw)
    y = spot.cy - d * math.sin(yaw)
    desc = {
        'head_in': 'drive in forwards',
        'back_in': 'reverse into the spot (nose out)',
        'parallel': 'parallel park along the curb',
        'angled': 'pull into the angled spot',
    }[sign_class]
    return ParkGoal(x, y, yaw, sign_class, desc, spot.geometry)


def heading_error_deg(yaw, goal_yaw):
    return abs(math.degrees(wrap(yaw - goal_yaw)))


def complies(final_yaw, spot, sign_class, tol_deg=30.0) -> bool:
    """Did the car end in the orientation the sign requires (e.g. nose out for back-in)?"""
    req = wrap(spot.yaw + (math.pi if sign_class == 'back_in' else 0.0))
    return heading_error_deg(final_yaw, req) <= tol_deg
