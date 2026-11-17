"""Project a sign board (known position from the lot map) into the front camera image."""
from __future__ import annotations

import math

import numpy as np

BOARD_FORWARD = 0.111  # face offset in front of the pole, see world_gen._sign_post


def board_corners_world(sign_x, sign_y, sign_yaw, center_z, width, height):
    """4 corners (3D, map frame) of the sign face."""
    fx, fy = math.cos(sign_yaw), math.sin(sign_yaw)
    cx, cy = sign_x + BOARD_FORWARD * fx, sign_y + BOARD_FORWARD * fy
    lx, ly = -fy, fx  # lateral direction of the board
    pts = []
    for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        pts.append((cx + a * width / 2 * lx, cy + a * width / 2 * ly, center_z + b * height / 2))
    return np.array(pts)


def project(points_world, car_x, car_y, car_yaw, cam_xyz, cam_yaw, img_w, img_h, hfov, K=None):
    """Pinhole projection for a level camera mounted at cam_xyz with heading cam_yaw (base_link).
    Returns (N,2) pixel coords and (N,) depth (distance along the optical axis)."""
    c, s = math.cos(car_yaw), math.sin(car_yaw)
    camx = car_x + c * cam_xyz[0] - s * cam_xyz[1]
    camy = car_y + s * cam_xyz[0] + c * cam_xyz[1]
    c, s = math.cos(car_yaw + cam_yaw), math.sin(car_yaw + cam_yaw)
    dx, dy = points_world[:, 0] - camx, points_world[:, 1] - camy
    dz = points_world[:, 2] - cam_xyz[2]
    fwd = c * dx + s * dy
    left = -s * dx + c * dy
    if K is not None:
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    else:
        fx = fy = (img_w / 2) / math.tan(hfov / 2)
        cx, cy = img_w / 2, img_h / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        u = cx - fx * left / fwd
        v = cy - fy * dz / fwd
    return np.stack([u, v], axis=1), fwd


def sign_view(spot, car_pose, cam, board, img_w, img_h, K=None, margin=8):
    """Return dict(bbox=(u0,v0,u1,v1), width_px, distance, facing) if the board is fully in view, else None."""
    x, y, yaw = car_pose
    pts = board_corners_world(spot.sign_x, spot.sign_y, spot.sign_yaw, board['center_z'],
                              board['width'], board['height'])
    uv, depth = project(pts, x, y, yaw, cam['xyz'], cam['yaw'], img_w, img_h, cam['hfov'], K)
    if np.any(depth < 1.0):
        return None
    u0, v0 = uv.min(axis=0)
    u1, v1 = uv.max(axis=0)
    if u0 < margin or v0 < margin or u1 > img_w - margin or v1 > img_h - margin:
        return None
    # how squarely the sign faces the camera (1 = head on)
    vx, vy = x - spot.sign_x, y - spot.sign_y
    dist = math.hypot(vx, vy)
    facing = (math.cos(spot.sign_yaw) * vx + math.sin(spot.sign_yaw) * vy) / max(dist, 1e-6)
    return dict(bbox=(int(u0), int(v0), int(math.ceil(u1)), int(math.ceil(v1))),
                width_px=float(u1 - u0), distance=dist, facing=facing)


def crop(img, bbox, pad_frac=0.15):
    u0, v0, u1, v1 = bbox
    pw, ph = int((u1 - u0) * pad_frac), int((v1 - v0) * pad_frac)
    h, w = img.shape[:2]
    return img[max(0, v0 - ph):min(h, v1 + ph), max(0, u0 - pw):min(w, u1 + pw)].copy()
