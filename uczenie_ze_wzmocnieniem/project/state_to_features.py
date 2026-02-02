import math

import numpy as np

def get_state_features(car, state):
    cx, cy = car.get_center()
    border_distances = state[0]
    car_distances = state[1]
    checkpoints = state[3]

    SENSE_DIST = 250.0
    CAR_MAX = 200.0

    border_norm = [min(d / SENSE_DIST, 1.0) for d in border_distances]
    cars_norm = [min(d / CAR_MAX, 1.0) for d in car_distances]

    cx, cy = car.get_center()

    ncp = len(checkpoints)
    cp0_idx = car.checkpoint_index % ncp
    cp1_idx = (cp0_idx + 1) % ncp
    cp2_idx = (cp0_idx + 2) % ncp

    cp0_x, cp0_y = checkpoints[cp0_idx]
    cp1_x, cp1_y = checkpoints[cp1_idx]
    cp2_x, cp2_y = checkpoints[cp2_idx]

    # --- wektor kierunku auta (forward) ---
    a = math.radians(car.angle)
    fx = math.sin(a)
    fy = -math.cos(a)

    def angle_and_dist_to(px, py):
        dx = px - cx
        dy = py - cy
        dist = math.hypot(dx, dy)
        dist_norm = min(dist / SENSE_DIST, 1.0)

        if dist > 1e-6:
            vx = dx / dist
            vy = dy / dist
        else:
            vx, vy = 0.0, 0.0

        dot = fx * vx + fy * vy
        cross = fx * vy - fy * vx
        ang = math.atan2(cross, dot)  # [-pi, pi]
        return ang, dist_norm, dot  # dot = "towards"

    ang0, dist0, towards0 = angle_and_dist_to(cp0_x, cp0_y)
    ang1, dist1, towards1 = angle_and_dist_to(cp1_x, cp1_y)
    ang2, dist2, towards2 = angle_and_dist_to(cp2_x, cp2_y)

    # prędkość (sklipuj do [-1,1])
    vel_norm = max(-1.0, min(car.vel / 8.0, 1.0))

    curve_signal = ang2 - ang1

    features = border_norm + cars_norm + [
        # CP0
        dist0,
        math.sin(ang0),
        math.cos(ang0),
        towards0,

        dist1,
        math.sin(ang1),
        math.cos(ang1),
        towards1,

        dist2,
        math.sin(ang2),
        math.cos(ang2),
        towards2,

        # stan auta
        vel_norm,
        fx, fy,

        curve_signal
    ]

    return np.array(features, dtype=np.float32)
