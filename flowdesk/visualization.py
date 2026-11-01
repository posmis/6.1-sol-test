"""Streamlines of a frozen velocity field, integrated with midpoint RK2."""
import numpy as np
from scipy.ndimage import map_coordinates


def streamlines(u, v, fluid, length, height, count=16, max_steps=None):
    ny, nx = u.shape
    dx, dy = length/nx, height/ny
    # Inlet seeds plus internal seeds to reveal closed recirculation regions.
    seeds = [(dx*0.6, y) for y in np.linspace(dy, height-dy, count)]
    seeds += [(x, y) for x in np.linspace(length*.23, length*.65, 5)
              for y in (height*.12, height*.28)]
    points = np.array(seeds)
    paths = [[p.copy()] for p in points]
    active = np.ones(len(points), bool)

    def sample(p):
        coord = np.array([p[:, 1]/dy-.5, p[:, 0]/dx-.5])
        vel = np.column_stack([map_coordinates(q, coord, order=1, mode="nearest") for q in (u, v)])
        speed = np.linalg.norm(vel, axis=1)
        return vel/np.maximum(speed[:, None], 1e-12), speed

    ds = .45*min(dx, dy)
    if max_steps is None:
        max_steps = min(4000, int(2*length/ds)+1)
    for _ in range(max_steps):
        direction, speed = sample(points)
        midpoint = points+.5*ds*direction
        direction, _ = sample(midpoint)
        new = points+ds*direction
        ix, iy = (new[:, 0]/dx).astype(int), (new[:, 1]/dy).astype(int)
        valid = ((new[:, 0] > 0) & (new[:, 0] < length) &
                 (new[:, 1] > 0) & (new[:, 1] < height) & (speed > 1e-5))
        valid &= fluid[np.clip(iy, 0, ny-1), np.clip(ix, 0, nx-1)]
        active &= valid
        for i in np.flatnonzero(active):
            paths[i].append(new[i].copy())
        points = new
        if not active.any():
            break
    return [np.array(path) for path in paths if len(path) > 3]
