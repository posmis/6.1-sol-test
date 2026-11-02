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

    # Drawing accuracy is independent of mesh size. Bound integration cost on 1024².
    ds = max(.45*min(dx, dy), length/1800)
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


class Tracers:
    """Passive display tracers integrated in physical time through saved fields.

    RK2 in space/time, limited to half a display cell per substep. Respawning
    particles and scrubbing reset trails; they never connect across a solid.
    """
    def __init__(self, result, count=280):
        self.result = result
        self.rng = np.random.default_rng(42)
        self.cells = np.argwhere(result.fluid)
        self.points = self.spawn(count)
        self.time = None
        self.trail = []

    def spawn(self, count):
        c = self.result.config
        cells = self.cells[self.rng.integers(len(self.cells), size=count)]
        xy = cells[:, ::-1]+self.rng.uniform(.1, .9, (count, 2))
        return xy*np.array([c.length/c.nx, c.height/c.ny])

    def advance(self, time):
        r, c = self.result, self.result.config
        dx, dy = c.length/c.nx, c.height/c.ny
        if self.time is None or time < self.time or time-self.time > c.duration/8:
            self.points = self.spawn(len(self.points))
            self.trail = []
            self.time = time
        def sample(points, t):
            i = int(np.clip(np.searchsorted(r.time, t)-1, 0, len(r.time)-2))
            f = np.clip((t-r.time[i])/(r.time[i+1]-r.time[i]), 0, 1)
            coords = np.array([points[:, 1]/dy-.5, points[:, 0]/dx-.5])
            return np.column_stack([(1-f)*map_coordinates(q[i], coords, order=1, mode='nearest')+
                f*map_coordinates(q[i+1], coords, order=1, mode='nearest') for q in (r.u, r.v)])
        def valid(points):
            ix, iy = np.floor(points[:, 0]/dx).astype(int), np.floor(points[:, 1]/dy).astype(int)
            return ((ix >= 0) & (ix < c.nx) & (iy >= 0) & (iy < c.ny) &
                    r.fluid[np.clip(iy, 0, c.ny-1), np.clip(ix, 0, c.nx-1)])
        moved = time > self.time
        while self.time < time-1e-14:
            speed = sample(self.points, self.time)
            dt = min(time-self.time, .45*min(dx, dy)/max(np.linalg.norm(speed, axis=1).max(), c.speed, 1e-12))
            mid = self.points+speed*dt*.5
            new = self.points+sample(mid, self.time+dt*.5)*dt
            bad = ~(valid(mid) & valid(new))
            new[bad] = self.spawn(int(bad.sum()))
            for previous in self.trail:
                previous[bad] = np.nan
            self.points = new
            self.time += dt
        if moved:
            self.trail.append(self.points.copy())
            self.trail = self.trail[-14:]
        if len(self.trail) < 2:
            return self.points, np.empty((0, 2))
        trails = np.stack(self.trail, axis=1)
        trails = np.concatenate([trails, np.full((len(trails), 1, 2), np.nan)], axis=1)
        return self.points, trails.reshape(-1, 2)
