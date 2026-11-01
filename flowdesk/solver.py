"""Staggered-grid projection, donor-cell transport and high-Re k-epsilon.

SI units throughout. The pressure solve uses mechanical RANS pressure p/rho+2k/3;
saved results contain static pressure relative to the outlet mean.
Smooth equilibrium wall functions model wall stress.
"""
from dataclasses import dataclass, asdict
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized


@dataclass(frozen=True)
class Config:
    scenario: str = "channel"
    nx: int = 96
    ny: int = 24
    length: float = 0.6
    height: float = 0.04
    speed: float = 1.0
    rho: float = 998.2
    nu: float = 1.004e-6
    duration: float = 1.2
    frames: int = 80
    intensity: float = 0.05

    def validate(self):
        if self.scenario not in ("channel", "step"):
            raise ValueError("Unknown scenario")
        if not (16 <= self.nx <= 384 and 8 <= self.ny <= 128):
            raise ValueError("Grid must be 16..384 × 8..128")
        for name in ("length", "height", "speed", "rho", "nu", "duration"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 2 <= self.frames <= 500 or not 0.001 <= self.intensity <= 0.2:
            raise ValueError("Invalid frames or turbulence intensity")


@dataclass
class Result:
    config: Config
    time: np.ndarray
    u: np.ndarray
    v: np.ndarray
    pressure: np.ndarray
    k: np.ndarray
    epsilon: np.ndarray
    fluid: np.ndarray
    diagnostics: dict

    def save(self, path):
        np.savez_compressed(path, config=json.dumps(asdict(self.config)),
                            diagnostics=json.dumps(self.diagnostics), time=self.time,
                            u=self.u, v=self.v, pressure=self.pressure, k=self.k,
                            epsilon=self.epsilon, fluid=self.fluid)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            cfg = Config(**json.loads(str(z["config"])))
            cfg.validate()
            arrays = {key: z[key].copy() for key in
                      ("time", "u", "v", "pressure", "k", "epsilon", "fluid")}
            shape = (len(arrays["time"]), cfg.ny, cfg.nx)
            if (shape[0] != cfg.frames or arrays["time"].ndim != 1
                or not np.isfinite(arrays["time"]).all()
                or not np.all(np.diff(arrays["time"]) > 0)
                or arrays["fluid"].dtype != bool
                or arrays["fluid"].shape != shape[1:]) or any(
                arrays[key].shape != shape or not np.isfinite(arrays[key]).all()
                for key in ("u", "v", "pressure", "k", "epsilon")
            ):
                raise ValueError("Invalid result dimensions or non-finite fields")
            return cls(cfg, **arrays, diagnostics=json.loads(str(z["diagnostics"])))


def diffusion(q, nu, mask, dx, dy):
    """Conservative diffusion; absent neighbours have zero normal flux."""
    out = np.zeros_like(q)
    for axis, spacing in ((0, dy), (1, dx)):
        a = [slice(None), slice(None)]
        b = a.copy()
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        flux = (q[b] - q[a]) * (nu[a] + nu[b]) * 0.5 / spacing**2
        flux *= mask[a] & mask[b]
        out[a] += flux
        out[b] -= flux
    return out


def upwind(q, ax, ay, dx, dy):
    p = np.pad(q, 1, mode="edge")
    return (np.maximum(ax, 0)*(q-p[1:-1, :-2])/dx
            + np.minimum(ax, 0)*(p[1:-1, 2:]-q)/dx
            + np.maximum(ay, 0)*(q-p[:-2, 1:-1])/dy
            + np.minimum(ay, 0)*(p[2:, 1:-1]-q)/dy)


class Solver:
    def __init__(self, config=Config()):
        config.validate()
        self.c = c = config
        self.dx, self.dy = c.length/c.nx, c.height/c.ny
        self.fluid = np.ones((c.ny, c.nx), dtype=bool)
        if c.scenario == "step":
            self.fluid[:c.ny//2, :c.nx//5] = False
        f = self.fluid
        self.um = np.zeros((c.ny, c.nx+1), bool)
        self.vm = np.zeros((c.ny+1, c.nx), bool)
        self.um[:, 1:-1] = f[:, :-1] & f[:, 1:]
        self.um[:, 0], self.um[:, -1] = f[:, 0], f[:, -1]
        self.vm[1:-1] = f[:-1] & f[1:]
        self.u = self.um.astype(float)*c.speed
        self.v = np.zeros_like(self.vm, float)
        self.p = np.zeros_like(f, float)
        self.k = np.full(f.shape, 1.5*(c.speed*c.intensity)**2)
        self.e = 0.09**0.75*self.k**1.5/(0.07*c.height)
        self.kin, self.ein = float(self.k[0, 0]), float(self.e[0, 0])
        self.walls = []
        for axis, sign, spacing in ((0, -1, self.dy), (0, 1, self.dy),
                                     (1, -1, self.dx), (1, 1, self.dx)):
            neighbour = np.roll(f, -sign, axis=axis)
            edge = [slice(None), slice(None)]
            edge[axis] = -1 if sign == 1 else 0
            neighbour[tuple(edge)] = axis == 1  # open x boundaries
            self.walls.append((axis, spacing, f & ~neighbour))
        self.wall = np.logical_or.reduce([w[2] for w in self.walls])
        self._pressure_matrix()
        self.project(1.0)  # admissible divergence-free initial velocity, including the step
        self.p.fill(0)

    def _pressure_matrix(self):
        f = self.fluid
        ids = np.full(f.shape, -1, int)
        ids[f] = np.arange(f.sum())
        rows, cols, values = [], [], []
        diag = np.zeros(f.sum())
        for axis, h in ((0, self.dy), (1, self.dx)):
            a, b = [slice(None)]*2, [slice(None)]*2
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            ok = f[a] & f[b]
            i, j = ids[a][ok], ids[b][ok]
            for r, col in ((i, j), (j, i)):
                rows.extend(r); cols.extend(col); values.extend(np.full(len(r), -1/h**2))
                np.add.at(diag, r, 1/h**2)
        diag[ids[:, -1][f[:, -1]]] += 2/self.dx**2
        rows.extend(np.arange(len(diag))); cols.extend(np.arange(len(diag))); values.extend(diag)
        self.solve_p = factorized(coo_matrix((values, (rows, cols)), shape=(len(diag),)*2).tocsc())

    def centres(self):
        return (self.u[:, :-1]+self.u[:, 1:])*0.5, (self.v[:-1]+self.v[1:])*0.5

    def divergence(self):
        return np.diff(self.u, axis=1)/self.dx + np.diff(self.v, axis=0)/self.dy

    def project(self, dt):
        self.p[self.fluid] = self.solve_p(-self.divergence()[self.fluid]/dt)
        self.u[:, 1:-1] -= dt*np.diff(self.p, axis=1)/self.dx*self.um[:, 1:-1]
        self.u[:, -1] += dt*2*self.p[:, -1]/self.dx
        self.v[1:-1] -= dt*np.diff(self.p, axis=0)/self.dy*self.vm[1:-1]
        self.u *= self.um
        self.v *= self.vm

    def wall_terms(self, uc, vc):
        fx, fy = np.zeros_like(uc), np.zeros_like(vc)
        kw, ew, prod = np.zeros_like(uc), np.zeros_like(uc), np.zeros_like(uc)
        count = np.zeros_like(uc)
        for axis, h, mask in self.walls:
            tangent = uc if axis == 0 else vc
            speed = np.abs(tangent)
            y = h*0.5
            # Solve smooth log law U/u_tau = log(E*y*u_tau/nu)/kappa.
            ut = np.maximum(np.sqrt(self.c.nu*speed/y), 1e-9)
            for _ in range(8):
                log = np.log(np.maximum(9.8*y*ut/self.c.nu, 1.01))
                residual = ut*log/0.41-speed
                ut = np.maximum(ut-residual/((log+1)/0.41), 1e-9)
            ut = np.where(y*ut/self.c.nu > 11.06, ut, np.sqrt(self.c.nu*speed/y))
            force = -np.sign(tangent)*ut**2/h*mask
            if axis == 0:
                fx += force
            else:
                fy += force
            kw += ut**2/0.3*mask
            ew += ut**3/(0.41*y)*mask
            prod += ut**3/(0.41*y)*mask
            count += mask
        return fx, fy, kw/np.maximum(count, 1), ew/np.maximum(count, 1), prod

    def step(self, remaining):
        c, dx, dy = self.c, self.dx, self.dy
        uc, vc = self.centres()
        nut = np.minimum(0.09*self.k**2/np.maximum(self.e, 1e-12), 0.1*c.speed*c.height)
        viscosity = c.nu+nut
        dt = min(0.35/(np.max(abs(self.u))/dx+np.max(abs(self.v))/dy+1e-9),
                 0.2/(np.max(viscosity)*(1/dx**2+1/dy**2)), remaining)
        fx, fy, kw, ew, wallprod = self.wall_terms(uc, vc)
        # Staggered velocities advect on their own control-volume locations.
        vu = np.pad((vc[:, :-1]+vc[:, 1:])*0.5, ((0, 0), (1, 1)), mode="edge")
        uv = np.pad((uc[:-1]+uc[1:])*0.5, ((1, 1), (0, 0)), mode="edge")
        nuu = np.pad((viscosity[:, :-1]+viscosity[:, 1:])*0.5, ((0, 0), (1, 1)), mode="edge")
        nuv = np.pad((viscosity[:-1]+viscosity[1:])*0.5, ((1, 1), (0, 0)), mode="edge")
        # Variable-viscosity symmetric strain stress: Laplacian term + transpose term.
        duy, dux = np.gradient(uc, dy, dx)
        dvy, dvx = np.gradient(vc, dy, dx)
        dnuty, dnutx = np.gradient(nut, dy, dx)
        fx += dnutx*dux+dnuty*dvx
        fy += dnutx*duy+dnuty*dvy
        fux = np.pad((fx[:, :-1]+fx[:, 1:])*0.5, ((0, 0), (1, 1)), mode="edge")
        fvy = np.pad((fy[:-1]+fy[1:])*0.5, ((1, 1), (0, 0)), mode="edge")
        newu = self.u+dt*(-upwind(self.u, self.u, vu, dx, dy)
                          +diffusion(self.u, nuu, self.um, dx, dy)+fux)
        newv = self.v+dt*(-upwind(self.v, uv, self.v, dx, dy)
                          +diffusion(self.v, nuv, self.vm, dx, dy)+fvy)
        newu[:, 0] = c.speed*self.um[:, 0]
        newu[:, -1] = newu[:, -2]
        self.u, self.v = newu*self.um, newv*self.vm
        self.project(dt)
        production = np.minimum(nut*(2*dux**2+2*dvy**2+(duy+dvx)**2), 10*self.e)
        production[self.wall] = wallprod[self.wall]
        # Conservative donor-cell scalar fluxes with specified inlet values.
        def transport(q, inlet):
            xq = np.pad(q, ((0, 0), (1, 1)), mode="edge")
            xq[:, 0] = inlet
            yq = np.pad(q, ((1, 1), (0, 0)), mode="edge")
            qx = self.u*np.where(self.u >= 0, xq[:, :-1], xq[:, 1:])
            qy = self.v*np.where(self.v >= 0, yq[:-1], yq[1:])
            return -np.diff(qx, axis=1)/dx-np.diff(qy, axis=0)/dy
        oldk = self.k.copy()
        self.k = np.maximum((self.k+dt*(transport(self.k, self.kin)
                    +diffusion(self.k, c.nu+nut, self.fluid, dx, dy)+production))
                    /(1+dt*self.e/np.maximum(oldk, 1e-12)), 1e-10)
        self.e = np.maximum((self.e+dt*(transport(self.e, self.ein)
                    +diffusion(self.e, c.nu+nut/1.3, self.fluid, dx, dy)
                    +1.44*production*self.e/np.maximum(oldk, 1e-12)))
                    /(1+dt*1.92*self.e/np.maximum(oldk, 1e-12)), 1e-10)
        self.k[self.wall] = np.maximum(kw[self.wall], 1e-10)
        self.e[self.wall] = np.maximum(ew[self.wall], 1e-10)
        self.k[~self.fluid], self.e[~self.fluid] = 0, 0
        if not all(np.isfinite(q).all() for q in (self.u, self.v, self.k, self.e)):
            raise FloatingPointError("Solution diverged. Reduce speed or refine the grid.")
        return dt

    def run(self, progress=None, cancel=None):
        times = np.linspace(0, self.c.duration, self.c.frames)
        fields = [[] for _ in range(5)]
        t, steps, maxdiv = 0., 0, 0.
        last, change = None, 1.
        for frame, target in enumerate(times):
            while t < target-1e-12:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("Calculation cancelled")
                t += self.step(target-t)
                steps += 1
                maxdiv = max(maxdiv, float(np.max(abs(self.divergence()[self.fluid]))))
            uc, vc = self.centres()
            if last is not None:
                change = float(np.linalg.norm((uc-last)[self.fluid]) /
                               max(np.linalg.norm(uc[self.fluid]), 1e-12))
            last = uc.copy()
            static = (self.p-2*self.k/3+2*np.mean(self.k[:, -1])/3)*self.c.rho
            for dest, q in zip(fields, (uc, vc, static, self.k, self.e)):
                dest.append(np.where(self.fluid, q, 0).astype(np.float32))
            if progress:
                progress(frame+1, len(times), t)
        inflow = float(self.u[:, 0].sum()*self.dy)
        outflow = float(self.u[:, -1].sum()*self.dy)
        diagnostics = dict(steps=steps, max_divergence=maxdiv,
            mass_imbalance=abs(outflow-inflow)/max(inflow, 1e-12),
            final_frame_velocity_change=change,
            reynolds=self.c.speed*self.c.height/self.c.nu,
            pressure_drop_pa=float(np.mean(static[:, 0][self.fluid[:, 0]])-np.mean(static[:, -1])),
            backflow=bool(np.any(self.u[:, -1] < 0)),
            note="Finite-time RANS result; small divergence alone does not establish steady convergence.")
        return Result(self.c, times, *[np.array(q) for q in fields], self.fluid.copy(), diagnostics)
