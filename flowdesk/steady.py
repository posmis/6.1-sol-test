"""Steady SIMPLE: implicit upwind momentum/scalars, pressure correction, AMG.

The saved two identical frames describe a final field for playback, not a transient.
"""
from dataclasses import replace
import time
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve, bicgstab, cg, splu, LinearOperator
from scipy.ndimage import map_coordinates
import pyamg
from .kernels import assemble, pressure_system
from threadpoolctl import threadpool_limits


class LinearSolves:
    def __init__(self, cancel):
        self.cancel, self.cache = cancel, {}
        self.total_iterations = 0

    def solve(self, matrix, rhs, guess, key, positive=False, refreshed=False):
        if self.cancel is not None and self.cancel.is_set():
            raise InterruptedError("Calculation cancelled")
        if len(rhs) <= 100:
            answer = spsolve(matrix, rhs)
        else:
            entry = self.cache.get(key)
            if entry is None or entry[1] >= 20:
                if len(rhs) <= 20000:
                    lu = splu(matrix.tocsc())
                    preconditioner = LinearOperator(matrix.shape, matvec=lu.solve)
                else:
                    hierarchy = pyamg.ruge_stuben_solver(matrix, max_coarse=64)
                    preconditioner = hierarchy.aspreconditioner(cycle="V")
                entry = [preconditioner, 0]
                self.cache[key] = entry
            entry[1] += 1
            def callback(_):
                self.total_iterations += 1
                if self.cancel is not None and self.cancel.is_set():
                    raise InterruptedError("Calculation cancelled")
            method = cg if positive else bicgstab
            answer, status = method(matrix, rhs, x0=guess, M=entry[0],
                rtol=1e-9 if positive else 1e-8, atol=1e-11, maxiter=400, callback=callback)
            if status != 0:
                self.cache.pop(key, None)
                if not refreshed:
                    return self.solve(matrix, rhs, guess, key, positive, refreshed=True)
                raise RuntimeError(f"SIMPLE linear solve {key} failed (status={status})")
        if not np.isfinite(answer).all():
            raise FloatingPointError(f"Non-finite SIMPLE solution for {key}")
        return answer


def operator(mask, viscosity, ax, ay, reaction, source, kind, fixed, dx, dy,
             conservative=False, u=None, v=None, inlet=0.):
    if u is None:
        u, v = np.empty((1, 1)), np.empty((1, 1))
    row, col, data, rhs = assemble(mask, viscosity, ax, ay, reaction, source,
                                 kind, fixed, dx, dy, conservative, u, v, inlet)
    return coo_matrix((data, (row, col)), shape=(len(rhs),)*2).tocsr(), rhs


def defect(matrix, q, rhs, scale):
    return float(np.linalg.norm(matrix@q-rhs)/max(np.linalg.norm(rhs), scale*np.sqrt(len(rhs)), 1e-14))


def relaxed_solve(linear, matrix, rhs, old, mask, kind, alpha, key):
    diagonal = matrix.diagonal()
    extra = np.where(kind[mask] == 0, diagonal*(1/alpha-1), 0.)
    result = linear.solve(matrix+diags(extra), rhs+extra*old[mask], old[mask], key)
    q = np.zeros(mask.shape)
    q[mask] = result
    d = np.zeros(mask.shape)
    d[mask] = 1/np.maximum(diagonal+extra, 1e-14)
    return q, d


def prolongate(old, name, coarse, fine, shape):
    oy, ox = (0.5, 0.) if name == "u" else (0., 0.5) if name == "v" else (.5, .5)
    y = (np.arange(shape[0])+oy)*coarse.ny/fine.ny-oy
    x = (np.arange(shape[1])+ox)*coarse.nx/fine.nx-ox
    yy, xx = np.meshgrid(y, x, indexing="ij")
    if name == "p":
        # Odd reflection at p=0 boundaries avoids a gauge offset from edge clamping.
        padded = np.pad(old, 1, mode="edge")
        if fine.bottom == "open":
            padded[0, 1:-1] = -old[0]
        if fine.top == "open":
            padded[-1, 1:-1] = -old[-1]
        padded[:, 0] = padded[:, 1]
        padded[:, -1] = -padded[:, -2]
        return map_coordinates(padded, [yy+1, xx+1], order=1, mode="nearest")
    return map_coordinates(old, [yy, xx], order=1, mode="nearest")


def warm_start(solver, cancel, progress):
    """Solve a small version first and interpolate each staggered field separately."""
    from .solver import Solver
    c = solver.c
    factor = max(c.nx/96, c.ny/32)
    if not c.coarse_start or factor <= 1.5:
        return None
    coarse_config = replace(c, nx=max(16, round(c.nx/factor)), ny=max(8, round(c.ny/factor)),
                            coarse_start=False, max_iterations=min(c.max_iterations, 200),
                            tolerance=max(c.tolerance, 5e-4))
    coarse = Solver(coarse_config)
    result = solve(coarse, cancel=cancel,
                   progress=(lambda i, n, t: progress(i, n, -1.) if i % 10 == 0 else None) if progress else None)
    for name in ("u", "v", "p", "k", "e"):
        old = getattr(coarse, name)
        shape = getattr(solver, name).shape
        setattr(solver, name, prolongate(old, name, coarse_config, c, shape))
    solver.u *= solver.um
    solver.v *= solver.vm
    solver.u[:, 0] = c.speed*solver.um[:, 0]
    solver.k[~solver.fluid] = 0
    solver.e[~solver.fluid] = 0
    pressure = solver.p.copy()
    solver.project(1.)
    solver.p = pressure
    return dict(grid=[coarse_config.nx, coarse_config.ny], iterations=result.diagnostics["iterations"],
                converged=result.diagnostics["converged"])


@threadpool_limits.wrap(limits=1, user_api="blas")
def solve(s, progress=None, cancel=None):
    from .solver import Result
    started = time.perf_counter()
    c, f, dx, dy = s.c, s.fluid, s.dx, s.dy
    if progress:
        progress(0, c.max_iterations, 0.)
    coarse = warm_start(s, cancel, progress)
    linear = LinearSolves(cancel)
    ukind, vkind = np.zeros(s.u.shape, np.int32), np.zeros(s.v.shape, np.int32)
    ukind[:, 0], ukind[:, -1] = 1, 3
    if c.bottom == "open":
        vkind[0] = 4
    if c.top == "open":
        vkind[-1] = 5
    skind = np.where(s.wall, 1, 0).astype(np.int32)
    acceleration = c.speed**2/c.length
    history, successful, converged = [], 0, False
    for iteration in range(1, c.max_iterations+1):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Calculation cancelled")
        uc, vc = s.centres()
        nut = np.minimum(.09*s.k**2/np.maximum(s.e, 1e-12), .1*c.speed*c.height)
        visc = c.nu+nut
        fx, fy, kw, ew, wp = s.wall_terms(uc, vc)
        duy, dux = np.gradient(uc, dy, dx)
        dvy, dvx = np.gradient(vc, dy, dx)
        dny, dnx = np.gradient(nut, dy, dx)
        fx += dnx*dux+dny*dvx
        fy += dnx*duy+dny*dvy
        def uface(q):
            return np.pad((q[:, :-1]+q[:, 1:])*.5, ((0, 0), (1, 1)), mode="edge")
        def vface(q):
            return np.pad((q[:-1]+q[1:])*.5, ((1, 1), (0, 0)), mode="edge")
        gradx, grady = np.zeros_like(s.u), np.zeros_like(s.v)
        gradx[:, 1:-1] = np.diff(s.p, axis=1)/dx
        grady[1:-1] = np.diff(s.p, axis=0)/dy
        ax, ay = uface(fx), vface(fy)
        # Implicit negative source linearization, exact at the current iterate.
        dampu = np.maximum(-ax*s.u/(s.u**2+1e-20), 0.)
        dampv = np.maximum(-ay*s.v/(s.v**2+1e-20), 0.)
        au, bu = operator(s.um, uface(visc), s.u, uface(vc), dampu,
                          ax+dampu*s.u-gradx, ukind, np.full(s.u.shape, c.speed), dx, dy)
        av, bv = operator(s.vm, vface(visc), vface(uc), s.v, dampv,
                          ay+dampv*s.v-grady, vkind, np.zeros_like(s.v), dx, dy)
        residual = dict(u=defect(au, s.u[s.um], bu, acceleration),
                        v=defect(av, s.v[s.vm], bv, acceleration))
        s.u, du = relaxed_solve(linear, au, bu, s.u, s.um, ukind, .65, "u")
        s.v, dv = relaxed_solve(linear, av, bv, s.v, s.vm, vkind, .65, "v")
        du[:, 0] = 0
        du[:, -1] = du[:, -2]
        if c.bottom == "open":
            dv[0] = dv[1]
        if c.top == "open":
            dv[-1] = dv[-2]
        row, col, data = pressure_system(f, du, dv, dx, dy, c.bottom == "open", c.top == "open")
        ap = coo_matrix((data, (row, col)), shape=(int(f.sum()),)*2).tocsr()
        correction = np.zeros(f.shape)
        correction[f] = linear.solve(ap, -s.divergence()[f], np.zeros(f.sum()), "p", positive=True)
        s.u[:, 1:-1] -= du[:, 1:-1]*np.diff(correction, axis=1)/dx*s.um[:, 1:-1]
        s.u[:, -1] += du[:, -1]*2*correction[:, -1]/dx
        s.v[1:-1] -= dv[1:-1]*np.diff(correction, axis=0)/dy*s.vm[1:-1]
        if c.bottom == "open":
            s.v[0] -= dv[0]*2*correction[0]/dy*s.vm[0]
        if c.top == "open":
            s.v[-1] += dv[-1]*2*correction[-1]/dy*s.vm[-1]
        s.u *= s.um
        s.v *= s.vm
        s.p += .25*correction
        production = np.minimum(nut*(2*dux**2+2*dvy**2+(duy+dvx)**2), 10*s.e)
        production[s.wall] = wp[s.wall]
        oldk = np.maximum(s.k, 1e-10)
        for key, q, diffusion, sink, source, fixed, inlet, scale in (
            ("k", s.k, c.nu+nut, s.e/oldk, production, np.maximum(kw, 1e-10), s.kin, s.kin*c.speed/c.length),
            ("e", s.e, c.nu+nut/1.3, 1.92*s.e/oldk, 1.44*production*s.e/oldk,
             np.maximum(ew, 1e-10), s.ein, s.ein*c.speed/c.length)):
            a, b = operator(f, diffusion, uc, vc, sink, source, skind, fixed, dx, dy,
                            conservative=True, u=s.u, v=s.v, inlet=inlet)
            residual[key] = defect(a, q[f], b, scale)
            new, _ = relaxed_solve(linear, a, b, q, f, skind, .7, key)
            new[f] = np.maximum(new[f], 1e-10)
            setattr(s, key, new)
        residual["continuity"] = float(np.max(abs(s.divergence()[f]))*c.length/c.speed)
        if not all(np.isfinite(q).all() for q in (s.u, s.v, s.p, s.k, s.e)):
            raise FloatingPointError("SIMPLE diverged")
        history.append(dict(iteration=iteration, **residual))
        successful = successful+1 if max(residual.values()) < c.tolerance else 0
        if progress:
            progress(iteration, c.max_iterations, 0.)
        if iteration >= 10 and successful >= 3:
            converged = True
            break
    uc, vc = s.centres()
    static = (s.p-2*s.k/3+2*np.mean(s.k[:, -1][f[:, -1]])/3)*c.rho
    fields = [np.repeat(np.where(f, q, 0)[None].astype(np.float32), 2, axis=0)
              for q in (uc, vc, static, s.k, s.e)]
    inlet = float(s.u[:, 0].sum()*dy)
    outlet = float(s.u[:, -1].sum()*dy)
    side = float((s.v[-1].sum()-s.v[0].sum())*dx)
    diagnostics = dict(mode="steady", converged=converged, iterations=iteration, residuals=residual,
        residual_history=history, coarse_start=coarse, linear_iterations=linear.total_iterations,
        wall_seconds=time.perf_counter()-started, steps=0, max_divergence=residual["continuity"]*c.speed/c.length,
        mass_imbalance=abs(outlet+side-inlet)/max(inlet, 1e-12), side_flow_m2_s=side,
        stored_frames=2, pressure_solver="SIMPLE / sparse-LU + AMG", final_frame_velocity_change=0.,
        reynolds=c.speed*c.height/c.nu,
        pressure_drop_pa=float(np.mean(static[:, 0][f[:, 0]])-np.mean(static[:, -1][f[:, -1]])),
        backflow=bool(np.any(s.u[:, -1] < 0)),
        note="Two identical final fields for playback; time is animation time, not a simulated transient.")
    return Result(c, np.array([0., c.duration]), *fields, f.copy(), diagnostics)
